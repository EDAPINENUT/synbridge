import re
from rdkit import Chem
from multiprocessing import Pool, cpu_count
import os

def extract_year_from_id(id_str):
    """从id中跳过英文字符，提取前四个数字作为年份"""
    # 跳过所有英文字符，找到第一个数字开始的位置
    match = re.search(r'[a-zA-Z]*(\d{4})', id_str)
    if match:
        return int(match.group(1))
    return None

def remove_atom_mapping(mol):
    """Remove atom mapping from molecule"""
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol

def clean_smiles(reaction):
    # 使用正则表达式分割，避免分割配位键中的 -> 或 <-
    parts = re.split(r'(?<![-<])>(?!-)', reaction)
    reactant, condition, product = parts[0], parts[1], parts[2]
    reactants = reactant.split(".")
    reactant_clean = []
    for reactant in reactants:
        mol = Chem.MolFromSmiles(reactant)
        mol = remove_atom_mapping(mol)
        smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True, allHsExplicit=False)
        reactant_clean.append(smi)
    reactant = ".".join(reactant_clean)
    mol = Chem.MolFromSmiles(product)
    mol = remove_atom_mapping(mol)
    product = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True, allHsExplicit=False)
    return reactant + ">" + condition + ">" + product
    

def extract_one_line(line):
    data = line.split()
    data_dict = {}
    smi = data[0]
    data_dict['smiles_raw'] = smi
    data_dict['smiles'] = clean_smiles(smi)
    data_dict['label'] = data[-2]
    data_dict['id'] = data[2]
    data_dict['year'] = extract_year_from_id(data[2])
    return data_dict

def extract_one_line_safe(line):
    """Wrapper function for multiprocessing with error handling"""
    try:
        return extract_one_line(line)
    except Exception as e:
        print(f"Error processing {line}: {e}")
        return None

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default='/instrument/ReRP/dataset/pistachio_zhen/pistachio_2023/pistachio.smi')
    parser.add_argument("--use_multi_process", type=bool, default=True)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--output_dir", type=str, default='/fs_mol/linhaitao/synflow_mix/data/all')
    args = parser.parse_args()
    file_path = args.file_path
    use_multi_process = args.use_multi_process
    year = args.year

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    import tqdm 
    if use_multi_process:
        num_processes = cpu_count()  # 使用所有可用CPU核心
        with Pool(processes=num_processes) as pool:
            extracted_lines = list(tqdm.tqdm(
                pool.imap(extract_one_line_safe, lines),
                total=len(lines),
                desc="Processing lines"
            ))
    else:
        extracted_lines = [extract_one_line_safe(line) for line in tqdm.tqdm(lines)]

    extracted_lines = [line for line in extracted_lines if line is not None]
    import json

    with open(os.path.join(args.output_dir, "extracted_lines.json"), "w", encoding="utf-8") as f:
        json.dump(extracted_lines, f, indent=4, ensure_ascii=False)

    train_data = []
    test_data = []
    train_index_mapping = []
    test_index_mapping = []
    def remove_condition(smiles):
        parts = re.split(r'(?<![-<])>(?!-)', smiles)
        reactant, condition, product = parts[0], parts[1], parts[2]
        return reactant + ">>" + product
        
    for line in tqdm.tqdm(extracted_lines):
        try:
            if line['year'] >= year:
                test_data.append(remove_condition(line['smiles']))
                test_index_mapping.append(line['id'])
            else:
                train_data.append(remove_condition(line['smiles']))
                train_index_mapping.append(line['id'])
        except:
            train_data.append(remove_condition(line['smiles']))
            train_index_mapping.append(line['id'])


    with open(os.path.join(args.output_dir, f"train_data_{year}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_data))

    with open(os.path.join(args.output_dir, f"test_data_{year}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_data))

    with open(os.path.join(args.output_dir, f"train_mapping_{year}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_index_mapping))

    with open(os.path.join(args.output_dir, f"test_mapping_{year}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_index_mapping))


        
    

