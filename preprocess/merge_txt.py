import os

import pandas as pd
import tqdm 
import re
from rdkit import Chem

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
    

with open('/fs_mol/linhaitao/synflow_mix/data/pistachio23/train.txt', 'r', encoding='utf-8') as f:
    reactions = [line.strip() for line in f if line.strip()]
reactions

with open('/fs_mol/linhaitao/synflow_mix/data/usptomit/train.txt', 'r', encoding='utf-8') as f:
    reactions_usptomit = [line.strip() for line in f if line.strip()]
for reaction in reactions_usptomit:
    reactions.append(reaction)

reaction_noatommap = []
for reaction in tqdm.tqdm(reactions, desc="Removing atom mapping"):
    try:
        reaction_noatommap.append(clean_smiles(reaction))
    except:
        pass

reaction_tests = []
with open('/fs_mol/linhaitao/synflow_mix/data/usptomit/test.txt', 'r', encoding='utf-8') as f:
    reactions_usptomit = [line.strip() for line in f if line.strip()]


with open('/fs_mol/linhaitao/synflow_mix/data/pistachio23/test.txt', 'r', encoding='utf-8') as f:
    reactions_pistachio = [line.strip() for line in f if line.strip()]


filter_final_tests = reactions_pistachio
for reaction_test in tqdm.tqdm(reactions_usptomit, desc="Filtering test reactions"):
    try:
        clean_reaction_test = clean_smiles(reaction_test)
        if clean_reaction_test not in reaction_noatommap:
            filter_final_tests.append(reaction_test)
    except:
        filter_final_tests.append(reaction_test)

with open(os.path.join('/fs_mol/linhaitao/synflow_mix/data/pistachioaisi/', 'train.txt'), 'w', encoding='utf-8') as f:
    f.write("\n".join(reactions))

with open(os.path.join('/fs_mol/linhaitao/synflow_mix/data/pistachioaisi/', 'test.txt'), 'w', encoding='utf-8') as f:
    f.write("\n".join(filter_final_tests))