import argparse
import lmdb
from rdkit import Chem
import pandas as pd
import numpy as np
import torch
import csv
import os
import pickle
import re
import pdb
from tqdm import tqdm, trange
from concurrent.futures import ProcessPoolExecutor
from rdkit import RDLogger
'''
aroma: [B, L]
e: [B, L]
b: [B, L, 4]
c: [B, L]
m: [B, L]
'''
MAX_BONDS = 6
MAX_DIFF = 4
prefix = "data"
mp = True


def remap_atoms(src_smiles, tgt_smiles):
    """
    Remap atoms in target SMILES starting from 1, and then map source SMILES accordingly.
    Also add mapping numbers to unmapped atoms in source SMILES.
    Returns remapped SMILES strings and the mapping dictionary.
    
    Args:
        src_smiles (str): Source SMILES string
        tgt_smiles (str): Target SMILES string
        
    Returns:
        tuple: (remapped_src, remapped_tgt, mapping_dict)
    """
    # Extract atom mapping numbers from target
    pattern = re.compile(r':(\d+)\]')
    tgt_maps = pattern.findall(tgt_smiles)
    tgt_maps = [int(x) for x in tgt_maps]
    
    # Create mapping dictionary starting from 1
    mapping_dict = {}
    for i, old_map in enumerate(sorted(set(tgt_maps)), 1):
        mapping_dict[old_map] = i
    
    # Get all atoms from source that need mapping
    src_maps = pattern.findall(src_smiles)
    src_maps = [int(x) for x in src_maps]
    
    # Add new mappings for atoms in source but not in target
    next_map = max(mapping_dict.values()) + 1
    for old_map in src_maps:
        if old_map not in mapping_dict:
            mapping_dict[old_map] = next_map
            next_map += 1
    
    # Create reverse mapping for easier replacement
    reverse_mapping = {str(k): str(v) for k, v in mapping_dict.items()}
    
    # Replace mappings in target
    remapped_tgt = tgt_smiles
    for old, new in reverse_mapping.items():
        remapped_tgt = remapped_tgt.replace(f':{old}]', f':{new}]')
    
    # Replace mappings in source and add mapping to unmapped atoms
    remapped_src = src_smiles
    for old, new in reverse_mapping.items():
        remapped_src = remapped_src.replace(f':{old}]', f':{new}]')
    
    # Add mapping to unmapped atoms in source
    mol = Chem.MolFromSmiles(remapped_src)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:  # if atom has no mapping
                atom.SetAtomMapNum(next_map)
                next_map += 1
        remapped_src = Chem.MolToSmiles(mol, isomericSmiles=True)
    
    return remapped_src, remapped_tgt, mapping_dict

def verify_mapping(src_smiles, tgt_smiles, remapped_src, remapped_tgt, mapping_dict):
    """
    Verify that the remapping is correct by checking if the molecules are equivalent.
    
    Args:
        src_smiles (str): Original source SMILES
        tgt_smiles (str): Original target SMILES
        remapped_src (str): Remapped source SMILES
        remapped_tgt (str): Remapped target SMILES
        mapping_dict (dict): Mapping dictionary
        
    Returns:
        bool: True if mapping is correct, False otherwise
    """
    # Remove atom mapping for comparison
    def remove_mapping(smiles):
        return re.sub(r':\d+\]', ']', smiles)
    
    # Check if original molecules are equivalent to remapped ones
    src_mol = Chem.MolFromSmiles(remove_mapping(src_smiles))
    remapped_src_mol = Chem.MolFromSmiles(remove_mapping(remapped_src))
    
    tgt_mol = Chem.MolFromSmiles(remove_mapping(tgt_smiles))
    remapped_tgt_mol = Chem.MolFromSmiles(remove_mapping(remapped_tgt))
    
    if src_mol is None or remapped_src_mol is None or tgt_mol is None or remapped_tgt_mol is None:
        return False
    
    return (Chem.MolToSmiles(src_mol) == Chem.MolToSmiles(remapped_src_mol) and 
            Chem.MolToSmiles(tgt_mol) == Chem.MolToSmiles(remapped_tgt_mol))



def molecule(mols, src_len, reactant_mask = None, ranges = None):
    features = {}
    element = np.zeros(src_len, dtype='int32')
    aroma = np.zeros(src_len, dtype='int32')
    bonds = np.zeros((src_len, MAX_BONDS), dtype='int32')
    charge = np.zeros(src_len, dtype='int32')
    
    reactant = np.zeros(src_len, dtype='int32') # 1 for reactant
    mask = np.ones(src_len, dtype='int32') # 1 for masked
    segment = np.zeros(src_len, dtype='int32')

    for molid, mol in enumerate(mols):
        for atom in mol.GetAtoms():
            idx = atom.GetAtomMapNum()-1
            if idx >= src_len or idx < 0:
                print(f"Warning: Invalid atom map number {idx+1} for molecule {molid}")
                continue

            segment[idx] = molid
            element[idx] = atom.GetAtomicNum()
            charge[idx] = atom.GetFormalCharge()
            mask[idx] = 0
            if reactant_mask:
                reactant[idx] = reactant_mask[molid]

            cnt = 0
            for j, b in enumerate(atom.GetBonds()):
                other = b.GetBeginAtomIdx() + b.GetEndAtomIdx() - atom.GetIdx()
                other = mol.GetAtoms()[other].GetAtomMapNum() - 1
                if other >= src_len or other < 0:
                    print(f"Warning: Invalid bond connection to atom {other+1}")
                    continue
                    
                num_map = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'AROMATIC': 1}
                num = num_map[str(b.GetBondType())]
                for k in range(num):
                    if cnt == MAX_BONDS:
                        return None
                    bonds[idx][cnt] = other
                    cnt += 1 
                if str(b.GetBondType()) == 'AROMATIC':
                    aroma[idx] = 1
            tmp = bonds[idx][0:cnt]
            tmp.sort()
            bonds[idx][0:cnt] = tmp
            while cnt < MAX_BONDS:
                bonds[idx][cnt] = idx
                cnt += 1
            
    features = {'element':element, 'bond':bonds, 'charge':charge, 'aroma':aroma, 'mask':mask, 'segment':segment, 'reactant': reactant}
    return features


def reaction(args):
    """ processes a reaction, returns dict of arrays"""
    src, tgt = args
    pattern = re.compile(":(\d+)\]") # atom map numbers
    src_mol = Chem.MolFromSmiles(src)
    if src_mol is None:
        return None
    src_len = src_mol.GetNumAtoms()

    remap_src, remap_tgt, mapping_dict = remap_atoms(src, tgt)
    is_correct = verify_mapping(src, tgt, remap_src, remap_tgt, mapping_dict)
    if not is_correct:
        return None
    src = remap_src
    tgt = remap_tgt

    # reactant mask
    src_mols = src.split('.')
    tgt_atoms = pattern.findall(tgt)
    reactant_mask = [False for i in src_mols]
    for j, item in enumerate(src_mols):
        atoms = pattern.findall(item)
        for atom in atoms:
            if atom in tgt_atoms:
                reactant_mask[j] = True
                break  
                
    # the atom map num ranges of each molecule for segment mask
    src_mols = [Chem.MolFromSmiles(item) for item in src_mols]
    tgt_mols = [Chem.MolFromSmiles(item) for item in tgt.split(".")]
    if any(src_mol is None for src_mol in src_mols) or any(tgt_mol is None for tgt_mol in tgt_mols):
        return None
    ranges = []
    for mol in src_mols:
        lower = 999
        upper = 0
        for atom in mol.GetAtoms():
            lower = min(lower, atom.GetAtomMapNum()-1)
            upper = max(upper, atom.GetAtomMapNum())
        ranges.append((lower, upper))
    
    src_features = molecule(src_mols, src_len, reactant_mask, ranges)
    tgt_features = molecule(tgt_mols, src_len)
    
    
    if not (src_features and tgt_features):
        return None
                
    src_bond = src_features['bond']
    tgt_bond = tgt_features['bond']
    bond_inc = np.zeros((src_len, MAX_DIFF), dtype='int32')
    bond_dec = np.zeros((src_len, MAX_DIFF), dtype='int32')
    for i in range(src_len):
        if tgt_features['mask'][i]:
            continue
        inc_cnt = 0
        dec_cnt = 0
        diff = [0 for _ in range(src_len)]
        for j in range(MAX_BONDS):
            diff[tgt_bond[i][j]] += 1
            diff[src_bond[i][j]] -= 1
        for j in range(src_len):
            if diff[j] > 0:
                if inc_cnt + diff[j] >MAX_DIFF:
                    return None
                bond_inc[i][inc_cnt:inc_cnt+diff[j]] = j
                inc_cnt += diff[j]
            if diff[j] < 0:
                bond_dec[i][dec_cnt:dec_cnt-diff[j]] = j
                dec_cnt -= diff[j]
        assert inc_cnt == dec_cnt
    if (bond_inc<0).sum() > 0 or (bond_dec<0).sum() > 0:
        print('wrong!', src, tgt)
        return None
    if (src_features['bond'] < 0).sum() > 0 or (tgt_features['bond'] < 0).sum() > 0:
        print('wrong!', src, tgt)
        return None
    item = {}
    for key in src_features:
        if key in ["element"]:
            item[key] = src_features[key]
        elif key == 'reactant':
            item['reactant_flag'] = src_features[key]
        else:
            item['reactant_' + key] = src_features[key]
            item['product_' + key] = tgt_features[key]
    return item


def process(name):
    tgt = []
    src = []
    with open(name + ".txt") as file:
        for line in file:
            rxn = line.split()[0].split('>>')
            src.append(rxn[0])
            tgt.append(rxn[1])
    dataset = []   
    if mp:
        pool = ProcessPoolExecutor(10)
        batch_size = 2048
        for i in trange(len(src)//batch_size+1):
            upper = min((i+1)*batch_size, len(src))
            arg_list = [(src[idx], tgt[idx]) for idx in range(i*batch_size, upper)]
            result = pool.map(reaction, arg_list, chunksize= 64)
            result = list(result)  
            for item in result:
                if not item is None:
                    dataset += [item]        
        pool.shutdown()
    else:
        for i in trange(len(src)):
            item = reaction((src[i], tgt[i]))
            if not item is None:
                dataset += [item]

    with open(name +"_"+prefix+ '.pickle', 'wb') as file:
        print('save to', name +"_"+prefix+ '.pickle')
        pickle.dump(dataset, file)
    print("total %d, legal %d"%(len(src), len(dataset)))
    print(name, 'file saved.')

if __name__ =='__main__':
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    RDLogger.DisableLog('rdApp.info') 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data/uspto50k")
    args = parser.parse_args()
   
    process(args.data_path + "/train")
    process(args.data_path + "/val")
    process(args.data_path + "/test")