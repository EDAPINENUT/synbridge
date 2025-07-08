import rdkit.Chem as Chem
from synflow.chem.constants import *
import torch
from rdkit.Chem import Atom
import os
import numpy as np
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
MAX_BONDS=6
def get_valence_electrons(atomic_num):
    """
    
    Args:
        atomic_num
    Returns:
        outer_electrons
    """
    periodic_table = Chem.GetPeriodicTable()
        
    n_outer_electrons = periodic_table.GetNOuterElecs(atomic_num)
    
    return n_outer_electrons


def calculate_formal_charge(atomic_num, num_h, adjacency):
    pt = Chem.GetPeriodicTable()
    n_outer_elecs = pt.GetNOuterElecs(atomic_num)
    
    total_bond_order = sum(adjacency) + num_h
    
    bonding_electron_pairs = total_bond_order
    
    lone_pairs = 4 - bonding_electron_pairs
    if lone_pairs < 0:
        lone_pairs = 0  
    formal_charge = n_outer_elecs - (lone_pairs * 2) - total_bond_order
    
    return int(formal_charge)


def reconstruct_molecule(atom_type, adjacency, atom_mask, num_charge=None):
    rdmol = Chem.RWMol()
    # Convert atom types to actual atom symbols
    atoms = [get_atom_type(int(t)) for t in atom_type]
    
    # Get indices where bonds exist (non-zero values in adjacency matrix)
    bond_indices = torch.nonzero(adjacency)
    
    # Only keep bonds where both atoms are valid (masked)
    valid_bonds = torch.logical_and(
        atom_mask[bond_indices[:, 0]], 
        atom_mask[bond_indices[:, 1]]
    )
    bond_indices = bond_indices[valid_bonds]
    
    valid_direction = bond_indices[:, 0] < bond_indices[:, 1]
    bond_indices = bond_indices[valid_direction]

    if num_charge is not None:
        for i, atom in enumerate(atoms):
            if atom_mask[i]:
                rd_atom = Atom(atom)
                charge = get_charge_class(int(num_charge[i]))
                rd_atom.SetFormalCharge(charge)
                rdmol.AddAtom(rd_atom)

    else:
        for i, atom in enumerate(atoms):
            if atom_mask[i]:
                rdmol.AddAtom(Atom(atom))
    for i, j in bond_indices:
        rdmol.AddBond(int(i), int(j), get_bond_type(int(adjacency[i, j])))
    # Get the largest fragment
    frags = Chem.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
    if len(frags) > 0:
        largest_frag = max(frags, key=lambda m: m.GetNumAtoms())
        return largest_frag
    return rdmol

def get_canonical_smiles(mol, isomericSmiles=False):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    
    if mol is None:
        return None
        
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    
    rec_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if rec_mol is None:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomericSmiles)
    
    return Chem.MolToSmiles(rec_mol, canonical=True, isomericSmiles=isomericSmiles)



def mol2array(mol):
    img = Draw.MolToImage(mol, kekulize=False)
    array = np.array(img)[:, :, 0:3]
    return array

def check(smile):
    smile = smile.split('.')
    smile.sort(key = len)
    try:
        mol = Chem.MolFromSmiles(smile[-1], sanitize=False)
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except Exception:
        return False

def mol2file(m, name):
    AllChem.Compute2DCoords(m)
    img = Draw.MolToImage(m)
    Draw.MolToFile(m, os.path.join('./img', name))


def result2mol(args): # for threading
    element, mask, bond, aroma, charge, reactant = args
    # [L], [L], [L, 4], [l], [l]
    mask = mask.ne(1).cpu()
    cur_len = sum(mask.long())
    l = element.shape[0]

    mol = Chem.RWMol()
    if isinstance(element, torch.Tensor):   
        element = element.cpu().numpy().tolist()
    if isinstance(charge, torch.Tensor):
        charge = charge.cpu().numpy().tolist()
  
    
    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(l):
        if mask[i] == False:
            continue
        a = Chem.Atom(element[i])
        if not reactant is None and reactant[i]:
            a.SetAtomMapNum(i+1)
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx
        if len(bond.shape) == 3:
            bond = bond.argmax(dim=-1)
        bond_indices = torch.nonzero(bond)
        valid_bonds = torch.logical_and(
            mask[bond_indices[:, 0]], 
            mask[bond_indices[:, 1]]
        )
        bond_indices = bond_indices[valid_bonds]
        
        valid_direction = bond_indices[:, 0] < bond_indices[:, 1]
        bond_indices = bond_indices[valid_direction]
    for i, j in bond_indices:
        i, j = int(i), int(j)
        if bond[i, j] == 1:
            if aroma[i] == aroma[j] and aroma[i] > 0:
                mol.AddBond(node_to_idx[i], node_to_idx[j], Chem.rdchem.BondType.AROMATIC)
            else:
                mol.AddBond(node_to_idx[i], node_to_idx[j], Chem.rdchem.BondType.SINGLE)
        elif bond[i, j] == 2:
            mol.AddBond(node_to_idx[i], node_to_idx[j], Chem.rdchem.BondType.DOUBLE)
        elif bond[i, j] == 3:
            mol.AddBond(node_to_idx[i], node_to_idx[j], Chem.rdchem.BondType.TRIPLE)

    for i, item in enumerate(charge):
        if mask[i] == False:
            continue
        if not item == 0:
            atom = mol.GetAtomWithIdx(node_to_idx[i])
            atom.SetFormalCharge(item)
    # Convert RWMol to Mol object
    mol = mol.GetMol() 
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        Chem.MolToSmiles(mol)
    except Exception:
        return None, None, False
    smile = Chem.MolToSmiles(mol)
    return mol, smile, check(smile)



def evaluate_single_sample(sample, topk=100):
    sample_num = len(sample['src_masks'])
    acc_smiles = []
    eval_num = min(topk, sample_num)
    for i in range(eval_num):
        pred_bonds = sample['pred_bonds'][i]
        pred_aromas = sample['pred_aromas'][i]
        pred_charges = sample['pred_charges'][i]
        true_bonds = sample['true_bonds'][i]
        true_aromas = sample['true_aromas'][i]
        true_charges = sample['true_charges'][i]
        src_flags = sample['src_flags'][i]
        src_masks = sample['src_masks'][i]
        tgt_masks = sample['tgt_masks'][i]
        if 'elements' in sample:
            pred_elements = sample['elements'][i]
            true_elements = sample['elements'][i]
        else:
            pred_elements = sample['pred_elements'][i]
            true_elements = sample['true_elements'][i]
        _, pred_s, pred_valid = result2mol((pred_elements, src_masks, 
                                            pred_bonds, pred_aromas,
                                            pred_charges, src_flags))
        _, tgt_s, tgt_valid = result2mol((true_elements, tgt_masks, 
                                          true_bonds, true_aromas,
                                          true_charges, src_flags))
        if tgt_s is None:
            acc_smiles.append(np.nan)
        elif pred_s is None:
            acc_smiles.append(0.0)
        elif tgt_s in pred_s:
            acc_smiles.append(1.0)
        else:
            acc_smiles.append(0.0)
    return acc_smiles


def evaluate_single_smiles(sample, mode='forward'):
    pred_bonds = sample['pred_bonds']
    pred_aromas = sample['pred_aromas']
    pred_charges = sample['pred_charges']
    true_bonds = sample['true_bonds']
    true_aromas = sample['true_aromas']
    true_charges = sample['true_charges']
    src_flags = sample['src_flags']
    src_masks = sample['src_masks']
    tgt_masks = sample['tgt_masks']
    if 'elements' in sample:
        pred_elements = sample['elements']
        true_elements = sample['elements']
    else:
        pred_elements = sample['pred_elements']
        true_elements = sample['true_elements']
    _, pred_s, pred_valid = result2mol((pred_elements, src_masks, 
                                        pred_bonds, pred_aromas,
                                        pred_charges, src_flags))
    _, tgt_s, tgt_valid = result2mol((true_elements, tgt_masks, 
                                      true_bonds, true_aromas,
                                      true_charges, src_flags))
    if mode == 'forward':
        if tgt_s is None:
            return np.nan
        elif pred_s is None:
            return 0.0
        else:
            tgt_split = tgt_s.split('.')
            pred_split = pred_s.split('.')
            for item in tgt_split:
                if item not in pred_split:
                    return 0.0
            return 1.0
    elif mode == 'reverse':
        if tgt_s is None:
            return np.nan
        elif pred_s is None:
            return 0.0
        else:
            tgt_split = tgt_s.split('.')
            pred_split = pred_s.split('.')
            tgt_split = sorted(tgt_split)
            pred_split = sorted(pred_split)
            if tgt_split == pred_split:
                return 1.0
            return 0.0