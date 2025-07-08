import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from .constants import *
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os
from syndiff.chem.mol import Molecule
from syndiff.chem.reaction import Reaction
import logging
from datetime import datetime
import re
def setup_logger(name=os.environ.get('LOG_DIR', 'logs')):
    # 创建logs目录（如果不存在）
    log_dir = os.path.join('./data', name)
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建带时间戳的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'reaction_parser_{timestamp}.log')
    
    # 配置日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()


def parse_reaction_rdmol(rdmol: Chem.Mol, 
                         kekulize: bool = False, 
                         remove_hs: bool = False) -> dict:
    """Parse rdmol
    
    Args:
        rdmol (rdkit.Chem.Mol): RDKit molecule
        kekulize (bool): Whether to kekulize the molecule
        remove_hs (bool): Whether to remove hydrogens
        
    Returns:
        dict: Dictionary containing atom information with keys:
            - smiles: SMILES string
            - element: Atomic numbers
            - atom_types: Atom types
            - atom_coords: Atom coordinates
            - bond_index: Bond indices
            - bond_types: Bond types
            - center_of_mass: Center of mass
            - atom_features: Atom features
            - hybridization: Atom hybridization
    """

    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    atom_index_to_map_num = {}
    for atom in rdmol.GetAtoms():
        atom_index_to_map_num[atom.GetIdx()] = atom.GetAtomMapNum()
    try:
        Chem.SanitizeMol(rdmol)
    except:
        logger.error(f'Failed to sanitize molecule {Chem.MolToSmiles(rdmol)}')
        return None
    
    if remove_hs:
        rdmol = Chem.RemoveHs(rdmol)
    if kekulize:
        Chem.Kekulize(rdmol)

    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(atom_families)], dtype=np.longlong)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), atom_families_id[feat.GetFamily()]] = 1

    # Get hybridization
    hybridization = []
    for atom in rdmol.GetAtoms():
        hybr = str(atom.GetHybridization())
        hybridization.append(get_hybridization_type_id(hybr))

    ptable = Chem.GetPeriodicTable()
    if rdmol.GetNumConformers() > 0:
        pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float32)
    else:
        pos = np.zeros((rd_num_atoms, 3), dtype=np.float32)
    
    # Process atoms
    element = []
    atom_type = []
    h_numbers = []
    charges = []
    accum_pos = 0
    accum_mass = 0
    seq_pos = []
    for atom_idx in range(rd_num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        atom_symbol = atom.GetSymbol()
        if atom_symbol not in atom_type_id:
            logger.error(f"Atom symbol {atom_symbol} not found in atom_type_id")
            return None
        atom_type.append(get_atom_type_id(atom_symbol))
        element.append(atom_num)
        atom_weight = ptable.GetAtomicWeight(atom_num)
        accum_pos += pos[atom_idx] * atom_weight
        accum_mass += atom_weight
        h_numbers.append(atom.GetTotalNumHs())
        charges.append(get_charge_class_id(atom.GetFormalCharge()))
        seq_pos.append(atom_idx)
    center_of_mass = accum_pos / accum_mass
    element = np.array(element, dtype=np.int64)
    

    return {
        'smiles':                Chem.MolToSmiles(rdmol),
        'atom_index_to_map_num': atom_index_to_map_num,
        'elements':              element,
        'atom_types':            atom_type,
        'atom_coords':           pos,
        'center_of_mass':        center_of_mass,
        'atom_features':         feat_mat,
        'hybridizations':        hybridization,
        'h_numbers':             h_numbers,
        'charges':               charges,
        'seq_pos':               seq_pos,
    }

def check_structure_match(smiles, prop_str):
    """
    检查 SMILES 是否与 prop_string 中的原子顺序匹配（不考虑 Atom Map）
    """
    clean_smiles = re.sub(r":\d+", "", smiles)  # 去除 `:N` 映射编号
    clean_prop = re.sub(r"</\d{4}]", "", prop_str)  # 移除 `</xxxx]`
    clean_prop = re.sub(r"[\[\]@=()]", "", clean_prop)  # 去除 RDKit 解析不需要的符号
    return clean_smiles == clean_prop

def set_atom_fsq(mol, fsq):
    atom_props = {}
    for idx, atom in enumerate(mol.GetAtoms()):
        if idx < len(fsq):  
            atom.SetProp("FSQ", fsq[idx])
            atom_props[idx] = fsq[idx]
    return mol, atom_props

class MappedReactionParser:
    def __init__(self, reaction_smiles, reactants=None, products=None, reaction=None, include_aromatic: bool = True):
        """Initialize MappedReactionParser
        
        Args:
            reaction_smiles (str): Reaction SMILES
            reactants (list[Chem.Mol], optional): List of reactant molecules
            products (list[Chem.Mol], optional): List of product molecules
            reaction (AllChem.ChemicalReaction, optional): RDKit reaction object
            include_aromatic (bool, optional): Whether to include aromatic bonds
        """
        self.reaction_smiles = reaction_smiles
        self.include_aromatic = include_aromatic
        try:
            self.reaction = reaction if reaction is not None else AllChem.ReactionFromSmarts(reaction_smiles)
            self.reactants = reactants if reactants is not None else self.reaction.GetReactants()
            self.products = products if products is not None else self.reaction.GetProducts()
            
            self.molecules = None
            self.reactant_info = None
            self.product_info = None
            
            self.molecules = self.get_molecules()
            if self.molecules is not None:
                self.reactant_info, self.product_info = self._parse_reaction()
                
        except Exception as e:
            logger.error(f"Failed to initialize reaction {reaction_smiles}: {e}")
            self.reaction = None
            self.reactants = None
            self.products = None
            self.molecules = None
            self.reactant_info = None
            self.product_info = None

    @classmethod
    def from_reaction_smiles(cls, reaction_smiles):
        """Create MappedReactionParser from reaction SMILES
        
        Args:
            reaction_smiles (str): Reaction SMILES
            
        Returns:
            MappedReactionParser or None: Parser instance if successful, None otherwise
        """
        try:
            reaction = AllChem.ReactionFromSmarts(reaction_smiles)
            reactants = reaction.GetReactants()
            products = reaction.GetProducts()
            instance = cls(
                reaction_smiles=reaction_smiles,
                reactants=reactants,
                products=products,
                reaction=reaction
            )
            if instance.is_valid():
                return instance
            return None
        except Exception as e:
            logger.error(f"Failed to create parser from reaction {reaction_smiles}: {e}")
            return None

    def is_valid(self):
        """Check if the parser is valid
        
        Returns:
            bool: True if valid, False otherwise
        """
        return (self.reaction is not None and 
                self.reactants is not None and 
                self.products is not None and 
                self.molecules is not None and 
                self.reactant_info is not None and 
                self.product_info is not None)

    def get_reactant_info(self) -> dict:
        """Get reactant information
        
        Returns:
            dict or None: Reactant information if valid, None otherwise
        """
        if not self.is_valid():
            logger.error("Parser is not valid")
            return None
        return self.reactant_info
    
    def get_product_info(self) -> dict:
        """Get product information
        
        Returns:
            dict or None: Product information if valid, None otherwise
        """
        if not self.is_valid():
            logger.error("Parser is not valid")
            return None
        return self.product_info

    def get_molecules(self) -> list[Molecule]:
        """Get all molecules
        
        Returns:
            list[Molecule] or None: List of molecules if valid, None otherwise
        """
        if self.reaction is None or self.reactants is None or self.products is None:
            logger.error("Reaction or molecules are not valid")
            return None
        try:
            reactant_smiles = [Chem.MolToSmiles(mol) for mol in self.reactants]
            product_smiles = [Chem.MolToSmiles(mol) for mol in self.products]
            return [Molecule(smiles) for smiles in reactant_smiles + product_smiles]
        except Exception as e:
            logger.error(f"Failed to get molecules: {e}")
            return None
    
    def get_reactants(self) -> list[Molecule]:
        """Get reactant molecules
        
        Returns:
            list[Molecule] or None: List of reactant molecules if valid, None otherwise
        """
        if not self.is_valid():
            logger.error("Parser is not valid")
            return None
        try:
            reactant_smiles = [Chem.MolToSmiles(mol) for mol in self.reactants]
            return [Molecule(smiles) for smiles in reactant_smiles]
        except Exception as e:
            logger.error(f"Failed to get reactants: {e}")
            return None
    
    def get_products(self) -> list[Molecule]:
        """Get product molecules
        
        Returns:
            list[Molecule] or None: List of product molecules if valid, None otherwise
        """
        if not self.is_valid():
            logger.error("Parser is not valid")
            return None
        try:
            product_smiles = [Chem.MolToSmiles(mol) for mol in self.products]
            return [Molecule(smiles) for smiles in product_smiles]
        except Exception as e:
            logger.error(f"Failed to get products: {e}")
            return None
    
    def get_reaction(self) -> Reaction:
        """Get reaction
        
        Returns:
            Reaction or None: Reaction if valid, None otherwise
        """
        if not self.is_valid():
            logger.error("Parser is not valid")
            return None
        try:
            return Reaction(self.reaction_smiles)
        except Exception as e:
            logger.error(f"Failed to get reaction: {e}")
            return None

    def _check_atom_map(self) -> bool:
        mapping_dict = {}
        
        for mol_idx, product in enumerate(self.products):
            for atom in product.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num == 0:
                    logger.error(f"Unmapped atom found in product {Chem.MolToSmiles(product)} in {self.reaction_smiles}")
                    return False
                if map_num > 0:
                    mapping_dict[map_num] = {
                        'product': (mol_idx, atom.GetIdx(), atom.GetSymbol()),
                        'reactant': None
                    }
            
        for mol_idx, reactant in enumerate(self.reactants):
            for atom in reactant.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    if map_num not in mapping_dict:
                        mapping_dict[map_num] = {
                            'product': None,
                            'reactant': (mol_idx, atom.GetIdx(), atom.GetSymbol())
                        }
                    else:
                        mapping_dict[map_num]['reactant'] = (mol_idx, atom.GetIdx(), atom.GetSymbol())

        for map_num, info in mapping_dict.items():
            if info['reactant'] is None:
                logger.error(f"Map {map_num} is not found in reactants, the reaction {self.reaction_smiles} is not valid.")
                return False
        return True
    
    def _reindex_atoms(self) -> tuple[list[Molecule], list[Molecule]]:
        product_atom_count = 0
        product_map_to_new = {}
        for product in self.products:
            for atom in product.GetAtoms():
                if atom.GetAtomMapNum() > 0:
                    product_map_to_new[atom.GetAtomMapNum()] = product_atom_count + 1  
                    product_atom_count += 1
        if not (len(product_map_to_new.keys()) == product_atom_count):
            logger.error(f"Atom map keys in product are not consecutive in {self.reaction_smiles}")
            return False
        # Update product atoms with new mapping using RDKit's SetAtomMapNum
        for product in self.products:
            for atom in product.GetAtoms():
                old_map = atom.GetAtomMapNum()
                atom.SetAtomMapNum(product_map_to_new[old_map])

        # Update reactant atoms with new mapping using RDKit's SetAtomMapNum
        next_available_idx = product_atom_count + 1
        for reactant in self.reactants:
            for atom in reactant.GetAtoms():
                old_map = atom.GetAtomMapNum()
                if old_map in product_map_to_new:
                    atom.SetAtomMapNum(product_map_to_new[old_map])
                else:
                    atom.SetAtomMapNum(next_available_idx)
                    next_available_idx += 1

        return True

    def _parse_atom_info(self) -> tuple[list[dict], list[dict]]:
        reactant_info = []
        product_info = []
        for i, mol in enumerate(self.reactants):
            mol_info = parse_reaction_rdmol(mol)
            if mol_info is None:
                logger.error(f"Failed to parse reactant {Chem.MolToSmiles(mol)} in {self.reaction_smiles}")
                return None, None
            reactant_info.append(mol_info)

        for i, mol in enumerate(self.products):
            mol_info = parse_reaction_rdmol(mol)
            if mol_info is None:
                logger.error(f"Failed to parse product {Chem.MolToSmiles(mol)} in {self.reaction_smiles}")
                return None, None
            product_info.append(mol_info)
        return reactant_info, product_info
    
    def _merge_atom_info(self, mols_info) -> dict:
        """Merge atom information from molecules
        
        Args:
            mols_info (list): List of molecule information
            
        Returns:
            list: List of merged atom information
        """
        selected_keys = ['elements', 'atom_types', 'atom_coords', 'atom_features', 'hybridizations', 'h_numbers', 'charges', 'seq_pos']
        merged_info = {k: np.zeros_like(
            np.concatenate([mol_info[k] for mol_info in mols_info])
            ) for k in selected_keys}

        merged_index = {k: [] for k in selected_keys}
        molecule_index = np.concatenate(
            [np.full(len(mol_info['elements']), i) for i, mol_info in enumerate(mols_info)]
        )
        for mol_info in mols_info:
            atom_map_dict = mol_info['atom_index_to_map_num']
            assert (np.sort(list(atom_map_dict.keys())) == np.arange(len(atom_map_dict))).all(), "Atom map keys are not consecutive"
            for k in selected_keys:
                for atom_idx, atom_map_num in atom_map_dict.items():
                    map_idx = atom_map_num - 1
                    merged_info[k][map_idx] = mol_info[k][atom_idx]
                    merged_index[k].append(map_idx)
        merged_info['molecule_index'] = molecule_index
        for k, v in merged_index.items():
            if not (np.sort(v) == np.arange(len(v))).all():
                logger.error(f"Atom map index for {k} are not consecutive in {self.reaction_smiles}")
                return None
        return merged_info
    
    def _expand_product_info(self, product_info, reactant_info) -> dict:

        for i in range(len(product_info['elements'])):
            if product_info['elements'][i] != reactant_info['elements'][i]:
                logger.error(f"Element mismatch in {self.reaction_smiles}")
                return None
            if product_info['atom_types'][i] != reactant_info['atom_types'][i]:
                logger.error(f"Type atom mismatch in {self.reaction_smiles}")
                return None
        
        molecule_index = np.concatenate(
            [product_info['molecule_index'], 
            np.full(len(reactant_info['elements'])-len(product_info['elements']), -1)]
        )
        product_info_expanded = {}
        product_info_expanded['product_atom_index'] = np.arange(len(product_info['elements']))
        for k in product_info.keys():
            val_product = product_info[k]
            val_reactant = reactant_info[k]
            val_product_expanded = np.zeros_like(val_reactant)
            for i in range(len(val_reactant)):
                if i < len(val_product):
                    val_product_expanded[i] = val_product[i]
                else:
                    val_product_expanded[i] = val_reactant[i]
            product_info_expanded[k] = val_product_expanded
        product_info_expanded['molecule_index'] = molecule_index
        return product_info_expanded
    
    def _parse_bond_info(self, include_aromatic: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        reactant_edge_index = []
        product_edge_index = []
        reactant_edge_type = []
        product_edge_type = []

        def convert_bond_type(bond):
            """Convert aromatic bonds to single/double bonds based on kekulization"""
            if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                # Convert aromatic bonds to single/double based on kekulization
                if bond.GetIsConjugated():
                    return Chem.rdchem.BondType.DOUBLE
                return Chem.rdchem.BondType.SINGLE
            return bond.GetBondType()

        # Parse reactant bonds
        for mol in self.reactants:
            if not include_aromatic:
                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                except:
                    logger.error(f"Failed to kekulize reactant {Chem.MolToSmiles(mol)} in {self.reaction_smiles}")
                    return None, None, None, None
            for bond in mol.GetBonds():
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()
                begin_map = begin_atom.GetAtomMapNum()
                end_map = end_atom.GetAtomMapNum()
                reactant_edge_index.append([begin_map-1, end_map-1])
                reactant_edge_index.append([end_map-1, begin_map-1])
                if not include_aromatic:
                    bond_type = convert_bond_type(bond)
                else:
                    bond_type = bond.GetBondType()
                if bond_type not in bond_type_id:
                    logger.error(f"Bond type {bond_type} not found in bond_type_id in {self.reaction_smiles}")
                    return None, None, None, None
                if not include_aromatic:
                    assert get_bond_type_id(bond_type) != 4, "Aromatic bond found in reactant"
                reactant_edge_type.append(get_bond_type_id(bond_type))
                reactant_edge_type.append(get_bond_type_id(bond_type))

        # Parse product bonds 
        for mol in self.products:
            if not include_aromatic:
                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                except:
                    logger.error(f"Failed to kekulize product {Chem.MolToSmiles(mol)} in {self.reaction_smiles}")
                    return None, None, None, None
            for bond in mol.GetBonds():
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()
                begin_map = begin_atom.GetAtomMapNum()
                end_map = end_atom.GetAtomMapNum()
                product_edge_index.append([begin_map-1, end_map-1])
                product_edge_index.append([end_map-1, begin_map-1])
                if not include_aromatic:
                    bond_type = convert_bond_type(bond)
                else:
                    bond_type = bond.GetBondType()
                if bond_type not in bond_type_id:
                    logger.error(f"Bond type {bond_type} not found in bond_type_id in {self.reaction_smiles}")
                    return None, None, None, None
                if not include_aromatic:
                    assert get_bond_type_id(bond_type) != 4, "Aromatic bond found in product"
                product_edge_type.append(get_bond_type_id(bond_type))
                product_edge_type.append(get_bond_type_id(bond_type))

        return (
            np.array(reactant_edge_index), 
            np.array(product_edge_index), 
            np.array(reactant_edge_type), 
            np.array(product_edge_type)
        )
    
    def _check_valid_molecules(self) -> bool:
        for mol in self.get_molecules():
            if mol._rdmol is None:
                logger.error(f"Invalid molecule found in {self.reaction_smiles}")
                return False
        return True

    def _parse_reaction(self) -> dict:
        if not self._check_valid_molecules():
            return None, None
        if not self._check_atom_map():
            return None, None            
        if not self._reindex_atoms():
            return None, None
        reactant_info, product_info = self._parse_atom_info()
        if reactant_info is None or product_info is None:
            return None, None
        reactant_info = self._merge_atom_info(reactant_info)
        if reactant_info is None:
            return None, None
        product_info = self._merge_atom_info(product_info)
        if product_info is None:
            return None, None
        product_info = self._expand_product_info(product_info, reactant_info)
        if product_info is None:
            return None, None
        (
            reactant_bond_index, product_bond_index, 
            reactant_bond_type, product_bond_type
        ) = self._parse_bond_info()
        if (reactant_bond_index is None or product_bond_index is None or 
            reactant_bond_type is None or product_bond_type is None):
            return None, None
        reactant_info['bond_index'] = reactant_bond_index
        reactant_info['bond_types'] = reactant_bond_type
        product_info['bond_index'] = product_bond_index
        product_info['bond_types'] = product_bond_type

        (
            reactant_edge_index, product_edge_index, 
            reactant_edge_type, product_edge_type
        ) = self._parse_bond_info(include_aromatic=self.include_aromatic)
        if (reactant_edge_index is None or product_edge_index is None or 
            reactant_edge_type is None or product_edge_type is None):
            return None, None
        reactant_info['edge_index'] = reactant_edge_index
        reactant_info['edge_types'] = reactant_edge_type
        product_info['edge_index'] = product_edge_index
        product_info['edge_types'] = product_edge_type
        if not self.include_aromatic and 4 in product_edge_type:
            logger.warning("Aromatic bond found in product while aromatic is not included")
        if not self.include_aromatic and 4 in reactant_edge_type:
            logger.warning("Aromatic bond found in reactant while aromatic is not included")
            
        if not self._check_valid_molecules():
            return None, None
        return reactant_info, product_info
    

        