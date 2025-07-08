from rdkit.Chem import AllChem, BondType
import torch

atom_families = ['Acceptor', 'Donor', 'Aromatic', 
                 'Hydrophobe', 'LumpedHydrophobe', 
                 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
atom_families_id = {s: i for i, s in enumerate(atom_families)}
hybridization_type = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
hybridization_type_id = {s: i+1 for i, s in enumerate(hybridization_type)}
atom_name = [
    'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 
    'K', 'Na', 'B', 'Al', 'Si', 'Cr', 'Li', 'Mg', 'Fe', 'Pd', 'Ni', 'Au',
    'Pt', 'Cu', 'Zn', 'Ag', 'Cd', 'Hg', 'Sn', 'Pb', 'Bi', 'Cs', 'Rb', 'Ba', 'Ca', 'Sr', 'Mn', 'Co', 
]
atom_type_id = {s: i+1 for i, s in enumerate(atom_name)}
atom_id_to_type = {v: k for k, v in atom_type_id.items()}
bond_type_id = {
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}
MAX_BONDS = 6
MAX_DIFF = 4
bond_id_to_type = {v: k for k, v in bond_type_id.items()}
hs_allowed_number = {
    'H': 1,
    'C': 4,
    'N': 3,
    'O': 2,
    'F': 1,
    'P': 5,
    'S': 6,
}
hs_allowed_max = 8 + 1
charge_class = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
charge_class_id = {s: i+1 for i, s in enumerate(charge_class)}
charge_class_id_to_type = {v: k for k, v in charge_class_id.items()}

reaction_class = [
    'Cbz_deprotection', 'DCC_condensation', 'reductive_amination', 
    'nucleophilic_attack_to_(thio)carbonyl_or_sulfonyl', 'carbonyl_reduction', 
    'SN2_alcohol(thiol)', 'O_demethylation', 'aldol_condensation', 
    'SN1', 'SN2', 'alcohol_attack_to_carbonyl_or_sulfonyl', 
    'nucleophilic_attack_to_iso(thio)cyanate', 'SN2_with_tosylate', 'Wittig_ver_2', 
    'SNAr(ortho)', 'SNAr(para)', 'alkynyl_attack_to_carbonyl', 
    'Boc_deprotection', 'Michael_addition', 'ester_reduction', 
    'Mannich', 'SNAr_alco(thi)ol(para)', 'base_cat_ester_hydrolysis', 
    'Wittig', 'Mitsunobu', 'Hantzsch_thiazole_synthesis', 
    'carboxylic_acid_derivative_hydrolysis_or_formation', 'Wolf_Kishner_reduction', 
    'imidazole_synthesis', 'imine_formation', 'Grignard', 'imine_reduction', 
    'Ing_Manske', 'Jones_oxidation', 'sulfide_oxidation', 'aldol_addition', 
    'alkene_epoxidation', 'SNAr_alco(thi)ol(ortho)', 'Swern_oxidation', 
    'nitrile_reduction', 'Horner_Wadsworth_Emmons', 'Knorr_pyrazole_synthesis',
    'isothiocyanate_synthesis', 'Appel', 'Friedel_Crafts_acylation', 
    'Vilsmeier_formylation', 'primary_amide_dehydration', '(hemi)acetal(aminal)_hydrolysis', 
    'Staudinger', 'amide_reduction', 'double_SN2', 'amine_oxidation', 
    'methyl_ester_synthesis', 'acetal_formation', 'Paal_Knorr_pyrrole_synthesis',
    'Weinreb_ketone_synthesis', 'sulfide_oxidation_by_peroxide', 'acetal_formation_from_enol_ether',
    'Fmoc_deprotection', 'Markovnikov_addition', 'lactone_reduction', 
    'intramolecular_lactonization', 'SN1_with_tosylate'
]
reaction_class_id = {s: i+1 for i, s in enumerate(reaction_class)}
reaction_class_id_to_type = {v: k for k, v in reaction_class_id.items()}

element_id = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 11, 12, 13, 14, 15, 16, 17, 18, 19,
               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38,
               39, 40, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
               59, 60, 62, 63, 66, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]
element_id_to_type = {v: k+1 for k, v in enumerate(element_id)}
element_type_to_id = {v: k for k, v in element_id_to_type.items()}
def get_element_type(element: int) -> int:
    if element not in element_id_to_type:
        return 0
    return element_id_to_type[element]

def get_element_id(element: int) -> int:
    if element not in element_id_to_type:
        return 0
    return element_id_to_type[element]

element_type_to_id_tensor = torch.tensor(
    [element_type_to_id.get(i, 0) for i in range(max(element_type_to_id.keys()) + 1)]
)

def get_element_id_batch(elements: torch.Tensor) -> torch.Tensor:
    return element_type_to_id_tensor.to(elements.device)[elements]

def get_reaction_class_id(reaction_class: str) -> int:
    if reaction_class not in reaction_class_id:
        return 0
    return reaction_class_id[reaction_class]

def get_reaction_class(reaction_type_id: int) -> str:
    if reaction_type_id not in reaction_class_id_to_type:
        return 'Unknown'
    return reaction_class_id_to_type[reaction_type_id]

def get_atom_type_id(atom_type: str) -> int:
    if atom_type not in atom_type_id:
        return 0
    return atom_type_id[atom_type]

def get_atom_type(type_id: int) -> str:
    if type_id not in atom_id_to_type:
        return 'H'
    return atom_id_to_type[type_id]

def get_bond_type_id(bond_type: int) -> int:
    if bond_type not in bond_type_id:
        return 0
    return bond_type_id[bond_type]

def get_bond_type(type_id: int) -> BondType:
    if type_id not in bond_id_to_type:
        return BondType.SINGLE
    return bond_id_to_type[type_id]

def get_hybridization_type_id(hybridization_type: str) -> int:
    if hybridization_type not in hybridization_type_id:
        return 5
    return hybridization_type_id[hybridization_type]

def get_charge_class_id(charge: int) -> int:
    if charge not in charge_class_id:
        return 0
    return charge_class_id[charge]

def get_charge_class(charge_id: int) -> int:
    if charge_id not in charge_class_id_to_type:
        return 0
    return charge_class_id_to_type[charge_id]
