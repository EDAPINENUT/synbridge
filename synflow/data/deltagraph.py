import enum
from collections.abc import Sequence
from typing import TypedDict

import torch
import numpy as np
from synflow.chem.constants import get_element_type


class DeltaGraphData(TypedDict, total=False):
    elements: torch.Tensor
    padding_masks: torch.Tensor
    #reactant
    reactant_elements: torch.Tensor
    reactant_element_types: torch.Tensor
    reactant_bonds: torch.Tensor
    reactant_charges: torch.Tensor
    reactant_masks: torch.Tensor
    reactant_aromas: torch.Tensor
    reactant_segments: torch.Tensor
    reactant_flags: torch.Tensor
    #product
    product_elements: torch.Tensor
    product_element_types: torch.Tensor
    product_bonds: torch.Tensor
    product_charges: torch.Tensor
    product_masks: torch.Tensor
    product_aromas: torch.Tensor
    product_segments: torch.Tensor
    product_flags: torch.Tensor


class DeltaGraphBatch(TypedDict, total=False):
    elements: torch.Tensor
    padding_masks: torch.Tensor
    #reactant
    reactant_elements: torch.Tensor
    reactant_element_types: torch.Tensor
    reactant_bonds: torch.Tensor
    reactant_charges: torch.Tensor
    reactant_masks: torch.Tensor
    reactant_aromas: torch.Tensor
    reactant_segments: torch.Tensor
    reactant_flags: torch.Tensor
    #product
    product_elements: torch.Tensor
    product_element_types: torch.Tensor
    product_bonds: torch.Tensor
    product_charges: torch.Tensor
    product_masks: torch.Tensor
    product_aromas: torch.Tensor
    product_segments: torch.Tensor
    product_flags: torch.Tensor


def sparse_to_dense(indices: torch.Tensor, values: torch.Tensor, length: int) -> torch.Tensor:
    if indices.dim() != 2:
        indices = torch.tensor([[0, 0]])
        values = torch.tensor([0])
    if indices.size(1) != 2:
        raise ValueError(f"Indices must have 2 columns, got {indices.size(1)} columns")
    sparse = torch.sparse_coo_tensor(indices.transpose(0, 1), values, (length, length))
    dense = sparse.to_dense()
    return dense

def torchify(data: dict) -> dict:
    """Convert dictionary values to PyTorch tensors with appropriate dtypes.
    
    Args:
        data (dict): Input dictionary containing various data types
        
    Returns:
        dict: Dictionary with values converted to appropriate torch tensors
    """
    for key in data:
        value = data[key]
        
        if isinstance(value, torch.Tensor):
            continue
            
        # numpy array
        elif isinstance(value, np.ndarray):
            if value.dtype in [np.int8, np.int16, np.int32, np.int64]:
                data[key] = torch.from_numpy(value).long()
            elif value.dtype in [np.float32, np.float64]:
                data[key] = torch.from_numpy(value).float()
            elif value.dtype == np.bool_:
                data[key] = torch.from_numpy(value).bool()
            else:
                data[key] = torch.from_numpy(value.astype(np.float32))
                
        # list
        elif isinstance(value, list):
            try:
                arr = np.array(value)
                if arr.dtype in [np.int8, np.int16, np.int32, np.int64]:
                    data[key] = torch.tensor(value, dtype=torch.long)
                elif arr.dtype in [np.float32, np.float64]:
                    data[key] = torch.tensor(value, dtype=torch.float)
                elif arr.dtype == np.bool_:
                    data[key] = torch.tensor(value, dtype=torch.bool)
                else:
                    data[key] = torch.tensor(value, dtype=torch.float)
            except:
                continue
                
        elif isinstance(value, (int, np.integer)):
            data[key] = torch.tensor(value, dtype=torch.long)
        elif isinstance(value, (float, np.floating)):
            data[key] = torch.tensor(value, dtype=torch.float)
        elif isinstance(value, bool):
            data[key] = torch.tensor(value, dtype=torch.bool)
            
        else:
            continue
            
    return data

def create_data(
    instance: dict,
):  
    element_types = torch.tensor(
        [get_element_type(int(i)) for i in instance['element']], dtype=torch.long
    )
    instance = torchify(instance)

    def apply_mask(tensor, mask, unsqueeze_dim=None):
        """Apply mask to tensor with optional dimension unsqueezing.
        
        Args:
            tensor: Input tensor to be masked
            mask: Boolean mask tensor
            unsqueeze_dim: Optional dimension to unsqueeze the mask
            
        Returns:
            Masked tensor where masked positions are set to zero
        """
        if unsqueeze_dim is not None:
            mask = mask.unsqueeze(unsqueeze_dim)
        return torch.where(~mask.bool(), tensor, torch.zeros_like(tensor))
    # MASK ATTRIBUTES FOR AVOIDING LABEL LEAKAGE!
    
    # mask element type and element
    instance['reactant_element_type'] = apply_mask(element_types, instance['reactant_mask'])
    instance['reactant_element'] = apply_mask(instance['element'], instance['reactant_mask'])

    instance['product_element_type'] = apply_mask(element_types, instance['product_mask'])
    instance['product_element'] = apply_mask(instance['element'], instance['product_mask'])

    # mask aromatic
    instance['reactant_aroma'] = apply_mask(instance['reactant_aroma'], instance['reactant_mask'])
    instance['product_aroma'] = apply_mask(instance['product_aroma'], instance['product_mask'])

    # mask charge
    instance['reactant_charge'] = apply_mask(instance['reactant_charge'], instance['reactant_mask'])
    instance['product_charge'] = apply_mask(instance['product_charge'], instance['product_mask'])

    # mask bond 
    instance['reactant_bond'] = apply_mask(instance['reactant_bond'], instance['reactant_mask'], unsqueeze_dim=-1)
    instance['product_bond'] = apply_mask(instance['product_bond'], instance['product_mask'], unsqueeze_dim=-1)

    atom_num = instance['element'].shape[0]
    padding_masks = torch.zeros(atom_num, dtype=torch.bool)

    data: "DeltaGraphData" = {
        "elements": instance['element'],
        "reactant_elements": instance['reactant_element'],
        "reactant_element_types": instance['reactant_element_type'],
        "reactant_bonds": instance['reactant_bond'],
        "reactant_charges": instance['reactant_charge'],
        "reactant_masks": instance['reactant_mask'],
        "reactant_aromas": instance['reactant_aroma'],
        "reactant_segments": instance['reactant_segment'],
        "reactant_flags": instance['reactant_flag'],
        "product_elements": instance['product_element'],
        "product_element_types": instance['product_element_type'],
        "product_bonds": instance['product_bond'],
        "product_charges": instance['product_charge'],
        "product_masks": instance['product_mask'],
        "product_aromas": instance['product_aroma'],
        "product_segments": instance['product_segment'],
        "padding_masks": padding_masks,
    }
    return data

