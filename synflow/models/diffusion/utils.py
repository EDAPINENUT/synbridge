import torch
from synflow.data.deltagraph import DeltaGraphBatch
from typing import TypedDict, Dict, Tuple
from synflow.chem.constants import MAX_BONDS, MAX_DIFF

class TranslateGraph(TypedDict):
    # atom-mask
    padding_masks: torch.Tensor
    # src
    src_element_types: torch.Tensor
    src_elements: torch.Tensor
    src_bonds: torch.Tensor
    src_aromas: torch.Tensor
    src_charges: torch.Tensor
    src_masks: torch.Tensor
    src_segments: torch.Tensor
    src_flags: torch.Tensor
    src_bond_masks: torch.Tensor
    # tgt
    tgt_element_types: torch.Tensor
    tgt_elements: torch.Tensor
    tgt_bonds: torch.Tensor
    tgt_aromas: torch.Tensor
    tgt_charges: torch.Tensor
    tgt_masks: torch.Tensor
    tgt_segments: torch.Tensor
    tgt_flags: torch.Tensor
    tgt_bond_masks: torch.Tensor

OneSideGraph = Dict[str, torch.Tensor]

def transform_to_adjacency(bonds, edge_mask=None, remove_self_loops=False):
    B, L = bonds.shape[:2]
    eye = torch.eye(L).to(bonds.device)
    adjacency = torch.index_select(eye, dim=0, index=bonds.reshape(-1)).view(B, L, MAX_BONDS, L).sum(dim=2) # adjacenct matrix
    if edge_mask is not None:
        adjacency = adjacency * edge_mask
    if remove_self_loops:
        adjacency = adjacency * (1 - eye)
    return adjacency.long()

def prepare_source_graph(data: DeltaGraphBatch, mode: str = 'retro') -> OneSideGraph:
    if mode == 'retro':
        padding_masks = data['padding_masks']
        product_bond_masks = torch.einsum("bl,bk->blk", ~data['product_masks'], ~data['product_masks'])
        src_bonds = transform_to_adjacency(
            data['product_bonds'],
            product_bond_masks,
            remove_self_loops=True
        )
        src_elements = data['product_elements']
        src_element_types = data['product_element_types']
        src_aromas = data['product_aromas']
        src_charges = data['product_charges']
        src_masks = data['product_masks']
        src_segments = data['product_segments']
        src_flags = (~data['product_masks']).long()
        src_bond_masks = product_bond_masks
    
    elif mode == 'forward':
        padding_masks = data['padding_masks']
        reactant_bond_masks = torch.einsum("bl,bk->blk", ~data['reactant_masks'], ~data['reactant_masks'])
        src_bonds = transform_to_adjacency(
            data['reactant_bonds'],
            reactant_bond_masks,
            remove_self_loops=True
        )
        src_elements = data['reactant_elements']
        src_element_types = data['reactant_element_types']
        src_aromas = data['reactant_aromas']
        src_charges = data['reactant_charges']
        src_masks = data['reactant_masks']
        src_segments = data['reactant_segments']
        src_flags = data['reactant_flags']
        src_bond_masks = reactant_bond_masks
        
    src_graph = {
        "padding_masks": padding_masks,
        "bonds": src_bonds,
        "elements": src_elements,
        "element_types": src_element_types,
        "aromas": src_aromas,
        "charges": src_charges,
        "masks": src_masks,
        "segments": src_segments,
        "flags": src_flags,
        "bond_masks": src_bond_masks,
    }
    return src_graph

def prepare_ground_truth(data: DeltaGraphBatch, mode: str = 'retro', return_src: bool = False) -> Tuple[OneSideGraph, OneSideGraph]:
    if mode == 'retro':
        padding_masks = data['padding_masks']
        reactant_bond_masks = torch.einsum("bl,bk->blk", ~data['reactant_masks'], ~data['reactant_masks'])
        tgt_bonds = transform_to_adjacency(
            data['reactant_bonds'],
            reactant_bond_masks,
            remove_self_loops=True
        )
        tgt_elements = data['reactant_elements']
        tgt_element_types = data['reactant_element_types']
        tgt_aromas = data['reactant_aromas']
        tgt_charges = data['reactant_charges']
        tgt_masks = data['reactant_masks']
        tgt_segments = data['reactant_segments']
        tgt_flags = data['reactant_flags']
        tgt_bond_masks = reactant_bond_masks

        product_bond_masks = torch.einsum("bl,bk->blk", ~data['product_masks'], ~data['product_masks'])
        src_bonds = transform_to_adjacency(
            data['product_bonds'],
            product_bond_masks,
            remove_self_loops=True
        )
        src_elements = data['product_elements']
        src_element_types = data['product_element_types']
        src_aromas = data['product_aromas']
        src_charges = data['product_charges']
        src_masks = data['product_masks']
        src_segments = data['product_segments']
        src_flags = (~data['product_masks']).long()
        src_bond_masks = product_bond_masks
    
    elif mode == 'forward':
        padding_masks = data['padding_masks']
        product_bond_masks = torch.einsum("bl,bk->blk", ~data['product_masks'], ~data['product_masks'])
        tgt_bonds = transform_to_adjacency(
            data['product_bonds'],
            product_bond_masks,
            remove_self_loops=True
        )
        tgt_elements = data['product_elements']
        tgt_element_types = data['product_element_types']
        tgt_aromas = data['product_aromas']
        tgt_charges = data['product_charges']
        tgt_masks = data['product_masks']
        tgt_segments = data['product_segments']
        tgt_flags = (~data['product_masks']).long()
        tgt_bond_masks = product_bond_masks

        reactant_bond_masks = torch.einsum("bl,bk->blk", ~data['reactant_masks'], ~data['reactant_masks'])
        src_bonds = transform_to_adjacency(
            data['reactant_bonds'],
            reactant_bond_masks,
            remove_self_loops=True
        )
        src_elements = data['reactant_elements']
        src_element_types = data['reactant_element_types']
        src_aromas = data['reactant_aromas']
        src_charges = data['reactant_charges']
        src_masks = data['reactant_masks']
        src_segments = data['reactant_segments']
        src_flags = data['reactant_flags']
        src_bond_masks = reactant_bond_masks
        
    ground_truth = {
        "padding_masks": padding_masks,
        "bonds": tgt_bonds,
        "elements": tgt_elements,
        "element_types": tgt_element_types,
        "aromas": tgt_aromas,
        "charges": tgt_charges,
        "masks": tgt_masks,
        "segments": tgt_segments,
        "flags": tgt_flags,
        "bond_masks": tgt_bond_masks,
    }
    src_graph = {
        "padding_masks": padding_masks,
        "bonds": src_bonds,
        "elements": src_elements,
        "element_types": src_element_types,
        "aromas": src_aromas,
        "charges": src_charges,
        "masks": src_masks,
        "segments": src_segments,
        "flags": src_flags,
        "bond_masks": src_bond_masks,
    }
    if return_src:
        return ground_truth, src_graph
    else:
        return ground_truth


def prepare_translate_graph(data: DeltaGraphBatch, mode: str = 'retro') -> TranslateGraph:
    reactant_bond_masks = torch.einsum("bl,bk->blk", ~data['reactant_masks'], ~data['reactant_masks'])
    product_bond_masks = torch.einsum("bl,bk->blk", ~data['product_masks'], ~data['product_masks'])
    assert torch.all(data['padding_masks'] == data['reactant_masks'])
    assert torch.all(data['reactant_elements'] == data['elements'])
    if mode == 'retro':
        padding_masks = data['padding_masks']
        src_bonds = transform_to_adjacency(
            data['product_bonds'],
            product_bond_masks,
            remove_self_loops=True
        )
        src_elements = data['product_elements']
        src_element_types = data['product_element_types']
        src_aromas = data['product_aromas']
        src_charges = data['product_charges']
        src_masks = data['product_masks']
        src_segments = data['product_segments']
        src_flags = (~data['product_masks']).long()
        src_bond_masks = product_bond_masks

        tgt_bonds = transform_to_adjacency(
            data['reactant_bonds'],
            reactant_bond_masks,
            remove_self_loops=True
        )
        tgt_elements = data['reactant_elements']
        tgt_element_types = data['reactant_element_types']
        tgt_aromas = data['reactant_aromas']
        tgt_charges = data['reactant_charges']
        tgt_masks = data['reactant_masks']
        tgt_segments = data['reactant_segments']
        tgt_flags = data['reactant_flags']
        tgt_bond_masks = reactant_bond_masks
    
    elif mode == 'forward':
        padding_masks = data['padding_masks']
        src_bonds = transform_to_adjacency(
            data['reactant_bonds'],
            reactant_bond_masks,
            remove_self_loops=True
        )
        src_elements = data['reactant_elements']
        src_element_types = data['reactant_element_types']
        src_aromas = data['reactant_aromas']
        src_charges = data['reactant_charges']
        src_masks = data['reactant_masks']
        src_segments = data['reactant_segments']
        src_flags = data['reactant_flags']
        src_bond_masks = reactant_bond_masks

        tgt_bonds = transform_to_adjacency(
            data['product_bonds'],
            product_bond_masks,
            remove_self_loops=True
        )
        tgt_elements = data['product_elements']
        tgt_element_types = data['product_element_types']
        tgt_aromas = data['product_aromas']
        tgt_charges = data['product_charges']
        tgt_masks = data['product_masks']
        tgt_segments = data['product_segments']
        tgt_flags = (~data['product_masks']).long()
        tgt_bond_masks = product_bond_masks
    
    assert src_bonds.max() <= MAX_DIFF
    assert tgt_bonds.max() <= MAX_DIFF

    translate_graph: "TranslateGraph" = {
        "padding_masks": padding_masks,
        "src_element_types": src_element_types,
        "src_elements": src_elements,
        "src_bonds": src_bonds,
        "src_aromas": src_aromas,
        "src_charges": src_charges,
        "src_masks": src_masks,
        "src_segments": src_segments,
        "src_flags": src_flags,
        "src_bond_masks": src_bond_masks,
        "tgt_element_types": tgt_element_types,
        "tgt_elements": tgt_elements,
        "tgt_bonds": tgt_bonds,
        "tgt_aromas": tgt_aromas,
        "tgt_charges": tgt_charges,
        "tgt_masks": tgt_masks,
        "tgt_segments": tgt_segments,
        "tgt_flags": tgt_flags,
        "tgt_bond_masks": tgt_bond_masks,
    }
    return translate_graph

