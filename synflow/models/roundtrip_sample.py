from typing import Dict 
import torch
from synflow.data.deltagraph import DeltaGraphBatch
from synflow.models.misc import expand_batch


def prepare_roundtrip_sample(preds_batched, tgts_batched, srcs_batched):
    output_samples = {
        "pred_bonds": preds_batched['bonds'],
        "pred_aromas": preds_batched['aromas'],
        "pred_charges": preds_batched['charges'],
        "pred_elements": preds_batched['elements'],
        "pred_element_types": preds_batched['element_types'],
        "src_masks": preds_batched['masks'],
        "src_flags": preds_batched['flags'],
        "tgt_masks": tgts_batched['masks'],
        "true_bonds": tgts_batched['bonds'],
        "true_aromas": tgts_batched['aromas'],    
        "true_charges": tgts_batched['charges'],
        "true_elements": tgts_batched['elements'],
        "true_element_types": tgts_batched['element_types'],
        "input_bonds": srcs_batched['bonds'],
        "input_aromas": srcs_batched['aromas'],
        "input_charges": srcs_batched['charges'],
        "input_elements": srcs_batched['elements'],
        "input_element_types": srcs_batched['element_types'],
        "input_masks": srcs_batched['masks'],
        "input_flags": srcs_batched['flags'],
    }
    return output_samples


def roundtrip_sample(
    wrapper, 
    batch: DeltaGraphBatch, 
    num_samples: int = 1000, 
    sample_steps: int = 20,
    mode: str = 'retro-forward-coverage'
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Generate round trip samples from the model.
    """
    if mode == 'retro-forward-coverage':
        sampled_batch = wrapper.sample(batch, 1, sample_steps, mode='retro', roundtrip=True)
        pred_batch = {}
        pred_batch['reactant_element_types'] = sampled_batch['pred_element_types']
        pred_batch['reactant_elements'] = sampled_batch['pred_elements']
        pred_batch['reactant_bonds'] = sampled_batch['pred_bonds']
        pred_batch['reactant_aromas'] = sampled_batch['pred_aromas']
        pred_batch['reactant_charges'] = sampled_batch['pred_charges']
        pred_batch['reactant_masks'] = sampled_batch['tgt_masks']
        pred_batch['reactant_flags'] = sampled_batch['tgt_flags']
        pred_batch['reactant_segments'] = batch['reactant_segments'] # label may leak!
        pred_batch['product_element_types'] = batch['product_element_types']   
        pred_batch['product_elements'] = batch['product_elements']   
        pred_batch['product_bonds'] = batch['product_bonds']
        pred_batch['product_aromas'] = batch['product_aromas']
        pred_batch['product_charges'] = batch['product_charges']
        pred_batch['product_masks'] = batch['product_masks']
        pred_batch['product_flags'] = batch['product_flags']
        pred_batch['product_segments'] = batch['product_segments']
        pred_batch['padding_masks'] = batch['padding_masks']
        roundtrip_sampled_batch = wrapper.sample(pred_batch, num_samples, sample_steps, mode='forward')
        
    elif mode == 'forward-retro-coverage':
        sampled_batch = wrapper.sample(batch, 1, sample_steps, mode='forward', roundtrip=True)
        pred_batch = {}
        pred_batch['product_element_types'] = sampled_batch['pred_element_types']
        pred_batch['product_elements'] = sampled_batch['pred_elements']
        pred_batch['product_bonds'] = sampled_batch['pred_bonds']
        pred_batch['product_aromas'] = sampled_batch['pred_aromas']
        pred_batch['product_charges'] = sampled_batch['pred_charges']
        pred_batch['product_masks'] = sampled_batch['tgt_masks']
        pred_batch['product_flags'] = sampled_batch['tgt_flags']
        pred_batch['product_segments'] = batch['product_segments']
        pred_batch['reactant_element_types'] = batch['reactant_element_types']
        pred_batch['reactant_elements'] = batch['reactant_elements']
        pred_batch['reactant_bonds'] = batch['reactant_bonds']
        pred_batch['reactant_aromas'] = batch['reactant_aromas']
        pred_batch['reactant_charges'] = batch['reactant_charges']
        pred_batch['reactant_masks'] = batch['reactant_masks']
        pred_batch['reactant_flags'] = batch['reactant_flags']
        pred_batch['reactant_segments'] = batch['reactant_segments']
        pred_batch['padding_masks'] = batch['padding_masks']
        roundtrip_sampled_batch = wrapper.sample(pred_batch, num_samples, sample_steps, mode='retro')
    
    elif mode == 'retro-forward-accuracy':
        sampled_batch = wrapper.sample(batch, num_samples, sample_steps, mode='forward', roundtrip=True)
        pred_batch = {}
        batch_expanded = expand_batch(batch, num_samples)
        pred_batch['reactant_element_types'] = sampled_batch['pred_element_types']
        pred_batch['reactant_elements'] = sampled_batch['pred_elements']
        pred_batch['reactant_bonds'] = sampled_batch['pred_bonds']
        pred_batch['reactant_aromas'] = sampled_batch['pred_aromas']
        pred_batch['reactant_charges'] = sampled_batch['pred_charges']
        pred_batch['reactant_masks'] = sampled_batch['tgt_masks']
        pred_batch['reactant_flags'] = sampled_batch['tgt_flags']
        pred_batch['reactant_segments'] = batch_expanded['reactant_segments'] # label may leak!
        pred_batch['product_element_types'] = batch_expanded['product_element_types']
        pred_batch['product_elements'] = batch_expanded['product_elements']
        pred_batch['product_bonds'] = batch_expanded['product_bonds']
        pred_batch['product_aromas'] = batch_expanded['product_aromas']
        pred_batch['product_charges'] = batch_expanded['product_charges']
        pred_batch['product_masks'] = batch_expanded['product_masks']
        pred_batch['product_flags'] = batch_expanded['product_flags']
        pred_batch['product_segments'] = batch_expanded['product_segments']
        pred_batch['padding_masks'] = batch_expanded['padding_masks']
        roundtrip_sampled_batch = wrapper.sample(pred_batch, 1, sample_steps, mode='retro')
    
    elif mode == 'forward-retro-accuracy':
        sampled_batch = wrapper.sample(batch, num_samples, sample_steps, mode='forward', roundtrip=True)
        pred_batch = {}
        batch_expanded = expand_batch(batch, num_samples)
        pred_batch['product_element_types'] = sampled_batch['pred_element_types']
        pred_batch['product_elements'] = sampled_batch['pred_elements']
        pred_batch['product_bonds'] = sampled_batch['pred_bonds']
        pred_batch['product_aromas'] = sampled_batch['pred_aromas']
        pred_batch['product_charges'] = sampled_batch['pred_charges']
        pred_batch['product_masks'] = sampled_batch['tgt_masks']
        pred_batch['product_flags'] = sampled_batch['tgt_flags']
        pred_batch['product_segments'] = batch_expanded['product_segments']
        pred_batch['reactant_element_types'] = batch_expanded['reactant_element_types']
        pred_batch['reactant_elements'] = batch_expanded['reactant_elements']
        pred_batch['reactant_bonds'] = batch_expanded['reactant_bonds']
        pred_batch['reactant_aromas'] = batch_expanded['reactant_aromas']
        pred_batch['reactant_charges'] = batch_expanded['reactant_charges']
        pred_batch['reactant_masks'] = batch_expanded['reactant_masks']
        pred_batch['reactant_flags'] = batch_expanded['reactant_flags']
        pred_batch['reactant_segments'] = batch_expanded['reactant_segments']
        pred_batch['padding_masks'] = batch_expanded['padding_masks']
        roundtrip_sampled_batch = wrapper.sample(pred_batch, 1, sample_steps, mode='retro')
    
    return roundtrip_sampled_batch
