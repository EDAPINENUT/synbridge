from torch import nn
import torch
import torch.nn.functional as F
from synflow.data.deltagraph import DeltaGraphBatch
from synflow.models.misc import shape_back, append_dims
from synflow.models.transformers import (
    get_encoder, 
    get_variational_decoder,
)
from synflow.chem.constants import element_id, get_element_id_batch, MAX_DIFF
from typing import Dict, Tuple
import torch.distributions as D
from tqdm import tqdm
import copy
from .utils import *

Code = torch.Tensor
ProbParam = torch.Tensor
Prob = D.Distribution
Order = torch.Tensor
Time = torch.Tensor

class LinearSchedule(nn.Module):
    def __init__(self, ):
        super().__init__()

    def __call__(self, t):
        return t
    
    def derivative(self, t):
        return 1

class SqrtSquareSchedule(nn.Module):
    def __init__(self, sigma_max=1.0):
        super().__init__()
        self.sigma_max = sigma_max

    def __call__(self, t):
        return torch.sqrt(t * (1 - t)) * self.sigma_max
    
    def derivative(self, t):
        return ((1 - 2 * t) * self.sigma_max / (2 * torch.sqrt(t * (1 - t)))).clamp(min=0, max=1e2)

class DiscUniformNoiseSampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.aroma_type = 2
        self.charge_type = 13
        kappa_type = config.kappa.type
        sigma_type = config.sigma.type
        sigma_max = getattr(config.sigma, 'max', 0.5)
        if kappa_type == 'linear':
            self.kappa = LinearSchedule()
        else:
            raise ValueError(f"Unknown kappa type: {kappa_type}")
        if sigma_type == 'sqrtsquare':
            self.sigma = SqrtSquareSchedule(sigma_max)
        else:
            raise ValueError(f"Unknown sigma type: {sigma_type}")

    def forward(self, x):
        return torch.randn_like(x)

    def sample_inter_discrete(self, x0: Code, x1: Code, t: Time, num_classes: int) -> Tuple[Prob, Code]:
        p0: ProbParam = F.one_hot(x0.long(), num_classes=num_classes)
        p1: ProbParam = F.one_hot(x1.long(), num_classes=num_classes)
        pu: ProbParam = torch.ones_like(p1.float()) / num_classes
        pt: Prob = D.Categorical(
            (1 - append_dims(self.sigma(t), p1.ndim)) * (
                append_dims(self.kappa(t), p1.ndim) * p1 + (1 - append_dims(self.kappa(t), p1.ndim)) * p0
            ) + append_dims(self.sigma(t), p1.ndim) * pu
        )
        xt = pt.sample()
        return pt, xt
    
    def correct_u(self, 
                  pt: ProbParam, p1: ProbParam, p0: ProbParam, 
                  t: Time, alpha_t: torch.Tensor, beta_t: torch.Tensor):
        alpha_t = append_dims(alpha_t, pt.ndim)
        beta_t = append_dims(beta_t, pt.ndim)
        return alpha_t * self.forward_u(pt, p1, t) - beta_t * self.backward_u(pt, p0, t)
    
    def forward_u(self, pt: ProbParam, p1: ProbParam, t: Time):
        kappa_coeff = self.kappa.derivative(t) / (1 - self.kappa(t))
        return append_dims(kappa_coeff, pt.ndim).to(pt.device) * (p1 - pt)
    
    def backward_u(self, pt: ProbParam, p0: ProbParam, t: Time):
        kappa_coeff = self.kappa.derivative(t) / (self.kappa(t))
        return append_dims(kappa_coeff, pt.ndim).to(pt.device) * (p0 - pt)
    
    def adaptive_dt(self, dt: Time, t: Time, alpha_t: torch.Tensor, beta_t: torch.Tensor):
        alpha_term = alpha_t * self.kappa.derivative(t) / (1 - self.kappa(t))
        beta_term = beta_t * self.kappa.derivative(t) / self.kappa(t)
        coeff = 1 / (alpha_term + beta_term)
        return torch.minimum(dt, coeff)
    
class VAEDIFMUniform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.flow is not None:
            self.flow_sampler = DiscUniformNoiseSampler(config.flow)
            self.interm_encoder = get_encoder(config.mol_encoder)
        else:
            self.flow_sampler = None
            self.interm_encoder = None

        self.reactant_encoder = get_encoder(config.mol_encoder)
        self.product_encoder = get_encoder(config.mol_encoder)
        self.product_decoder = get_variational_decoder(config.vae_decoder)
        self.decoder_only = config.decoder_only
        self.force_symbond = config.get('force_symbond', True)

    def _sample_intermediate(self, src, tgt, t):
        src_aromas = src['aromas']
        tgt_aromas = tgt['aromas']
        src_charges = src['charges'] 
        tgt_charges = tgt['charges'] 
        src_bonds = src['bonds']
        tgt_bonds = tgt['bonds']
        src_element_types = src['element_types']
        tgt_element_types = tgt['element_types']
        _, interm_aromas = self.flow_sampler.sample_inter_discrete(
            src_aromas, tgt_aromas, t, num_classes=self.flow_sampler.aroma_type
        )
        _, interm_charges = self.flow_sampler.sample_inter_discrete(
            src_charges, tgt_charges, t, num_classes=self.flow_sampler.charge_type
        )
        _, interm_bonds = self.flow_sampler.sample_inter_discrete(
            src_bonds, tgt_bonds, t, num_classes=MAX_DIFF
        )
        _, interm_element_types = self.flow_sampler.sample_inter_discrete(
            src_element_types, tgt_element_types, t, num_classes=len(element_id)+1
        )
        
        return {
            'aromas': interm_aromas,
            'charges': interm_charges,
            'bonds': interm_bonds,
            'element_types': interm_element_types
        }

    
    def _one_step_sample(
        self, 
        curr_bonds, curr_aromas, curr_charges, curr_element_types,
        pred_bonds, pred_aromas, pred_charges, pred_element_types,
        init_bonds, init_aromas, init_charges, init_element_types,
        t, dt, straight_forward=False
    ):
        N = curr_bonds.shape[1]
        
        def step_forward(curr, pred, init, num_classes, dt):
            input_shape = curr.shape
            pt = F.one_hot(curr, num_classes=num_classes)
            p1 = F.softmax(pred, dim=-1)
            p0 = F.one_hot(init, num_classes=num_classes)
            
            if straight_forward:
                u = self.flow_sampler.forward_u(pt, p1, t)
            else:
                dt = self.flow_sampler.adaptive_dt(dt, t, alpha_t(t), beta_t(t))
                u = self.flow_sampler.correct_u(pt, p1, p0, t, alpha_t(t), beta_t(t))
            
            prob = (u * append_dims(dt, pt.ndim) + pt).clamp(min=1e-10)
            
            next_val = torch.multinomial(
                prob.reshape(-1, num_classes), num_samples=1
            ).reshape(input_shape)
            
            return next_val

        alpha = 12.0
        a = 2.0
        b = 0.5
        alpha_t = lambda t: 1 + (alpha * (t ** a)) * ((1 - t) ** b)
        beta_t = lambda t: alpha_t(t) - 1

        next_bonds = step_forward(
            curr_bonds, pred_bonds, init_bonds, MAX_DIFF, dt
        )
        next_aromas = step_forward(
            curr_aromas, pred_aromas, init_aromas, self.flow_sampler.aroma_type, dt
        )
        next_charges = step_forward(
            curr_charges, pred_charges, init_charges, self.flow_sampler.charge_type, dt
        )
        next_element_types = step_forward(
            curr_element_types, pred_element_types, init_element_types, len(element_id)+1, dt
        )

        return next_bonds, next_aromas, next_charges, next_element_types
        

    def sample_intermediate(self, data: TranslateGraph, t, mode='retro'):
        src = {
            'element_types': data['src_element_types'],
            'elements': data['src_elements'],
            'bonds': data['src_bonds'],
            'aromas': data['src_aromas'],
            'charges': data['src_charges'] + 6,
            'masks': data['src_masks'],
            'padding_masks': data['padding_masks'],
            'segments': data['src_segments'],
            'flags': data['src_flags']
        }

        tgt = {
            'element_types': data['tgt_element_types'],
            'elements': data['tgt_elements'],
            'bonds': data['tgt_bonds'],
            'aromas': data['tgt_aromas'],
            'charges': data['tgt_charges'] + 6,
            'masks': data['tgt_masks'],
            'padding_masks': data['padding_masks'],
            'segments': data['tgt_segments'],
            'flags': data['tgt_flags']
        }
        
        interm = self._sample_intermediate(src, tgt, t)
        interm['segments'] = data['src_segments']
        interm['flags'] = data['src_flags']
        interm['masks'] = data['src_masks']
        interm['padding_masks'] = data['padding_masks']
        padding_masks_node = ~interm['padding_masks']
        padding_masks_edge = torch.einsum("bl,bk->blk", padding_masks_node, padding_masks_node)
        
        interm['bonds'] = torch.where(
            padding_masks_edge,
            interm['bonds'],
            src['bonds']
        )
        interm['charges'] = torch.where(
            padding_masks_node,
            interm['charges'],
            src['charges']
        )
        if self.force_symbond:
            B, L = interm['bonds'].shape[:2]
            lower_tri_mask = torch.tril(
                torch.ones(L, L, device=interm['bonds'].device)
            ).unsqueeze(0).expand(B, -1, -1).bool()
            # Set the lower triangular part the same as the upper triangular part
            interm['bonds'] = torch.where(
                lower_tri_mask, 
                interm['bonds'].permute(0, 2, 1), 
                interm['bonds']
            )
            assert (interm['bonds'] == interm['bonds'].permute(0, 2, 1)).all()

        interm['aromas'] = torch.where(
            padding_masks_node,
            interm['aromas'],
            src['aromas']
        )
        if mode == 'retro':
            # the product elements are unchanged, while the dummy elements should be recovered
            gen_masks_element = torch.logical_xor(data['src_masks'], data['padding_masks'])
            interm['element_types'] = torch.where(
                gen_masks_element,
                interm['element_types'],
                src['element_types']
            )
        else:
            interm['element_types'] = src['element_types']

        interm['elements'] = get_element_id_batch(interm['element_types'])
        if mode == 'forward':
            assert (interm['elements'] == src['elements']).all()

        src['charges'] = src['charges'] - 6
        interm['charges'] = interm['charges'] - 6

        return src, interm 

    def encode_decode(self, src, interm=None, t=None):
        src_bonds = src['bonds']
        src_aromas = src['aromas']
        src_charges = src['charges']
        src_masks = src['padding_masks']
        src_segments = src['segments']
        src_flags = src['flags']
        src_elements = src['elements']

        src_emb = self.reactant_encoder(
            src_elements, 
            src_bonds, 
            src_aromas,
            src_charges, 
            src_masks, 
            src_segments, 
            src_flags
        )

        interm_bonds = interm['bonds']
        interm_aromas = interm['aromas']
        interm_charges = interm['charges']
        interm_masks = interm['padding_masks']
        interm_segments = interm['segments']
        interm_flags = interm['flags']
        interm_elements = interm['elements']

        interm_emb = self.interm_encoder(
            interm_elements, 
            interm_bonds, 
            interm_aromas, 
            interm_charges, 
            interm_masks, 
            interm_segments, 
            interm_flags
        )

        pred = self.product_decoder(
            interm_emb + src_emb, 
            interm_bonds, 
            interm_masks,
            decoder_only=self.decoder_only
        )

        return pred


    def get_loss(
            self,
            data: DeltaGraphBatch,
            t: torch.Tensor,
            mode: str = 'retro' # retro, forward
        ):
        translate_graph = prepare_translate_graph(data, mode=mode)

        src, interm = self.sample_intermediate(translate_graph, t, mode=mode)
        
        pred = self.encode_decode(src, interm, t)
        
        loss_dict = self.compute_loss(pred, translate_graph, mode=mode)
        
        return loss_dict
    
    def compute_loss(self, pred: Dict[str, torch.Tensor], data: TranslateGraph, mode='retro'):
        loss_bond = self.ce_bond_loss(
            pred['bond_outputs'], 
            data['tgt_bonds'], 
            data['tgt_masks'], 
            data['padding_masks']
        )
        loss_aroma = self.aroma_loss(
            pred['aroma_logits'], 
            data['tgt_aromas'], 
            data['tgt_masks']
        )
        loss_charge = self.charge_loss(
            pred['charge_logits'], 
            data['tgt_charges'], 
            data['tgt_masks']
        )
        if mode == 'retro':
            loss_element = self.element_loss(
                pred['element_type_logits'], 
                data['tgt_element_types'], 
                data['tgt_masks']
            )
        else:
            loss_element = torch.tensor(0.0).to(data['src_elements'].device)

        return {
            'bond': loss_bond.mean(),
            'aroma': loss_aroma.mean(),
            'charge': loss_charge.mean(),
            'element': loss_element.mean(),
        }
    
    def sample(self, data, sample_steps=100, t_min=0.001, mode='retro'):
        src_graph = prepare_source_graph(data, mode=mode)
        preds = self.ode_sample(src_graph, sample_steps=sample_steps, t_min=t_min, mode=mode)
        return preds
        
    @torch.no_grad()
    def ode_sample(self, src: OneSideGraph, sample_steps=20, t_min=0.001, straight_forward=False, mode='retro'):
        B = src['padding_masks'].shape[0]
        default_h = 1 / sample_steps
        ts = torch.linspace(
            t_min,
            1.0 - default_h,
            sample_steps + 1
        )
        
        interm = copy.deepcopy(src)

        for idx, t in enumerate(tqdm(ts[:-1], desc="Sampling on single batch...")):
            dt = ts[idx + 1] - ts[idx]
            t_curr = t * torch.ones((B,), device=src['elements'].device)
            pred = self.encode_decode(src, interm, t_curr)
            interm = self.one_step_sample(interm, pred, src, t_curr, dt, straight_forward, mode)
            if mode == 'forward':
                assert (interm['elements'] == src['elements']).all()

        pred_bonds = F.one_hot(interm['bonds'].long(), num_classes=MAX_DIFF)
        pred_aromas = interm['aromas']
        pred_charges = interm['charges']
        pred_elements = interm['elements']
        if mode == 'forward':
            assert (pred_elements == src['elements']).all()
        src_masks = src['padding_masks']
        src_flags = src['flags']
        preds = {
            'bonds': pred_bonds,
            'aromas': pred_aromas,
            'charges': pred_charges,
            'elements': pred_elements,
            'masks': src_masks,
            'flags': src_flags
        }
        return preds

    def one_step_sample(self, interm, pred, init, t, dt, straight_forward=False, mode='retro'):
        pred['aroma_logits'] = torch.cat(
            (torch.zeros_like(pred['aroma_logits']), pred['aroma_logits']),
            dim=-1
        )

        sample_bonds, sample_aromas, sample_charges, sample_element_types = self._one_step_sample(
            interm['bonds'],
            interm['aromas'],
            interm['charges'] + 6,
            interm['element_types'],
            pred['bond_outputs'],
            pred['aroma_logits'],
            pred['charge_logits'],
            pred['element_type_logits'],
            init['bonds'],
            init['aromas'],
            init['charges'] + 6,
            init['element_types'],
            t, 
            dt,
            straight_forward
        )

        padding_masks_node = ~interm['padding_masks']
        padding_masks_edge = torch.einsum("bl,bk->blk", padding_masks_node, padding_masks_node)

        if self.force_symbond:
            B, L = sample_bonds.shape[:2]
            lower_tri_mask = torch.tril(
                torch.ones(L, L, device=sample_bonds.device)
            ).unsqueeze(0).expand(B, -1, -1).bool()
            sample_bonds = torch.where(
                lower_tri_mask,
                sample_bonds.permute(0, 2, 1),
                sample_bonds
            )
            # assert (sample_bonds == sample_bonds.permute(0, 2, 1)).all()

        sample_bonds = torch.where(
            padding_masks_edge,
            sample_bonds,
            interm['bonds']
        )
        sample_aromas = torch.where(
            padding_masks_node,
            sample_aromas,
            interm['aromas']
        )
        sample_charges = torch.where(
            padding_masks_node,
            sample_charges - 6,
            interm['charges']
        )
        if mode == 'retro': 
            gen_masks_node = torch.logical_xor(interm['masks'], interm['padding_masks'])
            sample_element_types = torch.where(
                gen_masks_node,
                sample_element_types,
                interm['element_types']
            )
            elements = get_element_id_batch(sample_element_types)
        elif mode == 'forward':
            sample_element_types = interm['element_types']
            elements = get_element_id_batch(sample_element_types)
            if (elements != interm['elements']).all():
                print('Elements changed in forward prediction!')
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        next_interm = {
            'padding_masks': interm['padding_masks'],
            'bonds': sample_bonds,
            'elements': elements,
            'element_types': sample_element_types,
            'aromas': sample_aromas,
            'charges': sample_charges,
            'masks': interm['masks'],
            'segments': interm['segments'],
            'flags': interm['flags'],
            'bond_masks': interm['bond_masks'],
        }

        return next_interm
    
    def element_loss(self, logits, tgt_element_types, tgt_masks):
        tgt_masks = tgt_masks.bool().float()
        B, L = logits.shape[:2]
        CE = nn.CrossEntropyLoss(reduction='none')
        element_loss = CE(logits.permute(0, 2, 1), tgt_element_types)
        element_loss = element_loss * (1-tgt_masks)
        element_loss = element_loss.sum(dim=1) / (tgt_masks.sum(dim=1)+1e-6)
        return element_loss
    
    def aroma_loss(self, logits, tgt_aromas, tgt_masks):
        tgt_masks = tgt_masks.bool().float()
        B, L = tgt_aromas.shape[:2]
        BCE = nn.BCEWithLogitsLoss(reduction='none')
        tgt_aromas = tgt_aromas.bool().float()
        logits = logits.view(B, L)
        aroma_loss = BCE(logits, tgt_aromas.float()) #[B, L]
        aroma_loss = aroma_loss * (1-tgt_masks)
        aroma_loss = aroma_loss.sum(dim=1) / (tgt_masks.sum(dim=1)+1e-6)
        return aroma_loss
    
    def charge_loss(self, logits, tgt_charges, tgt_masks):
        tgt_masks = tgt_masks.bool().float()
        CE = nn.CrossEntropyLoss(reduction='none')
        # assumes [B, C, L] (input, target)        
        tgt_charges = tgt_charges.long() + 6
        charge_loss = CE(logits.permute(0, 2, 1), tgt_charges)
        charge_loss = charge_loss * (1-tgt_masks)
        charge_loss = charge_loss.sum(dim=1) / (tgt_masks.sum(dim=1)+1e-6)
        return charge_loss

    def ce_bond_loss(self, bond_logits, tgt_bonds, tgt_masks, padding_masks):
        B, L = tgt_bonds.shape[:2]
        tgt_masks = tgt_masks.float() # 1 iff masked
        or_masks = 1 - torch.einsum("bl,bk->blk", tgt_masks, tgt_masks) # notice that this doesn't mask the edges between target and side products
        and_masks = torch.einsum("bl,bk->blk", 1-tgt_masks, 1-tgt_masks)
        tgt_bonds = tgt_bonds * and_masks

        pad_masks = 1 - padding_masks.float()
        # [B, L], 0 if padding
        pad_masks = torch.einsum("bl,bk->blk", pad_masks, pad_masks)
        # Set the lower triangular part (including diagonal) to -100
        
        if self.force_symbond:
            lower_tri_mask = torch.tril(
                torch.ones(L, L, device=tgt_bonds.device)
            ).unsqueeze(0).expand(B, -1, -1)
            tgt_bonds = tgt_bonds * (1 - lower_tri_mask) + lower_tri_mask * (-100)
        else:
            eye_mask = torch.eye(
                L, device=tgt_bonds.device
            ).unsqueeze(0).expand(B, -1, -1)
            tgt_bonds = tgt_bonds * (1 - eye_mask) + eye_mask * (-100)
        CE = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        ignor_masks = (tgt_bonds != -100)
        loss = CE(bond_logits.permute(0, 3, 1, 2), tgt_bonds.long())
        loss = loss*pad_masks*or_masks*ignor_masks
        loss = loss.sum(dim=(1, 2)) / (pad_masks*or_masks*ignor_masks).sum(dim=(1, 2))
        return loss