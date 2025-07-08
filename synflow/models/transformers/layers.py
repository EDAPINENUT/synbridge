from torch import nn
import torch
import torch.nn.functional as F
import math
import pdb
import os
from torch.nn import MultiheadAttention
from torch.distributions.multinomial import Multinomial
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import TransformerDecoder, TransformerDecoderLayer
from synflow.models.misc import append_dims
MAX_BONDS = 6
MAX_DIFF = 4

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings55 have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/dim))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/dim))
        \text{where pos is the word position and i is the embed idx)
    Args:
        dim: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """
    def __init__(self, dim, dropout=0.1, max_len = 192):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = nn.Parameter(pe) # trainable

    def forward(self, l):
        r"""
        returns the additive embedding, notice that addition isnot done in this function
        input shape [l, b, ...] outputshape [l, 1, dim]
        """
        tmp = self.pe[:l, :]
        return self.dropout(tmp)

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, 4*dim)
        self.linear2 = nn.Linear(4*dim, dim)
        self.linear3 = nn.Linear(dim, 4*dim)
        self.linear4 = nn.Linear(4*dim, dim)
        self.linear5 = nn.Linear(dim, dim)
        
    def forward(self, x):
        inter = self.linear1(x)
        inter = F.relu(inter)
        inter = self.linear2(inter)
        x = x + inter
        
        inter = self.linear3(x)
        inter = F.relu(inter)
        inter = self.linear4(inter)
        x = x + inter
        
        return self.linear5(x)


class SinusoidalEmbedding(nn.Module):

    def __init__(self, dim, max_period=10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, timesteps):
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
            
        device = timesteps.device
        half_dim = self.dim // 2
        
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=device) / half_dim
        )
        
        args = timesteps[:, None].float() * freqs[None, :]  # [batch_size, half_dim]
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding


class TimeEmbedding(nn.Module):
    def __init__(self, dim, time_embed_dim=None, max_period=10000.0):
        super().__init__()
        if time_embed_dim is None:
            time_embed_dim = dim * 4
            
        self.time_embed_dim = time_embed_dim
        
        self.sinusoidal_embedding = SinusoidalEmbedding(dim, max_period)
        
        self.mlp = MLP(dim)
        
    def forward(self, timesteps):
        sinusoidal_embedding = self.sinusoidal_embedding(timesteps)
        return self.mlp(sinusoidal_embedding)
    
    
class AtomEncoder(nn.Module):
    def __init__(self, ntoken, dim, dropout=0.1):
        super().__init__()
        self.position_embedding = PositionalEncoding(dim, dropout=dropout)
        self.element_embedding = nn.Embedding(ntoken, dim)
        self.charge_embedding = nn.Embedding(13, dim) #[-6, +6]
        self.aroma_embedding = nn.Embedding(2, dim)
        self.reactant_embedding = nn.Embedding(2, dim)
        self.segment_embedding = nn.Embedding(30, dim)
        self.time_embedding = TimeEmbedding(dim)
        self.mlp = MLP(dim)
        
    def forward(self, element, bond, aroma, charge, segment, flag=None, time=None):
        '''
        element, long [b, l] element index
        bonds, long [b, l, MAX_BONDS]
        aroma, long [b, l]
        charge, long [b, l] +2 +1 0 -1 -2
        
        returns [l, b, dim]
        
        '''
        b, l = element.shape
        # basic information
        element = element.transpose(1, 0) 
        element_embedding = self.element_embedding(element)
        embedding = element_embedding
        #[l, b, dim]

        position_embedding = self.position_embedding(l)
        embedding = embedding + position_embedding
        
        aroma = aroma.transpose(1, 0).long()
        aroma_embedding = self.aroma_embedding(aroma)
        embedding = embedding + aroma_embedding
        
        # additional information
        charge = charge.transpose(1, 0) + 6  
        charge_embedding = self.charge_embedding(charge)
        embedding = embedding + charge_embedding
        
        segment = segment.transpose(1, 0) 
        segment_embedding = self.segment_embedding(segment)
        embedding = embedding + segment_embedding
        
        if not flag is None:
            flag = flag.transpose(1, 0) 
            reactant_embedding = self.reactant_embedding(flag)
            embedding = embedding + reactant_embedding  
        if time is not None:
            time_embedding = self.time_embedding(time)
            embedding = embedding + time_embedding.unsqueeze(0)
            
        message = self.mlp(embedding)
        eye = torch.eye(l).to(element.device)

        if bond.shape[-1] == MAX_BONDS:
            tmp = torch.index_select(eye, dim=0, index=bond.reshape(-1)).view(b, l, MAX_BONDS, l).sum(dim=2) # adjacenct matrix
            tmp = tmp*(1-eye) # remove self loops
            message = torch.einsum("lbd,bkl->kbd", message, tmp)
        else:
            tmp = bond
            tmp = tmp*(1-eye) # remove self loops
            message = torch.einsum("lbd,bkl->kbd", message, tmp)
        
        embedding = embedding + message
        
        return embedding


class BondDecoder(nn.Module):
    def __init__(self, dim, pred_diff, out_logits=None):
        super().__init__()
        self.inc_attention = MultiheadAttention(dim, MAX_DIFF)
        self.inc_qk = nn.Linear(dim, dim*2, bias=False)
        
        self.dec_attention = MultiheadAttention(dim, MAX_DIFF)
        self.dec_qk = nn.Linear(dim, dim*2, bias=False)
        self.pred_diff = pred_diff
        self.cls_layer = nn.Linear(MAX_DIFF, out_logits) if out_logits is not None else None


    def forward(self, molecule_embedding, src_bond, src_mask, prob_shift=0.3, step=1.0):
        """
            mask == True iff masked
            molecule_embedding of shape [l, b, dim]
        """
        l, b, dim = molecule_embedding.shape
        molecule_embedding = molecule_embedding.permute(1, 0, 2)

        q, k = torch.split(self.inc_qk(molecule_embedding), dim, dim=-1)
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), molecule_embedding.permute(1, 0, 2)
        
        if self.cls_layer is not None:
            _, inc = self.inc_attention(q, k, v, key_padding_mask=src_mask, need_weights=True, average_attn_weights=False)
        else:
            _, inc = self.inc_attention(q, k, v, key_padding_mask=src_mask)

        q, k = torch.split(self.dec_qk(molecule_embedding), dim, dim=-1)
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), molecule_embedding.permute(1, 0, 2)
        
        if self.cls_layer is not None:
            _, dec = self.dec_attention(q, k, v, key_padding_mask=src_mask, need_weights=True, average_attn_weights=False)
        else:
            _, dec = self.dec_attention(q, k, v, key_padding_mask=src_mask)
        
        pad_mask = 1 - src_mask.float()
        # [B, L], 0 if padding
        pad_mask = torch.einsum("bl,bk->blk", pad_mask, pad_mask)
        if self.cls_layer is not None:
            diff = self.cls_layer((inc - dec).permute(0,2,3,1))*(MAX_DIFF*pad_mask).unsqueeze(-1)
        else:
            diff = (inc - dec)*MAX_DIFF*pad_mask
        
        eye = torch.eye(src_mask.shape[1]).to(molecule_embedding.device)
        if src_bond.shape[-1] == MAX_BONDS:
            src_weight = torch.index_select(
                eye, dim=0, index=src_bond.reshape(-1)
            ).view(b, l, MAX_BONDS, l).sum(dim=2)*pad_mask
        else:
            src_weight = src_bond*pad_mask

        if self.pred_diff and self.cls_layer is None:
            if torch.is_tensor(step):
                pred_weight = src_weight + diff*append_dims(step, diff.ndim)
            else:
                pred_weight = src_weight + diff*step
        elif self.pred_diff and self.cls_layer is not None:
            diag_mask = torch.eye(l, device=molecule_embedding.device).unsqueeze(0).expand(b, -1, -1)
            src_weight = src_weight * (1 - diag_mask) + diag_mask * (0.0)
            pred_weight = torch.log(self._shift_probs(src_weight.long(), prob_shift, MAX_DIFF)+1e-6) + diff
        else:
            pred_weight = diff
        return pred_weight
    
    def _shift_probs(self, probs, prob_shift, num_classes):
        probs = F.one_hot(probs, num_classes=num_classes)
        probs = probs * (1 - prob_shift) + ((1 - probs) * prob_shift) / (num_classes - 1)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

        # if tgt_bond is None: # inference
        #     # [b, l, l]
        #     bonds = []
        #     pred_weight = (pred_weight + pred_weight.permute(0, 2, 1))/2
        #     for i in range(MAX_BONDS):
        #         bonds += [pred_weight.argmax(2)]
        #         pred_weight -= torch.index_select(eye, dim=0, index=bonds[-1].reshape(-1)).view(b, l, l)
        #     pred_bond = torch.stack(bonds, dim =2)
        #     return pred_bond
            
        # else: # training
        #     tgt_mask = tgt_mask.float() # 1 iff masked
        #     or_mask = 1 - torch.einsum("bl,bk->blk", tgt_mask, tgt_mask) # notice that this doesn't mask the edges between target and side products
        #     and_mask = torch.einsum("bl,bk->blk", 1-tgt_mask, 1-tgt_mask)
        
        #     tgt_weight = torch.index_select(eye, dim=0, index=tgt_bond.reshape(-1)).view(b, l, MAX_BONDS, l).sum(dim=2)*and_mask
        #     error = pred_weight - tgt_weight
        #     error = error*error*pad_mask*or_mask
        #     loss = error.sum(dim=(1, 2))
        #     return {'bond_loss':loss}
