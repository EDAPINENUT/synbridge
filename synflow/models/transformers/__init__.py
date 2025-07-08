from torch import nn
from synflow.models.transformers.layers import (
    AtomEncoder, 
    TransformerEncoder, 
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    BondDecoder
)
import torch
from synflow.chem.constants import element_id_to_type
# from vector_quantize_pytorch import FSQ, VectorQuantize

class VariationalEncoder(nn.Module):
    def __init__(self, dim, nhead, nlayer, dropout):
        super().__init__()
        layer = TransformerDecoderLayer(dim, nhead, dim, dropout)
        self.transformer_decoder = TransformerDecoder(layer, nlayer)
        self.head = nn.Linear(dim, 2*dim)

    def KL(self, posterior):
        # prior is standard gaussian distribution
        mu, logsigma = posterior['mu'], posterior['logsigma']
        # Clip logsigma to prevent numerical instability
        logsigma = torch.clamp(logsigma, -20, 2)
        # no matter what shape
        logvar = logsigma*2
        # Compute KL divergence with numerical stability considerations
        kl_per_element = 0.5 * (mu * mu + torch.exp(logvar) - 1 - logvar)
        # Replace potential NaN values with zeros (they shouldn't contribute to the loss)
        kl_per_element = torch.nan_to_num(kl_per_element, nan=0.0, posinf=1e5, neginf=0.0)
        
        loss = torch.sum(kl_per_element, 1)
        return loss

    def forward(self, src, src_mask, tgt, tgt_mask):
        """
        src, tgt [L, b, dim]
        src_mask, tgt_mask, [B, L]
        """
        l, b, dim = src.shape
        src_mask, tgt_mask = src_mask.permute(0, 1), tgt_mask.permute(0, 1)
        decoder_output = self.transformer_decoder(src, tgt,
                                                  memory_key_padding_mask=tgt_mask, 
                                                  tgt_key_padding_mask=src_mask).permute(1, 2, 0)
        # [L, B, dim] to [B, dim, L]
        tmp = decoder_output * (1-src_mask.float().unsqueeze(1))
        tmp = tmp.mean(dim=2)
        # [B, dim]
        posterior = self.head(tmp)
        result = {}
        result['mu'], result['logsigma'] = torch.split(posterior, dim, dim=-1)
        return result, self.KL(result)


class VariationalDecoder(nn.Module):
    def __init__(self, dim, nhead, nlayer, dropout, pred_diff=False, out_logits=None):
        super().__init__()
        layer = TransformerEncoderLayer(dim, nhead, dim, dropout)
        self.transformer_encoder = TransformerEncoder(layer, nlayer)
        self.latent_head = nn.Linear(dim, dim)
        self.bond_decoder = BondDecoder(dim, pred_diff, out_logits)
        self.charge_aroma_head = nn.Linear(dim, 13 + 1)#-6 to +6
        self.element_type_head = nn.Linear(dim, len(element_id_to_type) + 1)

    def forward(self, src_emb, src_bond, src_mask, latent=None, interm=None, decoder_only=False):
        l, b, dim = src_emb.size()
        if decoder_only:
            latent = torch.zeros(b, dim).to(src_emb.device)
        else:
            if latent is None:
                latent = torch.randn(b, dim).to(src_emb.device)
            else:
                latent = latent['mu'] + torch.randn(b, dim).to(src_emb.device) * torch.exp(latent['logsigma'])
        
        latent = self.latent_head(latent)
        src_emb = src_emb + latent.expand(l, b, dim)
        if interm is not None:
            src_emb = src_emb + interm
        result = {}
        encoder_output = self.transformer_encoder(src_emb, src_key_padding_mask=src_mask)
        result['bond_outputs'] = self.bond_decoder(encoder_output, src_bond, src_mask, step=1.0)

        charge_aroma_logit = self.charge_aroma_head(encoder_output)
        aroma_logits, charge_logits = charge_aroma_logit.split([1, 13], dim=-1)
        result['aroma_logits'] = aroma_logits.permute(1, 0, 2)
        result['charge_logits'] = charge_logits.permute(1, 0, 2)
        
        element_type_logit = self.element_type_head(encoder_output)
        result['element_type_logits'] = element_type_logit.permute(1, 0, 2)

        return result


class Encoder(nn.Module):
    def __init__(self, ntoken, dim, nhead, nlayer, dropout):
        super().__init__()
        self.atom_encoder = AtomEncoder(ntoken, dim, dropout=dropout)
        layer = TransformerEncoderLayer(dim, nhead, dim, dropout)
        self.transformer_encoder = TransformerEncoder(layer, nlayer)

    def forward(self, element, bond, aroma, charge, mask, segment, flag=None, time=None):
        '''
        element, long [b, l] element index
        bonds, long [b, l, 4]
        aroma, long [b, l]
        charge, long [b, l] +1 0 -1
        mask, [b, l] true if masked
        returns [l, b, dim]
        '''
        embedding = self.atom_encoder(element, bond, aroma, charge, segment, flag, time)

        encoder_output = self.transformer_encoder(embedding, src_key_padding_mask=mask)
        return encoder_output


class MergeEncoder(nn.Module):
    def __init__(self, dim, nhead, nlayer=1, dropout=0.1, ntoken=2):
        super().__init__()
        self.task_embedding = nn.Embedding(ntoken, dim)
        self.cross_attn = torch.nn.ModuleList(
            [nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True) for _ in range(nlayer)]
        )
        self.norm = torch.nn.ModuleList(
            [nn.LayerNorm(dim) for _ in range(nlayer)]
        )
        self.ffn = torch.nn.ModuleList(
            [nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(nlayer)]
        )
        self.norm2 = torch.nn.ModuleList(
            [nn.LayerNorm(dim) for _ in range(nlayer)]
        )
        self.nlayer = nlayer

    def forward(self, task_token, token_emb, token_emb_padding_mask=None):
        """
        task_token: Tensor of shape (B, 1, 1)
        token_emb: Tensor of shape (N, B, D)
        token_emb_padding_mask: Bool Tensor of shape (B, N), True for padding positions
        returns: Tensor of shape (N, B, D)
        """
        task_token = self.task_embedding(task_token)
        token_emb = token_emb.permute(1, 0, 2)
        # Cross-attention: task_token attends to token_emb
        for i in range(self.nlayer):
            attn_out, _ = self.cross_attn[i](
                query=token_emb,       # [B, N, D]
                key=task_token,        # [B, 1, D]
                value=task_token       # [B, 1, D]
            )
            token_emb = self.norm[i](token_emb + attn_out)
            # Feed-forward
            ffn_out = self.ffn[i](token_emb)
            token_emb = self.norm2[i](token_emb + ffn_out)
            if token_emb_padding_mask is not None:
                # [B, N, 1] -> broadcast -> [B, N, D]
                mask = token_emb_padding_mask.unsqueeze(-1)  # True for padding
                token_emb = token_emb.masked_fill(mask, 0.0)
        return token_emb.permute(1, 0, 2)  # [B, N, D]


def get_encoder(config):
    return Encoder(**config)

def get_merge_encoder(config):
    return MergeEncoder(**config)

def get_variational_encoder(config):
    return VariationalEncoder(**config)

def get_variational_decoder(config):
    return VariationalDecoder(**config)