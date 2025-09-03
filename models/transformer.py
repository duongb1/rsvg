import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

# ---------------------------
# Visual Transformer (Encoder)
# ---------------------------
class Transformer_vis(nn.Module):
    """
    Visual Transformer Encoder.
    Input:
        src: (B, C, H, W)
        mask: (B, H, W) boolean (True = pad)
        pos_embed: (B, C, H, W)
    Output:
        (B, C, H, W)
    """

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_before=False):
        super().__init__()
        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                        dropout, activation, normalize_before)
        enc_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(layer, num_encoder_layers, enc_norm)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, mask: Tensor, pos_embed: Tensor) -> Tensor:
        # flatten NxCxHxW -> HWxNxC
        b, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)        # (HW, B, C)
        pos = pos_embed.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        key_padding = mask.flatten(1)                # (B, HW)
        memory = self.encoder(src, src_key_padding_mask=key_padding, pos=pos)  # (HW, B, C)
        return memory.permute(1, 2, 0).view(b, c, h, w)


# ---------------------------
# Decoder (Cross-Attention)
# ---------------------------
class Transformer_Decoder(nn.Module):
    """
    Cross-attention decoder stack.
    forward:
        tgt: (Tq, B, C)
        memory: (Tm, B, C)
        mask: (B, Tm) key_padding for memory+text
        pos: (Tm, B, C) positional for memory
        query_pos: (Tq, B, C) positional for queries
    return:
        if return_intermediate: (num_layers, Tq, B, C)
        else: (1, Tq, B, C)
    """

    def __init__(self, d_model=256, nhead=8, num_decoder_layers=1,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_before=False, return_intermediate_dec=True):
        super().__init__()
        layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                        dropout, activation, normalize_before)
        dec_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(layer, num_decoder_layers, dec_norm,
                                          return_intermediate=return_intermediate_dec)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt: Tensor, memory: Tensor, mask: Tensor,
                pos: Tensor, query_embed: Tensor) -> Tensor:
        # tgt: (Tq,B,C), memory: (Tm,B,C)
        hs = self.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=mask,  # (B, Tm)
            pos=pos,
            query_pos=query_embed
        )
        return hs  # (num_layers, Tq, B, C) if return_intermediate else (1, Tq, B, C)


# ---------------------------
# Sequence Transformer (Encoder-only)
# ---------------------------
class Transformer(nn.Module):
    """
    Encoder-only transformer for late fusion.
    Input:
        src: (B, C, T)
        mask: (B, T) boolean
        pos_embed: (B, T, C)
    Output:
        memory: (T, B, C)
    """

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=0, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=True):
        super().__init__()
        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                        dropout, activation, normalize_before)
        enc_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(layer, num_encoder_layers, enc_norm)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, mask: Tensor, pos_embed: Tensor) -> Tensor:
        # src: (B, C, T) -> (T, B, C)
        src = src.permute(2, 0, 1)
        pos = pos_embed.permute(1, 0, 2)  # (T, B, C)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos)
        return memory  # (T, B, C)


# ---------------------------
# Encoder / Decoder stacks
# ---------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module = None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask,
                           pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int,
                 norm: nn.Module = None, return_intermediate: bool = False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None) -> Tensor:
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


# ---------------------------
# Encoder/Decoder Layers
# ---------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", normalize_before: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src: Tensor,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None) -> Tensor:
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src: Tensor,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None) -> Tensor:
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", normalize_before: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt: Tensor, memory: Tensor,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None) -> Tensor:
        # self-attn on queries
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross-attn with memory
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt: Tensor, memory: Tensor,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None) -> Tensor:
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask,
                                    pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 pos, query_pos)


# ---------------------------
# Builders
# ---------------------------
def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def build_vis_transformer(args):
    return Transformer_vis(
        d_model=getattr(args, "hidden_dim", 256),
        dropout=getattr(args, "dropout", 0.1),
        nhead=getattr(args, "nheads", 8),
        dim_feedforward=getattr(args, "dim_feedforward", 2048),
        num_encoder_layers=getattr(args, "enc_layers", 6),
        normalize_before=getattr(args, "pre_norm", False),
    )


def build_de(args):
    return Transformer_Decoder(
        d_model=getattr(args, "hidden_dim", 256),
        dropout=getattr(args, "dropout", 0.1),
        nhead=getattr(args, "nheads", 8),
        dim_feedforward=getattr(args, "dim_feedforward", 2048),
        num_decoder_layers=1,
        normalize_before=getattr(args, "pre_norm", False),
        return_intermediate_dec=True
    )


def build_transformer(args):
    return Transformer(
        d_model=getattr(args, "hidden_dim", 256),
        dropout=getattr(args, "dropout", 0.1),
        nhead=getattr(args, "nheads", 8),
        dim_feedforward=getattr(args, "dim_feedforward", 2048),
        num_encoder_layers=getattr(args, "enc_layers", 6),
        num_decoder_layers=getattr(args, "dec_layers", 0),  # encoder-only
        normalize_before=getattr(args, "pre_norm", False),
        return_intermediate_dec=True,
    )


# ---------------------------
# Activation helper
# ---------------------------
def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
