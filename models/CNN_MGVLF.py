import torch
import torch.nn.functional as F
from torch import nn

from utils.misc import NestedTensor
from .backbone import build_backbone
from .transformer import build_vis_transformer, build_transformer, build_de
from .position_encoding import build_position_encoding

class CNN_MGVLF(nn.Module):
    """
    Multi-scale CNN encoder + cross-modal decoder to refine the final visual feature map.

    Input:
        - img:   (B, 3, H, W), normalized
        - mask:  (B, H, W) boolean, True = padded (ignored)
        - word_mask: (B, L) attention mask for text (1 for real tokens)
        - wordFeature: (B, L, D_text) token features from BERT
        - sentenceFeature: (B, D_text) pooled sentence feature

    Output:
        - refineFeature: (B, 256, H4, W4)   # feature at 1/32 scale (from resnet layer4) refined by DE + EN
    """

    def __init__(self, backbone, transformer, DE, position_encoding, max_txt_len: int = 40, hidden_dim: int = 256):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer      # visual-only transformer encoder (returns (B,256,H,W))
        self.DE = DE                        # one-layer cross-decoder to inject text into deepest visual tokens
        self.pos = position_encoding
        self.hidden_dim = hidden_dim

        # text positions for (max_txt_len + 1 [sentence token])
        self.text_pos_embed = nn.Embedding(max_txt_len + 1, hidden_dim)

        # downsample blocks after layer4 to form multi-scale chain
        self.conv6_1 = nn.Conv2d(in_channels=2048, out_channels=128, kernel_size=1, stride=1)
        self.conv6_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv7_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.conv7_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv8_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.conv8_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)

        # project visual channel to transformer width
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        # project language hidden into transformer width
        self.l_proj = nn.Sequential(nn.Linear(768, hidden_dim), nn.ReLU(inplace=True))

    @staticmethod
    def _resize_mask(next_feature_map: torch.Tensor, prev_mask: torch.Tensor) -> torch.Tensor:
        """
        Upsample boolean mask to the size of next_feature_map.
        prev_mask: (B, Hprev, Wprev) -> returns (B, Hnext, Wnext)
        """
        assert prev_mask is not None
        mask = F.interpolate(prev_mask[None].float(), size=next_feature_map.shape[-2:]).to(torch.bool)[0]
        return mask

    def forward(self, img, mask, word_mask, wordFeature, sentenceFeature):
        """
        Returns:
            (B, 256, H4, W4) refined by cross-modal DE + visual Transformer EN
        """
        # 1) Backbone multi-layer features + sine pos for each map
        samples = NestedTensor(img, mask)
        features, pos_list = self.backbone(samples)  # list of NestedTensor for layer1..layer4; pos_list aligned

        # we use deepest visual map (layer4) as base
        featureMap4, mask4 = features[3].decompose()     # (B, C=2048, H4, W4), (B, H4, W4)
        bs, _, h, w = featureMap4.shape

        # 2) Create a feature pyramid by further striding after layer4
        x = self.conv6_1(featureMap4)
        conv6_2 = self.conv6_2(x)         # (B, 256, H8, W8)
        x = self.conv7_1(conv6_2)
        conv7_2 = self.conv7_2(x)         # (B, 256, H16, W16)
        x = self.conv8_1(conv7_2)
        conv8_2 = self.conv8_2(x)         # (B, 256, H32, W32)

        # 3) Prepare tokens + masks + positional embeddings for multi-scale concat
        conv5 = self.input_proj(featureMap4)   # (B, 256, H4, W4)

        fv1 = conv5.view(bs, self.hidden_dim, -1)         # (B, 256, H4*W4)
        fv2 = conv6_2.view(bs, self.hidden_dim, -1)       # (B, 256, H8*W8)
        fv3 = conv7_2.view(bs, self.hidden_dim, -1)       # (B, 256, H16*W16)
        fv4 = conv8_2.view(bs, self.hidden_dim, -1)       # (B, 256, H32*W32)

        # Visual masks at each scale
        fv2_mask = self._resize_mask(conv6_2, mask4)      # (B, H8, W8)
        fv3_mask = self._resize_mask(conv7_2, fv2_mask)   # (B, H16, W16)
        fv4_mask = self._resize_mask(conv8_2, fv3_mask)   # (B, H32, W32)

        # sinusoidal pos for each scale
        pos1 = pos_list[-1]                                # (B, 256, H4, W4)
        pos2 = self.pos(NestedTensor(conv6_2, fv2_mask)).to(conv6_2.dtype)  # (B, 256, H8, W8)
        pos3 = self.pos(NestedTensor(conv7_2, fv3_mask)).to(conv7_2.dtype)  # (B, 256, H16, W16)
        pos4 = self.pos(NestedTensor(conv8_2, fv4_mask)).to(conv8_2.dtype)  # (B, 256, H32, W32)

        fvpos1 = pos1.view(bs, self.hidden_dim, -1)
        fvpos2 = pos2.view(bs, self.hidden_dim, -1)
        fvpos3 = pos3.view(bs, self.hidden_dim, -1)
        fvpos4 = pos4.view(bs, self.hidden_dim, -1)

        # concat multi-scale visual tokens (sequence-last)
        fv = torch.cat([fv1, fv2, fv3, fv4], dim=2)       # (B, 256, N_vis)
        fv = fv.permute(2, 0, 1)                          # (N_vis, B, 256) for decoder input

        # 4) Append language tokens (projected) + sentence token
        textFeature = torch.cat([wordFeature, sentenceFeature.unsqueeze(1)], dim=1)   # (B, L+1, 768)
        fl = self.l_proj(textFeature)                     # (B, L+1, 256)
        fl_seq = fl.permute(1, 0, 2)                      # (L+1, B, 256)

        # concat visual + language for decoder cross-attn
        fvl = torch.cat([fv, fl_seq], dim=0)              # (N_vis + L+1, B, 256)

        # 5) Build key_padding_mask for both visual and text parts (True = masked/padded)
        #    text mask: invert word_mask and add one valid token for sentence
        text_pad = ~word_mask.to(torch.bool)              # (B, L), True where padding
        sentence_mask = torch.zeros((bs, 1), dtype=torch.bool, device=text_pad.device)
        text_mask = torch.cat([text_pad, sentence_mask], dim=1)  # (B, L+1)

        vis_mask = torch.cat([
            mask4.view(bs, -1),
            fv2_mask.view(bs, -1),
            fv3_mask.view(bs, -1),
            fv4_mask.view(bs, -1)
        ], dim=1)                                         # (B, N_vis)
        fvl_mask = torch.cat([vis_mask, text_mask], dim=1)  # (B, N_vis + L+1)

        # 6) Positional embeddings for fused sequence (visual sine + learned text positions)
        flpos = self.text_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)   # (L+1, B, 256)
        fvpos = torch.cat([fvpos1, fvpos2, fvpos3, fvpos4], dim=2)         # (B, 256, N_vis)
        fvpos = fvpos.permute(2, 0, 1)                                     # (N_vis, B, 256)
        fvlpos = torch.cat([fvpos, flpos], dim=0)                          # (N_vis + L+1, B, 256)

        # 7) Cross-modal Decoder on deepest tokens only (fv1 tokens as queries)
        out_layers = self.DE(
            tgt=fv1.permute(2, 0, 1),          # (N1=H4*W4, B, 256) queries
            memory=fvl,                        # (N_vis+L+1, B, 256) keys/values
            mask=fvl_mask,                     # (B, N_all)
            pos=fvlpos,                        # (N_all, B, 256)
            query_embed=fvpos1.permute(2, 0, 1)  # (N1, B, 256)
        )
        # take last layer output
        fv1_encode = out_layers[-1].permute(1, 2, 0)      # (B, 256, N1)
        refineFeature = fv1_encode.view(bs, self.hidden_dim, h, w)  # (B, 256, H4, W4)

        # 8) Visual transformer encoder refinement
        out = self.transformer(refineFeature, mask4, pos1)  # (B, 256, H4, W4)
        return out


class VLFusion(nn.Module):
    """
    Late fusion of:
        - Visual tokens pooled (flatten(HxW), project)
        - Language tokens (B,L,768) -> project
        - One learned "retrieve" embedding

    Uses a transformer (encoder-only) over the concatenated sequence and returns the
    last token (retrieve) as fused vector (B, 256).
    """

    def __init__(self, transformer, pos):
        super().__init__()
        self.transformer = transformer
        self.pos = pos
        hidden_dim = transformer.d_model

        self.pr = nn.Embedding(1, hidden_dim)   # retrieve token
        self.v_proj = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True))
        self.l_proj = nn.Sequential(nn.Linear(768, 256), nn.ReLU(inplace=True))

    def forward(self, fv, fl):
        """
        fv: (B, 256, H, W)
        fl: (B, L, 768)
        return: (B, 256) fused vector
        """
        bs, c, h, w = fv.shape
        _, L, _ = fl.shape

        # (B, H*W, 256)
        pv = fv.view(bs, c, -1).permute(0, 2, 1)
        pv = self.v_proj(pv)                   # (B, N, 256)

        # (B, L, 256)
        pl = self.l_proj(fl)

        # retrieve token
        pr = self.pr.weight.expand(bs, -1)     # (B, 256)
        pr = pr.unsqueeze(2)                   # (B, 256, 1)

        # concat along sequence dimension last (-> (B, 256, T))
        x0 = torch.cat([pv.permute(0, 2, 1),    # (B,256,N)
                        pl.permute(0, 2, 1),    # (B,256,L)
                        pr], dim=2)             # (B,256,N+L+1)

        # learned positional embedding expects (B, C, T) -> returns (B, T, C)
        pos = self.pos(x0).to(x0.dtype)        # (B, T, 256)
        mask = torch.zeros([bs, x0.shape[2]], device=x0.device, dtype=torch.bool)  # no padding in late fusion

        # transformer encoder over sequence (see models/transformer.py)
        memory = self.transformer(x0, mask, pos)   # returned as (T, B, C)

        # take the last position (retrieve token) -> (B, C)
        fused = memory[-1]  # (B, 256)
        return fused


def build_CNN_MGVLF(args):
    """
    Factory: visual encoder with cross-modal decode + visual transformer refine.
    """
    backbone = build_backbone(args)                      # ResNet + FrozenBN
    EN = build_vis_transformer(args)                     # visual transformer encoder
    DE = build_de(args)                                  # one-layer decoder with cross-attention
    pos = build_position_encoding(args, position_embedding='sine')

    model = CNN_MGVLF(
        backbone=backbone,
        transformer=EN,
        DE=DE,
        position_encoding=pos,
        max_txt_len=getattr(args, "time", 40),
        hidden_dim=getattr(args, "hidden_dim", 256)
    )
    return model


def build_VLFusion(args):
    """
    Factory: late fusion module over concatenated V/L tokens (sequence Transformer).
    """
    transformer = build_transformer(args)                # encoder-only transformer
    pos = build_position_encoding(args, position_embedding='learned')
    model = VLFusion(transformer, pos)
    return model
