import torch
import torch.nn.functional as F
from torch import nn

from utils.misc import NestedTensor
from .backbone import build_backbone
from .transformer import build_vis_transformer, build_transformer, build_de
from .position_encoding import build_position_encoding
from .backbone_inject import QABM


class CNN_MGVLF(nn.Module):
    """
    Multi-scale CNN encoder + cross-modal decoder + QABM (optional).
    """

    def __init__(self, backbone, transformer, DE, position_encoding,
                 max_txt_len: int = 40, hidden_dim: int = 256,
                 use_qabm: bool = False, qabm_internal: bool = False):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.DE = DE
        self.pos = position_encoding
        self.hidden_dim = hidden_dim
        self.use_qabm = use_qabm
        self.qabm_internal = qabm_internal

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

        # Nếu dùng QABM thì chọn internal hoặc external projection
        if self.use_qabm:
            if self.qabm_internal:
                # QABM sẽ tự ép channel từ [256,512,1024,2048] -> hidden_dim
                self.qabm = QABM(c_vis_list=[256, 512, 1024, 2048],
                                 d_txt=768, out_dim=hidden_dim, use_direction=True)
            else:
                # CNN_MGVLF tự ép về hidden_dim, QABM chỉ modulate
                self.stage_projs = nn.ModuleList([
                    nn.Conv2d(256, hidden_dim, kernel_size=1),   # layer1
                    nn.Conv2d(512, hidden_dim, kernel_size=1),   # layer2
                    nn.Conv2d(1024, hidden_dim, kernel_size=1),  # layer3
                    nn.Conv2d(2048, hidden_dim, kernel_size=1),  # layer4
                ])
                self.qabm = QABM(c_vis_list=None,
                                 d_txt=768, out_dim=hidden_dim, use_direction=True)

    @staticmethod
    def _resize_mask(next_feature_map: torch.Tensor, prev_mask: torch.Tensor) -> torch.Tensor:
        assert prev_mask is not None
        mask = F.interpolate(prev_mask[None].float(), size=next_feature_map.shape[-2:]).to(torch.bool)[0]
        return mask

    def forward(self, img, mask, word_mask, wordFeature, sentenceFeature):
        # 1) Backbone multi-layer features + sine pos for each map
        samples = NestedTensor(img, mask)
        features, pos_list = self.backbone(samples)  # list of NestedTensor for layer1..layer4; pos_list aligned

        # Nếu bật QABM
        if self.use_qabm:
            if self.qabm_internal:
                feats = [f.tensors for f in features]  # raw channels [256,512,1024,2048]
                feats_mod, aux = self.qabm(feats, wordFeature, sentenceFeature, word_mask)
                featureMap4 = feats_mod[-1]  # deepest stage sau QABM
            else:
                feats = [proj(f.tensors) for proj, f in zip(self.stage_projs, features)]  # ép về hidden_dim
                feats_mod, aux = self.qabm(feats, wordFeature, sentenceFeature, word_mask)
                featureMap4 = feats_mod[-1]
            mask4 = features[3].mask
        else:
            featureMap4, mask4 = features[3].decompose()     # (B, C=2048, H4, W4)
            featureMap4 = self.input_proj(featureMap4)       # ép về hidden_dim

        bs, _, h, w = featureMap4.shape

        # 2) Create a feature pyramid by further striding after layer4
        x = self.conv6_1(featureMap4)
        conv6_2 = self.conv6_2(x)         # (B, 256, H8, W8)
        x = self.conv7_1(conv6_2)
        conv7_2 = self.conv7_2(x)         # (B, 256, H16, W16)
        x = self.conv8_1(conv7_2)
        conv8_2 = self.conv8_2(x)         # (B, 256, H32, W32)

        # 3) Prepare tokens + masks + positional embeddings for multi-scale concat
        conv5 = featureMap4   # (B, 256, H4, W4)

        fv1 = conv5.view(bs, self.hidden_dim, -1)
        fv2 = conv6_2.view(bs, self.hidden_dim, -1)
        fv3 = conv7_2.view(bs, self.hidden_dim, -1)
        fv4 = conv8_2.view(bs, self.hidden_dim, -1)

        # Visual masks at each scale
        fv2_mask = self._resize_mask(conv6_2, mask4)
        fv3_mask = self._resize_mask(conv7_2, fv2_mask)
        fv4_mask = self._resize_mask(conv8_2, fv3_mask)

        # sinusoidal pos for each scale
        pos1 = pos_list[-1]
        pos2 = self.pos(NestedTensor(conv6_2, fv2_mask)).to(conv6_2.dtype)
        pos3 = self.pos(NestedTensor(conv7_2, fv3_mask)).to(conv7_2.dtype)
        pos4 = self.pos(NestedTensor(conv8_2, fv4_mask)).to(conv8_2.dtype)

        fvpos1 = pos1.view(bs, self.hidden_dim, -1)
        fvpos2 = pos2.view(bs, self.hidden_dim, -1)
        fvpos3 = pos3.view(bs, self.hidden_dim, -1)
        fvpos4 = pos4.view(bs, self.hidden_dim, -1)

        # concat multi-scale visual tokens (sequence-last)
        fv = torch.cat([fv1, fv2, fv3, fv4], dim=2)
        fv = fv.permute(2, 0, 1)  # (N_vis, B, 256)

        # 4) Append language tokens (projected) + sentence token
        textFeature = torch.cat([wordFeature, sentenceFeature.unsqueeze(1)], dim=1)
        fl = self.l_proj(textFeature)  # (B, L+1, 256)
        fl_seq = fl.permute(1, 0, 2)

        # concat visual + language for decoder cross-attn
        fvl = torch.cat([fv, fl_seq], dim=0)

        # 5) Key_padding_mask
        text_pad = ~word_mask.to(torch.bool)
        sentence_mask = torch.zeros((bs, 1), dtype=torch.bool, device=text_pad.device)
        text_mask = torch.cat([text_pad, sentence_mask], dim=1)
        vis_mask = torch.cat([mask4.view(bs, -1),
                              fv2_mask.view(bs, -1),
                              fv3_mask.view(bs, -1),
                              fv4_mask.view(bs, -1)], dim=1)
        fvl_mask = torch.cat([vis_mask, text_mask], dim=1)

        # 6) Positional embeddings
        flpos = self.text_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        fvpos = torch.cat([fvpos1, fvpos2, fvpos3, fvpos4], dim=2).permute(2, 0, 1)
        fvlpos = torch.cat([fvpos, flpos], dim=0)

        # 7) Cross-modal Decoder
        out_layers = self.DE(
            tgt=fv1.permute(2, 0, 1),
            memory=fvl,
            mask=fvl_mask,
            pos=fvlpos,
            query_embed=fvpos1.permute(2, 0, 1)
        )
        fv1_encode = out_layers[-1].permute(1, 2, 0)
        refineFeature = fv1_encode.view(bs, self.hidden_dim, h, w)

        # 8) Visual transformer refinement
        out = self.transformer(refineFeature, mask4, pos1)
        return out


def build_CNN_MGVLF(args, use_qabm=False, qabm_internal=False):
    backbone = build_backbone(args)
    EN = build_vis_transformer(args)
    DE = build_de(args)
    pos = build_position_encoding(args, position_embedding='sine')

    model = CNN_MGVLF(
        backbone=backbone,
        transformer=EN,
        DE=DE,
        position_encoding=pos,
        max_txt_len=getattr(args, "time", 40),
        hidden_dim=getattr(args, "hidden_dim", 256),
        use_qabm=use_qabm,
        qabm_internal=qabm_internal
    )
    return model

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
        pv = self.v_proj(pv)

        # (B, L, 256)
        pl = self.l_proj(fl)

        # retrieve token
        pr = self.pr.weight.expand(bs, -1)
        pr = pr.unsqueeze(1)   # (B, 1, 256)

        # concat sequence: visual + lang + retrieve
        x0 = torch.cat([pv, pl, pr], dim=1)   # (B, N+L+1, 256)

        # positional encoding
        pos = self.pos(x0.permute(0, 2, 1)).to(x0.dtype)  # expects (B,C,T)
        mask = torch.zeros([bs, x0.shape[1]], device=x0.device, dtype=torch.bool)

        memory = self.transformer(x0.permute(0, 2, 1), mask, pos)  # (T,B,C)

        fused = memory[-1]  # retrieve token
        return fused

def build_VLFusion(args):
    transformer = build_transformer(args)
    pos = build_position_encoding(args, position_embedding='learned')
    model = VLFusion(transformer, pos)
    return model
