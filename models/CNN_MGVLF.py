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

        # text positions
        self.text_pos_embed = nn.Embedding(max_txt_len + 1, hidden_dim)

        # downsample after layer4
        self.conv6_1 = nn.Conv2d(2048, 128, 1)
        self.conv6_2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv7_1 = nn.Conv2d(256, 128, 1)
        self.conv7_2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv8_1 = nn.Conv2d(256, 128, 1)
        self.conv8_2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        # project to hidden_dim
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, 1)
        self.l_proj = nn.Sequential(nn.Linear(768, hidden_dim), nn.ReLU(inplace=True))

        # nếu dùng QABM
        if self.use_qabm:
            if self.qabm_internal:
                # QABM tự ép kênh từ raw backbone [256,512,1024,2048]
                self.qabm = QABM(c_vis_list=[256, 512, 1024, 2048],
                                 d_txt=768, out_dim=hidden_dim, use_direction=True)
            else:
                # CNN_MGVLF ép về hidden_dim, QABM chỉ modulate
                self.stage_projs = nn.ModuleList([
                    nn.Conv2d(256, hidden_dim, 1),
                    nn.Conv2d(512, hidden_dim, 1),
                    nn.Conv2d(1024, hidden_dim, 1),
                    nn.Conv2d(2048, hidden_dim, 1),
                ])
                self.qabm = QABM(c_vis_list=None,
                                 d_txt=768, out_dim=hidden_dim, use_direction=True)

    @staticmethod
    def _resize_mask(next_feature_map: torch.Tensor, prev_mask: torch.Tensor) -> torch.Tensor:
        mask = F.interpolate(prev_mask[None].float(), size=next_feature_map.shape[-2:], mode="nearest")
        return mask.to(torch.bool)[0]

    def forward(self, img, mask, word_mask, wordFeature, sentenceFeature):
        samples = NestedTensor(img, mask)
        features, pos_list = self.backbone(samples)

        # QABM
        if self.use_qabm:
            if self.qabm_internal:
                feats = [f.tensors for f in features]
                feats_mod, _ = self.qabm(feats, wordFeature, sentenceFeature, word_mask)
                featureMap4 = feats_mod[-1]
            else:
                feats = [proj(f.tensors) for proj, f in zip(self.stage_projs, features)]
                feats_mod, _ = self.qabm(feats, wordFeature, sentenceFeature, word_mask)
                featureMap4 = feats_mod[-1]
            mask4 = features[3].mask
        else:
            featureMap4, mask4 = features[3].decompose()
            featureMap4 = self.input_proj(featureMap4)

        bs, _, h, w = featureMap4.shape

        # feature pyramid
        x = self.conv6_1(featureMap4)
        conv6_2 = self.conv6_2(x)
        x = self.conv7_1(conv6_2)
        conv7_2 = self.conv7_2(x)
        x = self.conv8_1(conv7_2)
        conv8_2 = self.conv8_2(x)

        conv5 = featureMap4

        fv1 = conv5.view(bs, self.hidden_dim, -1)
        fv2 = conv6_2.view(bs, self.hidden_dim, -1)
        fv3 = conv7_2.view(bs, self.hidden_dim, -1)
        fv4 = conv8_2.view(bs, self.hidden_dim, -1)

        fv2_mask = self._resize_mask(conv6_2, mask4)
        fv3_mask = self._resize_mask(conv7_2, fv2_mask)
        fv4_mask = self._resize_mask(conv8_2, fv3_mask)

        pos1 = pos_list[-1]
        pos2 = self.pos(NestedTensor(conv6_2, fv2_mask)).to(conv6_2.dtype)
        pos3 = self.pos(NestedTensor(conv7_2, fv3_mask)).to(conv7_2.dtype)
        pos4 = self.pos(NestedTensor(conv8_2, fv4_mask)).to(conv8_2.dtype)

        fvpos1 = pos1.view(bs, self.hidden_dim, -1)
        fvpos2 = pos2.view(bs, self.hidden_dim, -1)
        fvpos3 = pos3.view(bs, self.hidden_dim, -1)
        fvpos4 = pos4.view(bs, self.hidden_dim, -1)

        fv = torch.cat([fv1, fv2, fv3, fv4], dim=2).permute(2, 0, 1)

        # language
        textFeature = torch.cat([wordFeature, sentenceFeature.unsqueeze(1)], dim=1)
        fl = self.l_proj(textFeature)
        fl_seq = fl.permute(1, 0, 2)
        fvl = torch.cat([fv, fl_seq], dim=0)

        text_pad = ~word_mask.to(torch.bool)
        sentence_mask = torch.zeros((bs, 1), dtype=torch.bool, device=text_pad.device)
        text_mask = torch.cat([text_pad, sentence_mask], dim=1)
        vis_mask = torch.cat([
            mask4.view(bs, -1),
            fv2_mask.view(bs, -1),
            fv3_mask.view(bs, -1),
            fv4_mask.view(bs, -1)
        ], dim=1)
        fvl_mask = torch.cat([vis_mask, text_mask], dim=1)

        flpos = self.text_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        fvpos = torch.cat([fvpos1, fvpos2, fvpos3, fvpos4], dim=2).permute(2, 0, 1)
        fvlpos = torch.cat([fvpos, flpos], dim=0)

        out_layers = self.DE(
            tgt=fv1.permute(2, 0, 1),
            memory=fvl,
            mask=fvl_mask,
            pos=fvlpos,
            query_embed=fvpos1.permute(2, 0, 1)
        )
        fv1_encode = out_layers[-1].permute(1, 2, 0)
        refineFeature = fv1_encode.view(bs, self.hidden_dim, h, w)

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
    Late fusion of visual tokens + language tokens + one retrieve token.
    """

    def __init__(self, transformer, pos):
        super().__init__()
        self.transformer = transformer
        self.pos = pos
        hidden_dim = transformer.d_model

        self.pr = nn.Embedding(1, hidden_dim)
        self.v_proj = nn.Sequential(nn.Linear(256, hidden_dim), nn.ReLU(inplace=True))
        self.l_proj = nn.Sequential(nn.Linear(768, hidden_dim), nn.ReLU(inplace=True))

    def forward(self, fv, fl):
        bs, c, h, w = fv.shape
        _, L, _ = fl.shape

        # visual flatten (B, N, C)
        pv = fv.view(bs, c, -1).permute(0, 2, 1)
        pv = self.v_proj(pv)

        # language (B, L, C)
        pl = self.l_proj(fl)

        # retrieve token (B,1,C)
        pr = self.pr.weight.expand(bs, -1, -1)

        # concat (B, T, C)
        x0 = torch.cat([pv, pl, pr], dim=1)

        # convert to (B, C, T)
        x0_bt = x0.permute(0, 2, 1)

        # positional encoding
        pos = self.pos(x0_bt).to(x0_bt.dtype)
        mask = torch.zeros([bs, x0.shape[1]], device=x0.device, dtype=torch.bool)

        # transformer expects (B,C,T)
        memory = self.transformer(x0_bt, mask, pos)  # (T,B,C)

        fused = memory[-1]  # retrieve token
        return fused


def build_VLFusion(args):
    transformer = build_transformer(args)
    pos = build_position_encoding(args, position_embedding='learned')
    model = VLFusion(transformer, pos)
    return model
