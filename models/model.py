import os
from typing import Optional

import torch
from torch import nn
from transformers import AutoModel, AutoConfig

from .CNN_MGVLF import build_VLFusion, build_CNN_MGVLF


def _masked_mean_pool(last_hidden_state: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pooling theo mask trên chuỗi (B, L, D) -> (B, D).
    """
    # last_hidden_state: (B, L, D), attention_mask: (B, L)
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B, L, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                   # (B, D)
    denom = mask.sum(dim=1).clamp(min=1e-6)                          # (B, 1)
    return summed / denom


def _try_load_partial_weights(module: nn.Module, ckpt_path: str, strict_key: str = "model") -> bool:
    """
    Cố gắng load state_dict một phần (không strict) nếu tồn tại file.
    Trả về True nếu load được, ngược lại False.
    """
    try:
        if not ckpt_path or not os.path.isfile(ckpt_path):
            return False
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and strict_key in sd:
            sd = sd[strict_key]
        missing, unexpected = module.load_state_dict(sd, strict=False)
        # Không raise nếu thiếu/ dư key — chấp nhận partial init
        print(f"[INFO] Partially loaded weights from: {ckpt_path}\n"
              f" - missing: {len(missing)} keys, unexpected: {len(unexpected)} keys")
        return True
    except Exception as e:
        print(f"[WARN] Could not load partial weights from {ckpt_path}: {e}")
        return False


class MGVLF(nn.Module):
    """
    Multi-Granularity Visual-Language Fusion model for text-guided localization.

    Pipeline:
      - Text encoder: BERT (bert-base-uncased mặc định) => token features (B,L,768) + sentence feature (B,768)
      - Visual encoder: CNN_MGVLF => multi-scale fusion feature (B,256,H',W')
      - VL Fusion: VLFusion => pooled vector (B,256)
      - Head: MLP => bbox (cx, cy, w, h) chuẩn hóa ~ [0,1], sau đó map về (-0.5, 1.5) để linh hoạt
    """

    def __init__(self,
                 bert_model: str = "bert-base-uncased",
                 tunebert: bool = True,
                 args: Optional[object] = None,
                 init_visual_from: str = "./saved_models/detr-r50-e632da11.pth",
                 init_vl_from: str = "./saved_models/detr-r50-e632da11.pth"):
        super().__init__()

        self.tunebert = tunebert

        # ---- Text encoder (BERT) ----
        # Luôn dùng return_dict + hidden states để lấy 4 layer cuối
        self.bert_name = bert_model
        self.bert_config = AutoConfig.from_pretrained(self.bert_name)
        self.textmodel = AutoModel.from_pretrained(self.bert_name)
        self.textdim = getattr(self.bert_config, "hidden_size", 768)

        # Có thể freeze BERT nếu không tune
        if not self.tunebert:
            for p in self.textmodel.parameters():
                p.requires_grad = False

        # ---- Visual encoder (CNN+Transformer) ----
        self.visumodel = build_CNN_MGVLF(args)

        # (tùy chọn) thử load partial weights nếu có
        _try_load_partial_weights(self.visumodel, init_visual_from)

        # ---- VL fusion encoder ----
        self.vlmodel = build_VLFusion(args)
        _try_load_partial_weights(self.vlmodel, init_vl_from)

        # ---- Prediction head ----
        self.Prediction_Head = nn.Sequential(
            nn.Linear(args.hidden_dim if hasattr(args, "hidden_dim") else 256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )
        # init Xavier cho các layer Linear
        for m in self.Prediction_Head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,
                image: torch.Tensor,       # (B, 3, H, W), normalized
                mask: torch.Tensor,        # (B, H, W) bool, True = PAD
                word_id: torch.Tensor,     # (B, L)
                word_mask: torch.Tensor    # (B, L)
                ) -> torch.Tensor:
        """
        Trả về: outbox (B, 4) với format (cx, cy, w, h) trong khoảng gần [0,1]
                sau đó được map sigmoid*2 - 0.5 => (-0.5, 1.5)
        """
        # ---- Text encoding ----
        # Lấy hidden states từ 4 layer cuối để làm token features
        outputs = self.textmodel(
            input_ids=word_id,
            attention_mask=word_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.hidden_states  # tuple(len=layers+1) of (B,L,D)
        last_hidden_state = outputs.last_hidden_state  # (B,L,D)
        pooler = getattr(outputs, "pooler_output", None)  # có thể None (với vài model)

        # Sentence feature: nếu không có pooler_output -> dùng mean pooling theo mask
        if pooler is None or pooler.numel() == 0:
            sentence_feature = _masked_mean_pool(last_hidden_state, word_mask)
        else:
            sentence_feature = pooler  # (B, D)

        # Token features: trung bình 4 lớp cuối
        fl = (hidden_states[-1] + hidden_states[-2] + hidden_states[-3] + hidden_states[-4]) / 4.0
        # Nếu không tune BERT -> detach để tránh backprop vào BERT
        if not self.tunebert:
            fl = fl.detach()
            sentence_feature = sentence_feature.detach()

        # ---- Visual encoder (multi-scale) ----
        # visumodel kỳ vọng: (img, mask, word_mask, token_feats(B,L,D), sent_feat(B,D))
        fv = self.visumodel(image, mask, word_mask, fl, sentence_feature)  # (B,256,H',W')

        # ---- VL Fusion (ra vector) ----
        fused = self.vlmodel(fv, fl)  # (B, 256) — theo thiết kế của VLFusion

        # ---- BBox head ----
        outbox = self.Prediction_Head(fused)          # (B, 4) — (cx, cy, w, h) unbounded
        outbox = outbox.sigmoid() * 2.0 - 0.5         # map sang (-0.5, 1.5) để linh hoạt
        return outbox
