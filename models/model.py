# models/model.py
import os
from typing import Optional

import torch
from torch import nn
from transformers import AutoModel, AutoConfig
import torchvision

from .CNN_MGVLF import build_VLFusion, build_CNN_MGVLF


def _masked_mean_pool(last_hidden_state: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pooling theo mask trên chuỗi (B, L, D) -> (B, D).
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B, L, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                  # (B, D)
    denom = mask.sum(dim=1).clamp(min=1e-6)                         # (B, 1)
    return summed / denom


def _try_load_partial_state_dict(module: nn.Module, sd) -> bool:
    """
    Cố gắng nạp state_dict vào `module` ở chế độ non-strict (partial init).
    """
    try:
        if isinstance(sd, str):
            if not os.path.isfile(sd):
                return False
            sd = torch.load(sd, map_location="cpu")
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]

        if not isinstance(sd, dict):
            return False

        missing, unexpected = module.load_state_dict(sd, strict=False)
        print(
            "[INFO] Partially loaded weights\n"
            f" - missing: {len(missing)} keys, unexpected: {len(unexpected)} keys"
        )
        return True
    except Exception as e:
        print(f"[WARN] Could not partially load weights: {e}")
        return False


def _try_load_from_torchvision_detr(module: nn.Module) -> bool:
    """
    Load weight từ DETR ResNet-50 pretrained có sẵn trong torchvision.
    """
    try:
        tv_ver = tuple(int(x) for x in torchvision.__version__.split('+')[0].split('.'))
        if tv_ver < (0, 15, 0):
            print(f"[INFO] torchvision {torchvision.__version__} < 0.15: skip DETR init.")
            return False

        from torchvision.models.detection import detr_resnet50
        from torchvision.models.detection.detr import Detr_ResNet50_Weights

        detr_model = detr_resnet50(weights=Detr_ResNet50_Weights.COCO_V1)
        sd = detr_model.state_dict()
        missing, unexpected = module.load_state_dict(sd, strict=False)
        print("[INFO] Partially loaded from torchvision DETR (COCO_V1)")
        print(f" - missing: {len(missing)} keys, unexpected: {len(unexpected)} keys")
        return True
    except Exception as e:
        print(f"[WARN] Could not load DETR weights from torchvision: {e}")
        return False


class MGVLF(nn.Module):
    """
    Multi-Granularity Visual-Language Fusion model for text-guided localization.
    """

    def __init__(self,
                 bert_model: str = "bert-base-uncased",
                 tunebert: bool = True,
                 args: Optional[object] = None,
                 use_torchvision_detr_init: bool = True,
                 also_init_vl_from_detr: bool = True):
        super().__init__()

        self.tunebert = tunebert

        # ---- Text encoder (BERT) ----
        self.bert_name = bert_model
        try:
            self.bert_config = AutoConfig.from_pretrained(self.bert_name, local_files_only=True)
            self.textmodel = AutoModel.from_pretrained(self.bert_name, local_files_only=True)
        except Exception:
            try:
                self.bert_config = AutoConfig.from_pretrained(self.bert_name)
                self.textmodel = AutoModel.from_pretrained(self.bert_name)
            except Exception as ee:
                raise RuntimeError(
                    f"Không thể tải BERT '{self.bert_name}'. "
                    f"Hãy đặt checkpoint vào thư mục local (vd. /kaggle/input/bert-base-uncased) "
                    f"và chạy với --bert_model /kaggle/input/bert-base-uncased. Lỗi: {ee}"
                )
        self.textdim = getattr(self.bert_config, "hidden_size", 768)

        if not self.tunebert:
            self.textmodel.eval()
            for p in self.textmodel.parameters():
                p.requires_grad = False
        else:
            # Freeze n lớp đầu nếu có yêu cầu
            n_freeze = int(getattr(args, "bert_freeze_layers", 0) or 0)
            if n_freeze > 0 and hasattr(self.textmodel, "encoder"):
                layers = self.textmodel.encoder.layer
                for li in range(min(n_freeze, len(layers))):
                    for p in layers[li].parameters():
                        p.requires_grad = False
            # Gradient checkpointing
            if getattr(args, "bert_grad_ckpt", False):
                if hasattr(self.textmodel, "gradient_checkpointing_enable"):
                    self.textmodel.gradient_checkpointing_enable()
                else:
                    print("[INFO] BERT model does not support gradient_checkpointing_enable().")

        # ---- Visual encoder ----
        self.visumodel = build_CNN_MGVLF(args)
        if use_torchvision_detr_init:
            loaded = _try_load_from_torchvision_detr(self.visumodel)
            if not loaded:
                print("[INFO] Fallback: skip DETR init for visual backbone.")

        # ---- VL fusion ----
        self.vlmodel = build_VLFusion(args)
        if use_torchvision_detr_init and also_init_vl_from_detr:
            _ = _try_load_from_torchvision_detr(self.vlmodel)

        # ---- Prediction head ----
        hidden_dim = getattr(args, "hidden_dim", 256)
        self.Prediction_Head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )
        for m in self.Prediction_Head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,
                image: torch.Tensor,       # (B, 3, H, W)
                mask: torch.Tensor,        # (B, H, W) bool
                word_id: torch.Tensor,     # (B, L)
                word_mask: torch.Tensor    # (B, L)
                ) -> torch.Tensor:
        """
        Trả về: outbox (B, 4) với format (cx, cy, w, h)
        """
        # ---- Text encoding ----
        outputs = self.textmodel(
            input_ids=word_id,
            attention_mask=word_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.hidden_states
        last_hidden_state = outputs.last_hidden_state
        pooler = getattr(outputs, "pooler_output", None)

        # Sentence feature
        if pooler is None or pooler.numel() == 0:
            sentence_feature = _masked_mean_pool(last_hidden_state, word_mask)
        else:
            sentence_feature = pooler

        # Token features: trung bình 4 lớp cuối
        fl = (hidden_states[-1] + hidden_states[-2] +
              hidden_states[-3] + hidden_states[-4]) / 4.0

        if not self.tunebert:
            fl = fl.detach()
            sentence_feature = sentence_feature.detach()

        # ---- Visual encoder ----
        fv = self.visumodel(image, mask, word_mask, fl, sentence_feature)  # (B,256,H',W')

        # ---- VL Fusion ----
        fused = self.vlmodel(fv, fl)  # (B, 256)

        # ---- BBox head ----
        outbox = self.Prediction_Head(fused)  # (B, 4)
        outbox = outbox.sigmoid() * 2.0 - 0.5
        return outbox
