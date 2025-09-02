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


def _try_load_partial_state_dict(module: nn.Module, sd) -> bool:
    """
    Cố gắng nạp state_dict vào `module` ở chế độ non-strict (partial init).
    Hỗ trợ cả đường dẫn (dict đã load) lẫn dict trực tiếp.
    """
    try:
        if isinstance(sd, str):
            if not os.path.isfile(sd):
                return False
            sd = torch.load(sd, map_location="cpu")
            # cho phép bọc trong khoá 'model'
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
    Load weight từ DETR ResNet-50 pretrained có sẵn trong torchvision,
    KHÔNG cần file .pth ngoài. Nạp theo non-strict để tận dụng tối đa các lớp trùng tên.
    """
    try:
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

    Pipeline:
      - Text encoder: BERT (bert-base-uncased mặc định) => token features (B,L,768) + sentence feature (B,768)
      - Visual encoder: CNN_MGVLF => multi-scale fusion feature (B,256,H',W')
      - VL Fusion: VLFusion => pooled vector (B,256)
      - Head: MLP => bbox (cx, cy, w, h) chuẩn hóa trong (-0.5, 1.5) sau khi map

    Điểm khác biệt chính:
      - Không cần tải thủ công file 'detr-r50-e632da11.pth'.
        Thay vào đó, ta load trực tiếp pretrained DETR từ torchvision
        và nạp partial vào các module phù hợp.
    """

    def __init__(self,
                 bert_model: str = "bert-base-uncased",
                 tunebert: bool = True,
                 args: Optional[object] = None,
                 use_torchvision_detr_init: bool = True,
                 also_init_vl_from_detr: bool = True):
        """
        Args:
            bert_model: tên model BERT trong HuggingFace.
            tunebert: có fine-tune BERT hay không.
            args: cấu hình cho các module vision/VL (được build ở .CNN_MGVLF).
            use_torchvision_detr_init: nếu True, sẽ lấy weight từ torchvision DETR để init phần nhìn.
            also_init_vl_from_detr: nếu True, thử nạp partial cả vào VL fusion (an toàn vì non-strict).
        """
        super().__init__()

        self.tunebert = tunebert

        # ---- Text encoder (BERT) ----
        self.bert_name = bert_model
        self.bert_config = AutoConfig.from_pretrained(self.bert_name)
        self.textmodel = AutoModel.from_pretrained(self.bert_name)
        self.textdim = getattr(self.bert_config, "hidden_size", 768)

        if not self.tunebert:
            for p in self.textmodel.parameters():
                p.requires_grad = False

        # ---- Visual encoder (CNN+Transformer) ----
        self.visumodel = build_CNN_MGVLF(args)

        # (khuyến nghị) khởi tạo từ torchvision DETR nếu sẵn
        if use_torchvision_detr_init:
            loaded = _try_load_from_torchvision_detr(self.visumodel)
            if not loaded:
                print("[INFO] Fallback: skip DETR init for visual backbone.")

        # ---- VL fusion encoder ----
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
                image: torch.Tensor,       # (B, 3, H, W), normalized
                mask: torch.Tensor,        # (B, H, W) bool, True = PAD
                word_id: torch.Tensor,     # (B, L)
                word_mask: torch.Tensor    # (B, L)
                ) -> torch.Tensor:
        """
        Trả về: outbox (B, 4) với format (cx, cy, w, h)
        Sau head: sigmoid * 2 - 0.5 => phạm vi (-0.5, 1.5) cho linh hoạt biên ảnh.
        """
        # ---- Text encoding ----
        outputs = self.textmodel(
            input_ids=word_id,
            attention_mask=word_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.hidden_states             # tuple(len=layers+1) of (B,L,D)
        last_hidden_state = outputs.last_hidden_state     # (B,L,D)
        pooler = getattr(outputs, "pooler_output", None)  # có thể None

        # Sentence feature
        if pooler is None or pooler.numel() == 0:
            sentence_feature = _masked_mean_pool(last_hidden_state, word_mask)  # (B,D)
        else:
            sentence_feature = pooler

        # Token features: trung bình 4 lớp cuối
        fl = (hidden_states[-1] + hidden_states[-2] + hidden_states[-3] + hidden_states[-4]) / 4.0

        if not self.tunebert:
            fl = fl.detach()
            sentence_feature = sentence_feature.detach()

        # ---- Visual encoder (multi-scale) ----
        fv = self.visumodel(image, mask, word_mask, fl, sentence_feature)  # (B,256,H',W')

        # ---- VL Fusion (pooled vector) ----
        fused = self.vlmodel(fv, fl)  # (B, 256) — theo thiết kế của VLFusion

        # ---- BBox head ----
        outbox = self.Prediction_Head(fused)     # (B, 4) — (cx, cy, w, h) unbounded
        outbox = outbox.sigmoid() * 2.0 - 0.5    # map sang (-0.5, 1.5)
        return outbox
