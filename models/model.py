# models/model.py
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.CNN_MGVLF import build_VLFusion, build_CNN_MGVLF
from transformers import AutoModel, AutoConfig


def load_weights(model, load_path):
    """
    Nạp trọng số linh hoạt từ nhiều định dạng checkpoint:
    - {'model': state_dict} hoặc {'state_dict': state_dict} hoặc state_dict thuần.
    Chỉ copy những key trùng tên và cùng shape.
    """
    ckpt = torch.load(load_path, map_location='cpu')
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    current = model.state_dict()
    for k in current.keys():
        if k in state and current[k].shape == state[k].shape:
            current[k] = state[k]
    model.load_state_dict(current, strict=False)
    del current, state, ckpt
    torch.cuda.empty_cache()
    return model


class MGVLF(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', tunebert=True, args=None):
        super(MGVLF, self).__init__()
        self.tunebert = tunebert

        # -------- Text model (HF Transformers) --------
        config = AutoConfig.from_pretrained(
            bert_model,
            output_hidden_states=True,
            add_pooling_layer=True,  # cố gắng có pooler_output nếu checkpoint hỗ trợ
        )
        self.textmodel = AutoModel.from_pretrained(bert_model, config=config)

        # Nếu không fine-tune BERT, tắt grad để tiết kiệm
        if not self.tunebert:
            for p in self.textmodel.parameters():
                p.requires_grad = False

        # -------- Visual model (CNN branch của MGVLF) --------
        self.visumodel = build_CNN_MGVLF(args)
        if args is not None and getattr(args, "pretrain", ""):
            self.visumodel = load_weights(self.visumodel, args.pretrain)

        # -------- Multimodal Fusion model --------
        self.vlmodel = build_VLFusion(args)
        if args is not None and getattr(args, "pretrain", ""):
            self.vlmodel = load_weights(self.vlmodel, args.pretrain)

        # -------- Localization Head --------
        self.Prediction_Head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )
        for p in self.Prediction_Head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image, mask, word_id, word_mask):
        """
        Inputs:
            image: (B, 3, H, W) tensor (đã ToTensor + Normalize)
            mask:  (B, H, W) bool, True = padding (chuẩn DETR)
            word_id:   (B, L) input_ids (có thể là np.array)
            word_mask: (B, L) attention_mask (1=token thật, 0=pad; có thể là np.array)
        Output:
            outbox: (B, 4) bbox normalized [0,1] in format (cx, cy, w, h)
        """
        # Đưa word_id/word_mask về LongTensor đúng device
        if not torch.is_tensor(word_id):
            word_id = torch.as_tensor(word_id, dtype=torch.long, device=image.device)
        else:
            word_id = word_id.to(image.device).long()

        if not torch.is_tensor(word_mask):
            word_mask = torch.as_tensor(word_mask, dtype=torch.long, device=image.device)
        else:
            word_mask = word_mask.to(image.device).long()

        # -------- 1) Language encoder --------
        # attention_mask: 1=real, 0=pad (chuẩn HF)
        outputs = self.textmodel(
            input_ids=word_id,
            attention_mask=word_mask,
            return_dict=True
        )
        hidden_states = outputs.hidden_states  # tuple of (B, L, H)
        # Lấy trung bình 4 lớp cuối như code gốc
        fl = (hidden_states[-1] + hidden_states[-2] + hidden_states[-3] + hidden_states[-4]) / 4.0  # (B, L, H)

        # Lấy câu (pooler) – nếu checkpoint không có pooler, dùng mean-pool theo mask
        sentence_feature = getattr(outputs, "pooler_output", None)
        if sentence_feature is None:
            am = word_mask.unsqueeze(-1).float()             # (B, L, 1)
            sentence_feature = (fl * am).sum(dim=1) / am.sum(dim=1).clamp_min(1.0)  # (B, H)

        if not self.tunebert:
            # nếu đã tắt grad, detach cho chắc (tránh backprop qua BERT)
            fl = fl.detach()
            sentence_feature = sentence_feature.detach()

        # -------- 2) Visual encoder (CNN MGVLF branch) --------
        # mask ảnh ở đây là True=padding (đúng quy ước DETR)
        fv = self.visumodel(image, mask, word_mask, fl, sentence_feature)

        # -------- 3) Fusion encoder --------
        # VLFusion đã được sửa để nhận word_mask (attention_mask) và tự tạo pad mask True=padding
        try:
            x = self.vlmodel(fv, fl, word_mask)  # phiên bản mới đã nhận word_mask
        except TypeError:
            # fallback nếu VLFusion cũ chưa thêm word_mask
            x = self.vlmodel(fv, fl)

        # -------- 4) Localization head --------
        # Đảm bảo lấy đúng token 'pr' (token cuối) nếu vlmodel trả chuỗi
        if x.dim() == 3:
            # chấp nhận (B, C, L) hoặc (B, L, C)
            if x.size(1) == 256:      # (B, 256, L)
                x = x[:, :, -1]       # -> (B, 256)
            elif x.size(2) == 256:    # (B, L, 256)
                x = x[:, -1, :]       # -> (B, 256)
            else:
                raise RuntimeError(f"Unexpected fusion output shape {x.shape}")

        outbox = self.Prediction_Head(x)   # (B, 4): (cx, cy, w, h)
        outbox = outbox.sigmoid()          # đảm bảo [0,1], khớp Reg_Loss/GIoU_Loss
        return outbox
