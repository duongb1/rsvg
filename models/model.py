# models/model.py
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.CNN_MGVLF import build_VLFusion, build_CNN_MGVLF
from transformers import AutoModel, AutoConfig
from utils.coords import generate_coord 


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
        self.args = args

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

        # ===== (1) Weighted Layer Pooling (4 lớp cuối của BERT) =====
        self.wlp_weights = nn.Parameter(torch.ones(4) / 4)

        # -------- Visual model (CNN branch của MGVLF) --------
        self.visumodel = build_CNN_MGVLF(args)
        if args is not None and getattr(args, "pretrain", ""):
            self.visumodel = load_weights(self.visumodel, args.pretrain)

        # ===== (2) LVFE-lite: FiLM gating bằng sentence_feature =====
        self.film_gamma = nn.Linear(768, 256)
        self.film_beta  = nn.Linear(768, 256)

        # ===== (Optional) Coordinate prior (8 kênh) =====
        self.use_coord = getattr(args, "use_coord", False) if args is not None else False
        if self.use_coord:
            self.coord_proj = nn.Conv2d(8, 256, kernel_size=1, bias=False)
            # scale nhỏ để không lấn át đặc trưng ảnh lúc đầu
            self.coord_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # -------- Multimodal Fusion model --------
        self.vlmodel = build_VLFusion(args)
        if args is not None and getattr(args, "pretrain", ""):
            self.vlmodel = load_weights(self.vlmodel, args.pretrain)

        # -------- Localization Heads --------
        # Box head + Quality head (q̂≈IoU)
        self.box_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1), nn.Sigmoid()
        )
        # Alias giữ tương thích code cũ
        self.Prediction_Head = self.box_head

        for m in list(self.box_head.modules()) + list(self.quality_head.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Số retrieval tokens được kỳ vọng ở cuối chuỗi fusion để pooling
        self.num_retrieval = getattr(args, 'num_retrieval', 1) if args is not None else 1

        # chỗ tạm giữ phụ trợ (không ảnh hưởng API)
        self._last_qhat = None

    @staticmethod
    def _weighted_layer_pool(last4, w):
        """last4: (4,B,L,H), w: learnable weights len=4 -> (B,L,H)"""
        w = torch.softmax(w, dim=0).view(4, 1, 1, 1)
        return (last4 * w).sum(0)

    def _apply_film(self, fv, sentence_feature):
        """
        Áp dụng FiLM-lite vào đặc trưng thị giác fv dựa trên câu.
        - Nếu fv (B,256,H,W): nhân/cộng theo không gian.
        - Nếu fv (B,N,256): áp dụng theo token.
        - Nếu dạng khác -> trả nguyên vẹn.
        """
        if fv is None:
            return None
        gamma = torch.tanh(self.film_gamma(sentence_feature))  # (B,256)
        beta  = self.film_beta(sentence_feature)               # (B,256)

        if fv.dim() == 4 and fv.size(1) == 256:  # (B,256,H,W)
            B, C, H, W = fv.shape
            g = gamma.view(B, C, 1, 1)
            b = beta.view(B, C, 1, 1)
            return fv * (1 + g) + b
        elif fv.dim() == 3 and fv.size(-1) == 256:  # (B,N,256)
            B, N, C = fv.shape
            g = gamma.view(B, 1, C)
            b = beta.view(B, 1, C)
            return fv * (1 + g) + b
        else:
            return fv

    def forward(self, image, mask, word_id, word_mask, return_aux: bool = False):
        """
        Inputs:
            image: (B, 3, H, W) tensor (đã ToTensor + Normalize)
            mask:  (B, H, W) bool, True = padding (chuẩn DETR)
            word_id:   (B, L) input_ids (có thể là np.array)
            word_mask: (B, L) attention_mask (1=token thật, 0=pad; có thể là np.array)
            return_aux: nếu True, trả thêm qhat để tính quality-loss ở ngoài.
        Output:
            outbox: (B, 4) bbox normalized [0,1] in format (cx, cy, w, h)
            (tuỳ chọn) aux: {'qhat': (B,1)}
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
        outputs = self.textmodel(
            input_ids=word_id,
            attention_mask=word_mask,
            return_dict=True
        )
        hidden_states = outputs.hidden_states  # tuple of (B, L, H)

        # WLP thay vì avg 4 lớp cuối
        last4 = torch.stack(hidden_states[-4:], dim=0)  # (4,B,L,H)
        fl = self._weighted_layer_pool(last4, self.wlp_weights)  # (B,L,H)

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

        # (Optional) Inject coordinate prior nếu fv là map 2D
        if self.use_coord and fv is not None and fv.dim() == 4 and fv.size(1) == 256:
            B, C, H, W = fv.shape
            coord = generate_coord(B, H, W, device=fv.device, dtype=fv.dtype)  # (B,8,H,W)
            fv = fv + self.coord_proj(coord) * self.coord_scale

        # (FiLM-lite) tăng cường fv theo câu
        fv = self._apply_film(fv, sentence_feature)

        # -------- 3) Fusion encoder --------
        # VLFusion đã được sửa để nhận word_mask (attention_mask) và tự tạo pad mask True=padding
        try:
            x = self.vlmodel(fv, fl, word_mask)  # phiên bản mới đã nhận word_mask
        except TypeError:
            # fallback nếu VLFusion cũ chưa thêm word_mask
            x = self.vlmodel(fv, fl)

        # -------- 4) Lấy đặc trưng truy hồi từ đầu ra Fusion --------
        # Hỗ trợ các khả năng: (B,256,L) hoặc (B,L,256) hoặc vector (B,256).
        # Nếu self.num_retrieval>1 và vlmodel sắp xếp K token cuối là retrieval tokens,
        # ta sẽ mean-pool các token cuối; nếu không, lấy token cuối như cũ.
        if x.dim() == 3:
            B = x.size(0)
            if x.size(1) == 256:      # (B,256,L)
                L = x.size(2)
                K = min(self.num_retrieval, L)
                feat = x[:, :, -K:].mean(dim=2)     # (B,256)
            elif x.size(2) == 256:    # (B,L,256)
                L = x.size(1)
                K = min(self.num_retrieval, L)
                feat = x[:, -K:, :].mean(dim=1)     # (B,256)
            else:
                raise RuntimeError(f"Unexpected fusion output shape {x.shape}")
        elif x.dim() == 2 and x.size(1) == 256:
            feat = x  # (B,256)
        else:
            raise RuntimeError(f"Unsupported fusion output shape {x.shape}")

        # -------- 5) Heads --------
        outbox = self.box_head(feat).sigmoid()  # (B,4) in [0,1]
        qhat = self.quality_head(feat)          # (B,1) in [0,1]
        self._last_qhat = qhat

        if return_aux:
            return outbox, {"qhat": qhat}
        return outbox
