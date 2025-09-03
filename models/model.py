import torch
from torch import nn
from transformers import AutoModel, AutoConfig
from .CNN_MGVLF import build_VLFusion, build_CNN_MGVLF
from .backbone_inject import QABM

def masked_mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def try_load_partial_state_dict(module: nn.Module, ckpt_url: str, name: str) -> None:
    print(f"[INFO] Downloading and loading checkpoint from {ckpt_url}")
    sd = torch.hub.load_state_dict_from_url(
        ckpt_url, map_location="cpu", check_hash=False
    )
    if "model" in sd:
        sd = sd["model"]
    if "state_dict" in sd:
        sd = sd["state_dict"]

    missing, unexpected = module.load_state_dict(sd, strict=False)
    print(f"[INFO] {name}: loaded (missing={len(missing)}, unexpected={len(unexpected)})")


class MGVLF(nn.Module):
    """
    MGVLF auto-download DETR checkpoint mỗi lần khởi tạo.
    """

    def __init__(self,
                 bert_model: str = "bert-base-uncased",
                 tunebert: bool = True,
                 args=None,
                 detr_ckpt_url: str = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth",
                 init_vl_from_detr: bool = False):
        super().__init__()

        self.tunebert = tunebert

        # ---- Text encoder ----
        self.bert_name = bert_model
        self.bert_config = AutoConfig.from_pretrained(self.bert_name)
        self.textmodel = AutoModel.from_pretrained(self.bert_name)
        self.textdim = getattr(self.bert_config, "hidden_size", 768)
        if not self.tunebert:
            for p in self.textmodel.parameters():
                p.requires_grad = False

        # ---- Visual encoder ----
        self.visumodel = build_CNN_MGVLF(
            args,
            use_qabm=getattr(args, "use_qabm", False),
            qabm_internal=getattr(args, "qabm_internal", False)
        )
        try_load_partial_state_dict(self.visumodel, detr_ckpt_url, "visumodel")


        # ---- VL fusion ----
        self.vlmodel = build_VLFusion(args)
        if init_vl_from_detr:
            try_load_partial_state_dict(self.vlmodel, detr_ckpt_url, "vlmodel")

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

    def forward(self, image, mask, word_id, word_mask):
        outputs = self.textmodel(
            input_ids=word_id,
            attention_mask=word_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.hidden_states
        last_hidden_state = outputs.last_hidden_state
        pooler = getattr(outputs, "pooler_output", None)

        if pooler is None or pooler.numel() == 0:
            sentence_feature = masked_mean_pool(last_hidden_state, word_mask)
        else:
            sentence_feature = pooler

        fl = (hidden_states[-1] + hidden_states[-2] +
              hidden_states[-3] + hidden_states[-4]) / 4.0

        if not self.tunebert:
            fl = fl.detach()
            sentence_feature = sentence_feature.detach()

        fv = self.visumodel(image, mask, word_mask, fl, sentence_feature)
        fused = self.vlmodel(fv, fl)

        outbox = self.Prediction_Head(fused)
        return outbox.sigmoid() * 2.0 - 0.5
