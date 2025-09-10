# utils/ema.py
import copy
import torch

class EMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for p, q in zip(model.parameters(), self.ema.parameters()):
            if q.data.dtype.is_floating_point:
                q.data.mul_(d).add_(p.data, alpha=1.0 - d)
