# utils/schedule.py
import math

class WarmupCosine:
    def __init__(self, optimizer, base_lr, warmup_steps, total_steps):
        self.opt = optimizer
        self.base_lr = base_lr
        self.warmup = max(1, int(warmup_steps))
        self.total = max(self.warmup + 1, int(total_steps))
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup:
            lr = self.base_lr * self.step_num / self.warmup
        else:
            t = (self.step_num - self.warmup) / (self.total - self.warmup)
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * t))
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr
