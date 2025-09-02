"""
Misc helpers: NestedTensor, distributed utils, safe interpolate
"""
import datetime
import pickle
from collections import defaultdict, deque
from typing import List, Optional

import torch
import torch.distributed as dist
import torchvision
from torch import Tensor


class SmoothedValue:
    """Track a series of values and provide smoothed stats."""
    def __init__(self, window_size=20, fmt=None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / max(1, self.count)

    @property
    def max(self):
        return max(self.deque) if self.deque else 0.0

    @property
    def value(self):
        return self.deque[-1] if self.deque else 0.0

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg,
            global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def log_every(self, iterable, print_freq, header=""):
        i = 0
        start_time = datetime.datetime.now().timestamp()
        end = start_time
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header, "[{0:{width}d}/{1}]",
                "eta: {eta}", "{meters}",
                "time: {time}", "data: {data}", "max mem: {memory:.0f}"
            ])
        else:
            log_msg = self.delimiter.join([
                header, "[{0:{width}d}/{1}]",
                "eta: {eta}", "{meters}", "time: {time}", "data: {data}"
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(datetime.datetime.now().timestamp() - end)
            yield obj
            iter_time.update(datetime.datetime.now().timestamp() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string, meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB,
                        width=len(str(len(iterable)))
                    ))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string, meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        width=len(str(len(iterable)))
                    ))
            i += 1
            end = datetime.datetime.now().timestamp()
        total_time = datetime.datetime.now().timestamp() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time/len(iterable):.4f} s / it)")


def all_gather(data):
    """gather arbitrary picklable data across ranks"""
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(x.item()) for x in size_list]
    max_size = max(size_list)
    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device="cuda") for _ in size_list]
    if local_size != max_size:
        padding = torch.empty((max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)
    data_list = []
    for size, t in zip(size_list, tensor_list):
        buf = t.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buf))
    return data_list


def reduce_dict(input_dict, average=True):
    """reduce tensor values in a dict across processes"""
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names, values = [], []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        return {k: v for k, v in zip(names, values)}


class NestedTensor:
    def __init__(self, tensors: Tensor, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_t = self.tensors.to(device)
        cast_m = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(cast_t, cast_m)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def _max_by_axis(the_list: List[List[int]]) -> List[int]:
    maxes = the_list[0]
    for sub in the_list[1:]:
        for i, item in enumerate(sub):
            maxes[i] = max(maxes[i], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    assert tensor_list[0].ndim == 3, "Only 3D tensors (C,H,W) supported"
    if torchvision._is_tracing():
        return _onnx_nested_tensor_from_tensor_list(tensor_list)
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    batch = [len(tensor_list)] + max_size
    b, c, h, w = batch
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch, dtype=dtype, device=device)
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        m[: img.shape[1], : img.shape[2]] = False
    return NestedTensor(tensor, mask)


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)
    padded_imgs, padded_masks = [], []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)
        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))
    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)
    return NestedTensor(tensor, mask=mask)


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def interpolate(input: Tensor, size=None, scale_factor=None, mode="nearest", align_corners=None) -> Tensor:
    # Torchvision empty-batch wrapper (kept simple)
    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)
