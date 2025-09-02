# main.py
import os
import warnings

# --- Giảm ồn TensorFlow/XLA trong môi trường Kaggle ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import absl.logging as _absl_logging
    _absl_logging.set_verbosity(_absl_logging.ERROR)
except Exception:
    pass
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")

import argparse
import time
import logging
import datetime
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader

# ---- schedulers từ HuggingFace ----
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

# ---- local imports ----
from dataset import RSVGDataset
from models.model import MGVLF
from utils.loss import Reg_Loss, GIoU_Loss
from utils.utils import AverageMeter, xyxy2xywh, bbox_iou, adjust_learning_rate
from utils.checkpoint import save_checkpoint, load_pretrain, load_resume

# AMP
from torch.cuda.amp import autocast, GradScaler


def parse_args():
    parser = argparse.ArgumentParser(description="MGVLF Remote Sensing Grounding (Kaggle/Colab-ready)")
    # data
    parser.add_argument('--images_path', type=str, default='/kaggle/input/dior-rsvg/DIOR_RSVG/JPEGImages')
    parser.add_argument('--anno_path', type=str, default='/kaggle/input/dior-rsvg/DIOR_RSVG/Annotations')
    parser.add_argument('--split_root', type=str, default='/kaggle/input/dior-rsvg/DIOR_RSVG')
    parser.add_argument('--size', default=640, type=int)
    parser.add_argument('--time', default=40, type=int)
    # train
    parser.add_argument('--nb_epoch', default=150, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_dec', default=0.1, type=float)
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--print_freq', default=50, type=int)
    parser.add_argument('--savename', default='default', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--pretrain', default='', type=str)
    parser.add_argument('--tunebert', default=True, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    # model
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--position_embedding', default='sine', choices=('sine', 'learned'))
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=441, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    # perf toggles
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--use_torchvision_detr_init', action='store_true', default=False)
    parser.add_argument('--also_init_vl_from_detr', action='store_true', default=False)
    # HF BERT controls
    parser.add_argument('--bert_freeze_layers', type=int, default=0)
    parser.add_argument('--bert_lr', type=float, default=None)
    parser.add_argument('--bert_llrd', type=float, default=0.95)
    parser.add_argument('--bert_grad_ckpt', action='store_true', default=False)
    parser.add_argument('--sched_warmup_ratio', type=float, default=0.06)
    parser.add_argument('--sched_type', type=str, default='linear', choices=['linear', 'cosine'])
    return parser.parse_args()


def set_seed(seed: int):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 3)


def build_dataloaders(args):
    tfm = Compose([ToTensor(),
                   Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    train_dataset = RSVGDataset(args.images_path, args.anno_path, args.split_root,
                                'train', args.size, tfm, args.time, args.bert_model)
    val_dataset = RSVGDataset(args.images_path, args.anno_path, args.split_root,
                              'val', args.size, tfm, args.time, args.bert_model)
    test_dataset = RSVGDataset(args.images_path, args.anno_path, args.split_root,
                               'test', args.size, tfm, args.time, args.bert_model, testmode=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=True, num_workers=args.workers,
                              persistent_workers=(args.workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, drop_last=True, num_workers=args.workers,
                            persistent_workers=(args.workers > 0))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=True, drop_last=True, num_workers=0)

    print('trainset:', len(train_dataset), 'validationset:', len(val_dataset), 'testset:', len(test_dataset))
    return train_loader, val_loader, test_loader


def build_model(args):
    model = MGVLF(bert_model=args.bert_model,
                  tunebert=args.tunebert,
                  args=args,
                  use_torchvision_detr_init=args.use_torchvision_detr_init,
                  also_init_vl_from_detr=args.also_init_vl_from_detr)
    model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    device = torch.device('cuda' if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    model = model.to(device)
    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled.")
        except Exception as e:
            print(f"torch.compile skipped: {e}")
    return model


def _no_decay(name: str) -> bool:
    return any(nd in name for nd in ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"])


def optimizer_for(model: nn.Module, args):
    """Safe optimizer grouping with LLRD for BERT + DP-safe."""
    m = model.module if isinstance(model, nn.DataParallel) else model

    if args.tunebert:
        visu_param = list(m.visumodel.parameters())
        text_param = list(m.textmodel.parameters())
        visu_ids = {id(p) for p in visu_param}
        text_ids = {id(p) for p in text_param}
        rest_param = [p for p in m.parameters() if id(p) not in visu_ids and id(p) not in text_ids]

        # ---- Layer-wise LR decay cho BERT ----
        bert_groups = []
        bert_lr_base = (args.bert_lr if args.bert_lr is not None else args.lr / 10.0)
        llrd = float(args.bert_llrd)
        if hasattr(m.textmodel, "encoder") and hasattr(m.textmodel.encoder, "layer"):
            layers = list(m.textmodel.encoder.layer)
            n_layers = len(layers)
            for li in range(n_layers):
                lr = bert_lr_base * (llrd ** (n_layers - 1 - li))
                decay, nodecay = [], []
                for name, param in layers[li].named_parameters():
                    full = f"textmodel.encoder.layer.{li}.{name}"
                    (nodecay if _no_decay(full) else decay).append(param)
                if decay: bert_groups.append({'params': decay, 'lr': lr, 'weight_decay': 1e-4})
                if nodecay: bert_groups.append({'params': nodecay, 'lr': lr, 'weight_decay': 0.0})
        else:
            decay, nodecay = [], []
            for name, param in m.textmodel.named_parameters():
                (nodecay if _no_decay(name) else decay).append(param)
            if decay: bert_groups.append({'params': decay, 'lr': bert_lr_base, 'weight_decay': 1e-4})
            if nodecay: bert_groups.append({'params': nodecay, 'lr': bert_lr_base, 'weight_decay': 0.0})

        # visu + rest
        visu_decay, visu_nodecay = [], []
        for name, p in m.visumodel.named_parameters():
            (visu_nodecay if _no_decay(name) else visu_decay).append(p)
        rest_decay, rest_nodecay = [], []
        for p in rest_param:
            (rest_nodecay if _no_decay("") else rest_decay).append(p)

        param_groups = []
        if rest_decay: param_groups.append({'params': rest_decay, 'lr': args.lr, 'weight_decay': 1e-4})
        if rest_nodecay: param_groups.append({'params': rest_nodecay, 'lr': args.lr, 'weight_decay': 0.0})
        if visu_decay: param_groups.append({'params': visu_decay, 'lr': args.lr / 10.0, 'weight_decay': 1e-4})
        if visu_nodecay: param_groups.append({'params': visu_nodecay, 'lr': args.lr / 10.0, 'weight_decay': 0.0})
        param_groups.extend(bert_groups)

        opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=1e-4)
        print(f'visu/text/rest params: {sum(p.numel() for p in visu_param)}, '
              f'{sum(p.numel() for p in text_param)}, '
              f'{sum(p.numel() for p in rest_param)}')
        return opt

    else:
        visu_param = list(m.visumodel.parameters())
        rest_param = [p for p in m.parameters() if id(p) not in {id(v) for v in visu_param}]
        opt = torch.optim.AdamW(
            [{'params': rest_param},
             {'params': visu_param, 'lr': args.lr / 10.0}],
            lr=args.lr, weight_decay=1e-4
        )
        print(f'visu/rest params: {sum(p.numel() for p in visu_param)}, '
              f'{sum(p.numel() for p in rest_param)}')
        return opt



def train_epoch(train_loader, model, optimizer, epoch, args, scheduler=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    l1_losses = AverageMeter()
    giou_losses = AverageMeter()
    acc5 = AverageMeter(); acc6 = AverageMeter(); acc7 = AverageMeter(); acc8 = AverageMeter(); acc9 = AverageMeter()
    meanIoU = AverageMeter()
    inter_area = AverageMeter(); union_area = AverageMeter()

    model.train()
    device = next(model.parameters()).device
    end = time.time()
    scaler = GradScaler(enabled=torch.cuda.is_available())

    for batch_idx, (imgs, masks, word_id, word_mask, gt_bbox) in enumerate(train_loader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        masks = masks[:, :, :, 0] == 255  # boolean
        word_id = torch.as_tensor(word_id, device=device).long()
        word_mask = torch.as_tensor(word_mask, device=device).long()
        gt_bbox = torch.as_tensor(gt_bbox, device=device)
        gt_bbox = torch.clamp(gt_bbox, min=0, max=args.size - 1)

        with autocast(enabled=torch.cuda.is_available()):
            pred_bbox = model(imgs, masks, word_id, word_mask)  # (B,4)
            giou_loss = GIoU_Loss(pred_bbox * (args.size - 1), gt_bbox, args.size - 1)
            gt_bbox_xywh = xyxy2xywh(gt_bbox)
            l1_loss = Reg_Loss(pred_bbox, gt_bbox_xywh / (args.size - 1))
            loss = giou_loss + l1_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        # meters
        losses.update(loss.item(), imgs.size(0))
        l1_losses.update(l1_loss.item(), imgs.size(0))
        giou_losses.update(giou_loss.item(), imgs.size(0))

        # metrics
        pred_xyxy = torch.cat([pred_bbox[:, :2] - (pred_bbox[:, 2:] / 2),
                               pred_bbox[:, :2] + (pred_bbox[:, 2:] / 2)], dim=1)
        pred_xyxy = pred_xyxy * (args.size - 1)
        iou, interA, unionA = bbox_iou(pred_xyxy.detach().cpu(), gt_bbox.detach().cpu(), x1y1x2y2=True)

        cumInter = float(interA.sum().item())
        cumUnion = float(unionA.sum().item())
        meanIoU.update(float(iou.mean().item()), imgs.size(0))
        inter_area.update(cumInter); union_area.update(cumUnion)
        iou_np = iou.numpy()
        bs = imgs.size(0)
        acc5.update((iou_np > 0.5).sum() / bs, bs)
        acc6.update((iou_np > 0.6).sum() / bs, bs)
        acc7.update((iou_np > 0.7).sum() / bs, bs)
        acc8.update((iou_np > 0.8).sum() / bs, bs)
        acc9.update((iou_np > 0.9).sum() / bs, bs)

        batch_time.update(time.time() - end); end = time.time()

        if batch_idx % args.print_freq == 0:
            # in đủ LR từng param group
            lr_str = [f"lr[{i}] {g['lr']:.6e}" for i, g in enumerate(optimizer.param_groups)]
            print(f"Epoch[{epoch}] [{batch_idx}/{len(train_loader)}] "
                  f"acc@0.5 {acc5.avg:.4f} | acc@0.6 {acc6.avg:.4f} | acc@0.7 {acc7.avg:.4f} | "
                  f"acc@0.8 {acc8.avg:.4f} | acc@0.9 {acc9.avg:.4f} | "
                  f"mIoU {meanIoU.avg:.4f} | cumIoU {inter_area.sum/ max(1e-6, union_area.sum):.4f} | "
                  f"Loss {losses.avg:.4f} | L1 {l1_losses.avg:.4f} | GIoU {giou_losses.avg:.4f} | "
                  + " | ".join(lr_str))
    return (acc5.avg, acc6.avg, acc7.avg, acc8.avg, acc9.avg,
            meanIoU.avg, inter_area.sum / max(1e-6, union_area.sum), losses.avg)


@torch.no_grad()
def validate_epoch(val_loader, model, args):
    batch_time = AverageMeter()
    losses = AverageMeter(); l1_losses = AverageMeter(); giou_losses = AverageMeter()
    acc5 = AverageMeter(); acc6 = AverageMeter(); acc7 = AverageMeter(); acc8 = AverageMeter(); acc9 = AverageMeter()
    meanIoU = AverageMeter(); inter_area = AverageMeter(); union_area = AverageMeter()

    model.eval()
    device = next(model.parameters()).device
    end = time.time()
    print(datetime.datetime.now())

    for batch_idx, (imgs, masks, word_id, word_mask, bbox) in enumerate(val_loader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        masks = masks[:, :, :, 0] == 255
        word_id = torch.as_tensor(word_id, device=device).long()
        word_mask = torch.as_tensor(word_mask, device=device).long()
        bbox = torch.as_tensor(bbox, device=device)
        bbox = torch.clamp(bbox, min=0, max=args.size - 1)

        pred_bbox = model(imgs, masks, word_id, word_mask)
        giou_loss = GIoU_Loss(pred_bbox * (args.size - 1), bbox, args.size - 1)
        l1_loss = Reg_Loss(pred_bbox, xyxy2xywh(bbox) / (args.size - 1))
        loss = giou_loss + l1_loss

        losses.update(loss.item(), imgs.size(0))
        l1_losses.update(l1_loss.item(), imgs.size(0))
        giou_losses.update(giou_loss.item(), imgs.size(0))

        pred_xyxy = torch.cat([pred_bbox[:, :2] - (pred_bbox[:, 2:] / 2),
                               pred_bbox[:, :2] + (pred_bbox[:, 2:] / 2)], dim=1)
        pred_xyxy = pred_xyxy * (args.size - 1)
        iou, interA, unionA = bbox_iou(pred_xyxy.detach().cpu(), bbox.detach().cpu(), x1y1x2y2=True)

        cumInter = float(interA.sum().item())
        cumUnion = float(unionA.sum().item())
        meanIoU.update(float(iou.mean().item()), imgs.size(0))
        inter_area.update(cumInter); union_area.update(cumUnion)
        iou_np = iou.numpy()
        bs = imgs.size(0)
        acc5.update((iou_np > 0.5).sum() / bs, bs)
        acc6.update((iou_np > 0.6).sum() / bs, bs)
        acc7.update((iou_np > 0.7).sum() / bs, bs)
        acc8.update((iou_np > 0.8).sum() / bs, bs)
        acc9.update((iou_np > 0.9).sum() / bs, bs)

        batch_time.update(time.time() - end); end = time.time()

        if batch_idx % args.print_freq == 0:
            print(f"[VAL {batch_idx}/{len(val_loader)}] "
                  f"acc@0.5 {acc5.avg:.4f} | acc@0.6 {acc6.avg:.4f} | acc@0.7 {acc7.avg:.4f} | "
                  f"acc@0.8 {acc8.avg:.4f} | acc@0.9 {acc9.avg:.4f} | "
                  f"mIoU {meanIoU.avg:.4f} | cumIoU {inter_area.sum/ max(1e-6, union_area.sum):.4f} | "
                  f"Loss {losses.avg:.4f}")
    final = (acc5.avg, acc6.avg, acc7.avg, acc8.avg, acc9.avg,
             meanIoU.avg, inter_area.sum / max(1e-6, union_area.sum), losses.avg)
    print(f"VAL Final: acc@0.5 {final[0]:.4f} | acc@0.6 {final[1]:.4f} | acc@0.7 {final[2]:.4f} | "
          f"acc@0.8 {final[3]:.4f} | acc@0.9 {final[4]:.4f} | mIoU {final[5]:.4f} | cumIoU {final[6]:.4f} | Loss {final[7]:.4f}")
    return final


@torch.no_grad()
def test_epoch(test_loader, model, args):
    batch_time = AverageMeter()
    acc5 = AverageMeter(); acc6 = AverageMeter(); acc7 = AverageMeter(); acc8 = AverageMeter(); acc9 = AverageMeter()
    meanIoU = AverageMeter(); inter_area = AverageMeter(); union_area = AverageMeter()

    model.eval()
    device = next(model.parameters()).device
    end = time.time()

    for batch_idx, (imgs, masks, word_id, word_mask, bbox, ratio, dw, dh, im_id, phrase) in enumerate(test_loader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        masks = masks[:, :, :, 0] == 255
        word_id = torch.as_tensor(word_id, device=device).long()
        word_mask = torch.as_tensor(word_mask, device=device).long()
        bbox = torch.as_tensor(bbox, device=device)
        bbox = torch.clamp(bbox, min=0, max=args.size - 1)

        pred_bbox = model(imgs, masks, word_id, word_mask)
        pred_xyxy = torch.cat([pred_bbox[:, :2] - (pred_bbox[:, 2:] / 2),
                               pred_bbox[:, :2] + (pred_bbox[:, 2:] / 2)], dim=1) * (args.size - 1)

        iou, interA, unionA = bbox_iou(pred_xyxy.detach().cpu(), bbox.detach().cpu(), x1y1x2y2=True)
        cumInter = float(interA.sum().item()); cumUnion = float(unionA.sum().item())

        meanIoU.update(float(iou.mean().item()), imgs.size(0))
        inter_area.update(cumInter); union_area.update(cumUnion)
        iou_np = iou.numpy()
        acc5.update((iou_np > 0.5).sum() / 1, 1)
        acc6.update((iou_np > 0.6).sum() / 1, 1)
        acc7.update((iou_np > 0.7).sum() / 1, 1)
        acc8.update((iou_np > 0.8).sum() / 1, 1)
        acc9.update((iou_np > 0.9).sum() / 1, 1)

        batch_time.update(time.time() - end); end = time.time()

        if batch_idx % args.print_freq == 0:
            print(f"[TEST {batch_idx}/{len(test_loader)}] "
                  f"acc@0.5 {acc5.avg:.4f} | acc@0.6 {acc6.avg:.4f} | acc@0.7 {acc7.avg:.4f} | "
                  f"acc@0.8 {acc8.avg:.4f} | acc@0.9 {acc9.avg:.4f} | "
                  f"mIoU {meanIoU.avg:.4f} | cumIoU {inter_area.sum/ max(1e-6, union_area.sum):.4f}")
    print(f"TEST Final: acc@0.5 {acc5.avg:.4f} | acc@0.6 {acc6.avg:.4f} | acc@0.7 {acc7.avg:.4f} | "
          f"acc@0.8 {acc8.avg:.4f} | acc@0.9 {acc9.avg:.4f} | mIoU {meanIoU.avg:.4f} | cumIoU {inter_area.sum/ max(1e-6, union_area.sum):.4f}")


def main():
    args = parse_args()
    os.makedirs('./logs', exist_ok=True)
    if args.savename == 'default':
        args.savename = f"MGVLF_batch{args.batch_size}_epoch{args.nb_epoch}_lr{args.lr}_seed{args.seed}"

    logging.basicConfig(level=logging.INFO, filename=f"./logs/{args.savename}.log", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    print('----------------------------------------------------------------------')
    print('Args:', args)
    print('----------------------------------------------------------------------')

    # auto device (no need to set CUDA_VISIBLE_DEVICES on Kaggle/Colab)
    set_seed(args.seed)

    # data
    train_loader, val_loader, test_loader = build_dataloaders(args)

    # model
    model = build_model(args)
    if args.pretrain:
        model = load_pretrain(model, args, logging)
    if args.resume:
        model = load_resume(model, args, logging)

    # optimizer + scheduler
    optimizer = optimizer_for(model, args)
    scheduler = None
    if not args.test:
        total_steps = len(train_loader) * args.nb_epoch
        warmup_steps = int(total_steps * args.sched_warmup_ratio)
        if args.sched_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # train / test
    best_accu = -float('inf')
    if args.test:
        test_epoch(test_loader, model, args)
        return

    for epoch in range(args.nb_epoch):
        # Nếu dùng scheduler theo batch, KHÔNG gọi adjust_learning_rate theo epoch để tránh xung đột
        if scheduler is None:
            adjust_learning_rate(args, optimizer, epoch)

        _ = train_epoch(train_loader, model, optimizer, epoch, args, scheduler)
        accs = validate_epoch(val_loader, model, args)
        acc_new = accs[0]  # acc@0.5
        is_best = acc_new >= best_accu
        best_accu = max(acc_new, best_accu)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()),
            'best_loss': accs[-1],
            'optimizer': optimizer.state_dict(),
        }, is_best, args, filename=args.savename)

    print(f'\nBest Accu@0.5: {best_accu:.4f}\n')
    logging.info(f'\nBest Accu@0.5: {best_accu:.4f}\n')


if __name__ == "__main__":
    main()