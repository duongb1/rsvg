# main.py
import argparse
import os
import sys
import time
import logging
import datetime
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn

from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torch.autograd import Variable

# ---- local imports (refactor) ----
from dataset import RSVGDataset
from models.model import MGVLF
from utils.loss import Reg_Loss, GIoU_Loss, EIoU_Loss_Compat
from utils.utils import AverageMeter, xyxy2xywh, bbox_iou, adjust_learning_rate
from utils.checkpoint import save_checkpoint, load_pretrain, load_resume


def parse_args():
    parser = argparse.ArgumentParser(description="MGVLF Remote Sensing Grounding (Kaggle/Colab-ready)")
    # data
    parser.add_argument('--images_path', type=str, default='/kaggle/input/dior-rsvg/DIOR_RSVG/JPEGImages',
                        help='path to images')
    parser.add_argument('--anno_path', type=str, default='/kaggle/input/dior-rsvg/DIOR_RSVG/Annotations',
                        help='path to VOC-style xml annotations')
    parser.add_argument('--split_root', type=str, default='/kaggle/input/dior-rsvg/DIOR_RSVG',
                        help='folder contains train.txt/val.txt/test.txt')
    parser.add_argument('--size', default=640, type=int, help='image size (square)')
    parser.add_argument('--time', default=40, type=int, help='max text length per sample')
    # train
    parser.add_argument('--nb_epoch', default=150, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_dec', default=0.1, type=float)
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--print_freq', default=50, type=int)
    parser.add_argument('--savename', default='default', type=str)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH')
    parser.add_argument('--tunebert', dest='tunebert', default=True, action='store_true')
    parser.add_argument('--test', dest='test', default=False, action='store_true')
    # model
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str,
                        help='English BERT checkpoint')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--masks', action='store_true', help='return intermediate layers (always on internally)')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=441, type=int)  # unused, kept for compatibility
    parser.add_argument('--pre_norm', action='store_true')
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
    input_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RSVGDataset(
        images_path=args.images_path,
        anno_path=args.anno_path,
        splits_dir=args.split_root,
        split='train',
        imsize=args.size,
        transform=input_transform,
        max_query_len=args.time,
        bert_model=args.bert_model
    )
    val_dataset = RSVGDataset(
        images_path=args.images_path,
        anno_path=args.anno_path,
        splits_dir=args.split_root,
        split='val',
        imsize=args.size,
        transform=input_transform,
        max_query_len=args.time,
        bert_model=args.bert_model
    )
    test_dataset = RSVGDataset(
        images_path=args.images_path,
        anno_path=args.anno_path,
        splits_dir=args.split_root,
        split='test',
        imsize=args.size,
        transform=input_transform,
        max_query_len=args.time,
        testmode=True,
        bert_model=args.bert_model
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, drop_last=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=True, drop_last=True, num_workers=0)

    print('trainset:', len(train_dataset), 'validationset:', len(val_dataset), 'testset:', len(test_dataset))
    return train_loader, val_loader, test_loader


def build_model(args):
    # MGVLF mới: đặt tên tham số theo model.py
    model = MGVLF(
        text_model_name=args.bert_model,
        init_backbone_from_detr=True,
        freeze_text_encoder=not args.tunebert,  # nếu không tune BERT thì freeze
        freeze_backbone=False,                  # tuỳ bạn
        heads=args.nheads,
        layers=3,                               # theo gợi ý paper (có thể để args.enc_layers nếu muốn)
        lvfe_iters=3,
        dropout=args.dropout
    )
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    device = torch.device('cuda' if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    model = model.to(device)
    return model



def optimizer_for(model: nn.Module, args):
    # Lấy module thật nếu DataParallel
    mm = model.module if isinstance(model, nn.DataParallel) else model
    # nhóm LR: backbone/text/fusion-head
    param_groups = mm.get_param_groups(
        lr_backbone=args.lr / 10.0,
        lr_text=args.lr / 10.0
        if args.tunebert
        else 0.0,  # nếu không tune BERT, lr_text dùng nhưng tham số đã freeze
        lr_head=args.lr,
        weight_decay=1e-4,
    )
    opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=1e-4)

    # log số lượng tham số mỗi nhóm
    n0 = sum(p.numel() for p in param_groups[0]["params"])
    n1 = sum(p.numel() for p in param_groups[1]["params"])
    n2 = sum(p.numel() for p in param_groups[2]["params"])
    print(f"groups: backbone={n0} | text={n1} | fusion+head={n2}")
    return opt




def train_epoch(train_loader, model, optimizer, epoch, args):
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

    for batch_idx, (imgs, masks, word_id, word_mask, gt_bbox) in enumerate(train_loader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        masks = masks[:, :, :, 0] == 255  # boolean
        word_id = torch.as_tensor(word_id, device=device)
        word_mask = torch.as_tensor(word_mask, device=device)
        gt_bbox = torch.as_tensor(gt_bbox, device=device)
        gt_bbox = torch.clamp(gt_bbox, min=0, max=args.size - 1)

        out = model(imgs, input_ids=word_id, attention_mask=word_mask)
        pred_bbox = out["pred_cxcywh_norm"]  # (B,4) đã chuẩn hoá [0,1]


        # losses
        loss = 0.0
        giou_loss = GIoU_Loss(pred_bbox * (args.size - 1), gt_bbox, args.size - 1)
        gt_bbox_xywh = xyxy2xywh(gt_bbox)
        l1_loss = Reg_Loss(pred_bbox, gt_bbox_xywh / (args.size - 1))
        loss_eiou = EIoU_Loss_Compat(pred_bbox * (args.size - 1), gt_bbox, args.size - 1)

        loss = giou_loss + l1_loss + loss_eiou

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
            print(f"Epoch[{epoch}] [{batch_idx}/{len(train_loader)}] "
                  f"acc@0.5 {acc5.avg:.4f} | acc@0.6 {acc6.avg:.4f} | acc@0.7 {acc7.avg:.4f} | "
                  f"acc@0.8 {acc8.avg:.4f} | acc@0.9 {acc9.avg:.4f} | "
                  f"mIoU {meanIoU.avg:.4f} | cumIoU {inter_area.sum/ max(1e-6, union_area.sum):.4f} | "
                  f"Loss {losses.avg:.4f} | L1 {l1_losses.avg:.4f} | GIoU {giou_losses.avg:.4f} | "
                  f"lr_v {optimizer.param_groups[0]['lr']:.6e}")
    return acc5.avg, acc6.avg, acc7.avg, acc8.avg, acc9.avg, meanIoU.avg, inter_area.sum / max(1e-6, union_area.sum), losses.avg


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
        word_id = torch.as_tensor(word_id, device=device)
        word_mask = torch.as_tensor(word_mask, device=device)
        bbox = torch.as_tensor(bbox, device=device)
        bbox = torch.clamp(bbox, min=0, max=args.size - 1)

        out = model(imgs, input_ids=word_id, attention_mask=word_mask)
        pred_bbox = out["pred_cxcywh_norm"]

        giou_loss = GIoU_Loss(pred_bbox * (args.size - 1), bbox, args.size - 1)
        l1_loss = Reg_Loss(pred_bbox, xyxy2xywh(bbox) / (args.size - 1))
        loss_eiou = EIoU_Loss_Compat(pred_bbox * (args.size - 1), bbox, args.size - 1)
        loss = giou_loss + l1_loss + loss_eiou

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
        word_id = torch.as_tensor(word_id, device=device)
        word_mask = torch.as_tensor(word_mask, device=device)
        bbox = torch.as_tensor(bbox, device=device)
        bbox = torch.clamp(bbox, min=0, max=args.size - 1)

        out = model(imgs, input_ids=word_id, attention_mask=word_mask)
        pred_bbox = out["pred_cxcywh_norm"]

        pred_xyxy = torch.cat([pred_bbox[:, :2] - (pred_bbox[:, 2:] / 2),
                               pred_bbox[:, :2] + (pred_bbox[:, 2:] / 2)], dim=1) * (args.size - 1)

        # de-letterbox back to original scale if cần (ở đây giữ nguyên metric ở canvas size)
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

    # optimizer
    optimizer = optimizer_for(model, args)

    # train / test
    best_accu = -float('inf')
    if args.test:
        test_epoch(test_loader, model, args)
        return

    for epoch in range(args.nb_epoch):
        adjust_learning_rate(args, optimizer, epoch)
        _ = train_epoch(train_loader, model, optimizer, epoch, args)
        accs = validate_epoch(val_loader, model, args)
        acc_new = accs[0]  # acc@0.5
        is_best = acc_new >= best_accu
        best_accu = max(acc_new, best_accu)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': accs[-1],
            'optimizer': optimizer.state_dict(),
        }, is_best, args, filename=args.savename)

    print(f'\nBest Accu@0.5: {best_accu:.4f}\n')
    logging.info(f'\nBest Accu@0.5: {best_accu:.4f}\n')


if __name__ == "__main__":
    main()
