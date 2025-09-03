# main.py
import argparse
import os
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

# ---- local imports ----
from dataset import RSVGDataset
from models.model import MGVLF
from utils.loss import Reg_Loss, GIoU_Loss, EIoU_Loss_Compat
from utils.utils import AverageMeter, xyxy2xywh, bbox_iou, adjust_learning_rate
from utils.checkpoint import save_checkpoint, load_pretrain, load_resume


def parse_args():
    parser = argparse.ArgumentParser(description="MGVLF Remote Sensing Grounding (VGRSS re-implementation)")
    # data
    parser.add_argument('--images_path', type=str, default='/kaggle/input/dior-rsvg/DIOR_RSVG/JPEGImages')
    parser.add_argument('--anno_path', type=str, default='/kaggle/input/dior-rsvg/DIOR_RSVG/Annotations')
    parser.add_argument('--split_root', type=str, default='/kaggle/input/dior-rsvg/DIOR_RSVG')
    parser.add_argument('--size', default=640, type=int)
    parser.add_argument('--time', default=40, type=int, help='max text length per sample (paper=20; dùng --time 40 cho DIOR-RSVG)')
    # train
    parser.add_argument('--nb_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)  # giữ để log; lr theo nhóm set trong optimizer_for
    parser.add_argument('--lr_dec', default=0.1, type=float, help='decay factor at epoch 60 (set in utils.adjust_learning_rate)')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--savename', default='default', type=str)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH')
    parser.add_argument('--tunebert', dest='tunebert', default=True, action='store_true')
    parser.add_argument('--test', dest='test', default=False, action='store_true')
    # model
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--enc_layers', default=4, type=int, help="VLF layers (paper=4)")
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
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
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RSVGDataset(args.images_path, args.anno_path, args.split_root,
                                split='train', imsize=args.size, transform=transform,
                                max_query_len=args.time, bert_model=args.bert_model)
    val_dataset = RSVGDataset(args.images_path, args.anno_path, args.split_root,
                              split='val', imsize=args.size, transform=transform,
                              max_query_len=args.time, bert_model=args.bert_model)
    test_dataset = RSVGDataset(args.images_path, args.anno_path, args.split_root,
                               split='test', imsize=args.size, transform=transform,
                               max_query_len=args.time, testmode=True, bert_model=args.bert_model)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, drop_last=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=True, drop_last=True, num_workers=0)

    print(f"trainset={len(train_dataset)}, valset={len(val_dataset)}, testset={len(test_dataset)}")
    return train_loader, val_loader, test_loader


def build_model(args):
    model = MGVLF(
        text_model_name=args.bert_model,
        freeze_text_encoder=not args.tunebert,
        freeze_backbone=False,
        v_dim=args.hidden_dim,
        heads=args.nheads,
        lvfe_layers=3,          # theo paper
        vlf_layers=4,           # theo paper
        vlf_query_repeats=4,
        dropout=args.dropout,
    )
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.to(device)


def optimizer_for(model: nn.Module, args):
    """
    AdamW với 3 nhóm tham số:
      - backbone (ResNet50):      lr = 1e-5
      - text encoder (BERT):      lr = 1e-5  (nếu --tunebert; ngược lại = 0.0)
      - fusion + head (LVFE/VLF): lr = 1e-4
    weight_decay = 1e-4
    """
    mm = model.module if isinstance(model, nn.DataParallel) else model

    param_groups = mm.get_param_groups(
        lr_backbone=1e-5,
        lr_text=1e-5 if args.tunebert else 0.0,
        lr_head=1e-4,
        weight_decay=1e-4,
    )

    # Tạo optimizer. Không đặt lr toàn cục để tránh ghi đè lr theo nhóm.
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # Log thông tin nhóm tham số
    n_backbone = sum(p.numel() for p in param_groups[0]["params"])
    n_text     = sum(p.numel() for p in param_groups[1]["params"])
    n_head     = sum(p.numel() for p in param_groups[2]["params"])
    print(
        f"[ParamGroups] backbone={n_backbone:,} | text={n_text:,} | fusion+head={n_head:,}\n"
        f"[LRs] backbone={param_groups[0]['lr']:.2e} | text={param_groups[1]['lr']:.2e} | head={param_groups[2]['lr']:.2e} | "
        f"weight_decay={param_groups[0].get('weight_decay', 0.0):.1e}"
    )

    return optimizer


def train_epoch(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    l1_losses = AverageMeter()
    giou_losses = AverageMeter()
    eiou_losses = AverageMeter()
    acc5 = AverageMeter(); acc6 = AverageMeter(); acc7 = AverageMeter(); acc8 = AverageMeter(); acc9 = AverageMeter()
    meanIoU = AverageMeter()
    inter_area = AverageMeter(); union_area = AverageMeter()

    model.train()
    device = next(model.parameters()).device
    end = time.time()

    for batch_idx, (imgs, masks, word_id, word_mask, gt_bbox) in enumerate(train_loader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        masks = masks[:, :, :, 0] == 255  # boolean (True = pad)
        word_id = torch.as_tensor(word_id, device=device)
        word_mask = torch.as_tensor(word_mask, device=device)  # 1=valid, 0=pad
        gt_bbox = torch.as_tensor(gt_bbox, device=device)
        gt_bbox = torch.clamp(gt_bbox, min=0, max=args.size - 1)

        # forward
        pred_bbox_dict = model(imgs, input_ids=word_id, attention_mask=word_mask)
        pred_bbox = pred_bbox_dict["pred_cxcywh_norm"]  # (B,4) normalized [0,1]

        # losses
        giou_loss = GIoU_Loss(pred_bbox * (args.size - 1), gt_bbox, args.size - 1)
        gt_bbox_xywh = xyxy2xywh(gt_bbox)
        l1_loss = Reg_Loss(pred_bbox, gt_bbox_xywh / (args.size - 1))
        eiou_loss = EIoU_Loss_Compat(pred_bbox * (args.size - 1), gt_bbox, args.size - 1)
        loss = l1_loss + giou_loss + eiou_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # meters
        bsz = imgs.size(0)
        losses.update(loss.item(), bsz)
        l1_losses.update(l1_loss.item(), bsz)
        giou_losses.update(giou_loss.item(), bsz)
        eiou_losses.update(eiou_loss.item(), bsz)

        # metrics
        pred_xyxy = torch.cat([pred_bbox[:, :2] - (pred_bbox[:, 2:] / 2),
                               pred_bbox[:, :2] + (pred_bbox[:, 2:] / 2)], dim=1)
        pred_xyxy = pred_xyxy * (args.size - 1)
        iou, interA, unionA = bbox_iou(pred_xyxy.detach().cpu(), gt_bbox.detach().cpu(), x1y1x2y2=True)

        cumInter = float(interA.sum().item())
        cumUnion = float(unionA.sum().item())
        meanIoU.update(float(iou.mean().item()), bsz)
        inter_area.update(cumInter); union_area.update(cumUnion)
        iou_np = iou.numpy()
        acc5.update((iou_np > 0.5).sum() / bsz, bsz)
        acc6.update((iou_np > 0.6).sum() / bsz, bsz)
        acc7.update((iou_np > 0.7).sum() / bsz, bsz)
        acc8.update((iou_np > 0.8).sum() / bsz, bsz)
        acc9.update((iou_np > 0.9).sum() / bsz, bsz)

        batch_time.update(time.time() - end); end = time.time()

        if batch_idx % args.print_freq == 0:
            lrs = [pg['lr'] for pg in optimizer.param_groups]
            print(f"Epoch[{epoch}] [{batch_idx}/{len(train_loader)}] "
                  f"acc@0.5 {acc5.avg:.4f} | acc@0.6 {acc6.avg:.4f} | acc@0.7 {acc7.avg:.4f} | "
                  f"acc@0.8 {acc8.avg:.4f} | acc@0.9 {acc9.avg:.4f} | "
                  f"mIoU {meanIoU.avg:.4f} | cumIoU {inter_area.sum/ max(1e-6, union_area.sum):.4f} | "
                  f"Loss {losses.avg:.4f} | L1 {l1_losses.avg:.4f} | GIoU {giou_losses.avg:.4f} | EIoU {eiou_losses.avg:.4f} | "
                  f"LRs bb/txt/head: {lrs[0]:.2e}/{lrs[1]:.2e}/{lrs[2]:.2e}")
    return acc5.avg, acc6.avg, acc7.avg, acc8.avg, acc9.avg, meanIoU.avg, inter_area.sum / max(1e-6, union_area.sum), losses.avg


@torch.no_grad()
def validate_epoch(val_loader, model, args):
    batch_time = AverageMeter()
    losses = AverageMeter(); l1_losses = AverageMeter(); giou_losses = AverageMeter(); eiou_losses = AverageMeter()
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

        pred_bbox_dict = model(imgs, input_ids=word_id, attention_mask=word_mask)
        pred_bbox = pred_bbox_dict["pred_cxcywh_norm"]

        giou_loss = GIoU_Loss(pred_bbox * (args.size - 1), bbox, args.size - 1)
        l1_loss = Reg_Loss(pred_bbox, xyxy2xywh(bbox) / (args.size - 1))
        eiou_loss = EIoU_Loss_Compat(pred_bbox * (args.size - 1), bbox, args.size - 1)
        loss = l1_loss + giou_loss + eiou_loss

        bsz = imgs.size(0)
        losses.update(loss.item(), bsz)
        l1_losses.update(l1_loss.item(), bsz)
        giou_losses.update(giou_loss.item(), bsz)
        eiou_losses.update(eiou_loss.item(), bsz)

        pred_xyxy = torch.cat([pred_bbox[:, :2] - (pred_bbox[:, 2:] / 2),
                    pred_bbox[:, :2] + (pred_bbox[:, 2:] / 2)], dim=1) * (args.size - 1)
        iou, interA, unionA = bbox_iou(pred_xyxy.detach().cpu(), bbox.detach().cpu(), x1y1x2y2=True)

        cumInter = float(interA.sum().item())
        cumUnion = float(unionA.sum().item())
        meanIoU.update(float(iou.mean().item()), bsz)
        inter_area.update(cumInter); union_area.update(cumUnion)
        iou_np = iou.numpy()
        acc5.update((iou_np > 0.5).sum() / bsz, bsz)
        acc6.update((iou_np > 0.6).sum() / bsz, bsz)
        acc7.update((iou_np > 0.7).sum() / bsz, bsz)
        acc8.update((iou_np > 0.8).sum() / bsz, bsz)
        acc9.update((iou_np > 0.9).sum() / bsz, bsz)

        batch_time.update(time.time() - end); end = time.time()

        if batch_idx % args.print_freq == 0:
            print(f"[VAL {batch_idx}/{len(val_loader)}] "
                  f"acc@0.5 {acc5.avg:.4f} | acc@0.6 {acc6.avg:.4f} | acc@0.7 {acc7.avg:.4f} | "
                  f"acc@0.8 {acc8.avg:.4f} | acc@0.9 {acc9.avg:.4f} | "
                  f"mIoU {meanIoU.avg:.4f} | cumIoU {inter_area.sum/ max(1e-6, union_area.sum):.4f} | "
                  f"Loss {losses.avg:.4f} | L1 {l1_losses.avg:.4f} | GIoU {giou_losses.avg:.4f} | EIoU {eiou_losses.avg:.4f}")
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

        pred_bbox_dict = model(imgs, input_ids=word_id, attention_mask=word_mask)
        pred_bbox = pred_bbox_dict["pred_cxcywh_norm"]
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
        args.savename = f"MGVLF_b{args.batch_size}_e{args.nb_epoch}_lr{args.lr}_seed{args.seed}"

    logging.basicConfig(level=logging.INFO,
                        filename=f"./logs/{args.savename}.log",
                        filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    print("Args:", args)
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
    best_acc = -1
    if args.test:
        test_epoch(test_loader, model, args)
        return

    for epoch in range(args.nb_epoch):
        adjust_learning_rate(args, optimizer, epoch)  # giảm LR ở epoch=60 theo utils/utils.py
        _ = train_epoch(train_loader, model, optimizer, epoch, args)
        accs = validate_epoch(val_loader, model, args)
        acc_new = accs[0]  # acc@0.5
        is_best = acc_new >= best_acc
        best_acc = max(acc_new, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': accs[-1],
            'optimizer': optimizer.state_dict(),
        }, is_best, args, filename=args.savename)

    print(f"\nBest Acc@0.5: {best_acc:.4f}\n")
    logging.info(f"\nBest Acc@0.5: {best_acc:.4f}\n")


if __name__ == "__main__":
    main()