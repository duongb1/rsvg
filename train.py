#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train + Validate for MGVLF
"""
import argparse
import os, sys, time, datetime, logging, random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib as mpl
mpl.use('Agg')

from torchvision.transforms import Compose, ToTensor, Normalize

from data_loader import RSVGDataset
from models.model import MGVLF
from models.loss import Reg_Loss, GIoU_Loss
from utils.utils import AverageMeter, xyxy2xywh, bbox_iou, adjust_learning_rate
from utils.checkpoint import save_checkpoint, load_pretrain, load_resume


def parse_args():
    parser = argparse.ArgumentParser(description='MGVLF Train/Val')
    parser.add_argument('--size', default=640, type=int, help='image size')
    parser.add_argument('--images_path', type=str, default='./DIOR_RSVG/JPEGImages')
    parser.add_argument('--anno_path', type=str, default='./DIOR_RSVG/Annotations')
    parser.add_argument('--time', default=40, type=int, help='max language length')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--nb_epoch', default=150, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_dec', default=0.1, type=float)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH')
    parser.add_argument('--print_freq', '-p', default=50, type=int)
    parser.add_argument('--savename', default='default', type=str)
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str)
    parser.add_argument('--tunebert', dest='tunebert', default=True, action='store_true')

    # * DETR / backbone / transformer
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=400 + 40 + 1, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    return parser.parse_args()


def build_loaders(args):
    input_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RSVGDataset(images_path=args.images_path, anno_path=args.anno_path,
                                split='train', imsize=args.size, transform=input_transform, max_query_len=args.time)
    val_dataset = RSVGDataset(images_path=args.images_path, anno_path=args.anno_path,
                              split='val', imsize=args.size, transform=input_transform, max_query_len=args.time)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, drop_last=True, num_workers=args.workers)
    print('trainset:', len(train_dataset), 'validationset:', len(val_dataset))
    return train_loader, val_loader


def build_model(args):
    model = MGVLF(bert_model=args.bert_model, tunebert=args.tunebert, args=args)
    model = torch.nn.DataParallel(model).cuda()

    if args.pretrain:
        model = load_pretrain(model, args, logging)
    if args.resume:
        model = load_resume(model, args, logging)

    num_params = sum(p.nelement() for p in model.parameters())
    print('Num of parameters:', num_params)
    logging.info('Num of parameters:%d' % int(num_params))

    # ==== group params (SO SÁNH THEO IDENTITY) ====
    if args.tunebert:
        # Lấy list ngay từ đầu để có object identity ổn định
        visu_param = list(model.module.visumodel.parameters())
        text_param = list(model.module.textmodel.parameters())

        visu_ids = {id(p) for p in visu_param}
        text_ids = {id(p) for p in text_param}

        # phần còn lại = tất cả trừ 2 nhóm trên (theo id)
        rest_param = [p for p in model.parameters() if id(p) not in visu_ids and id(p) not in text_ids]

        # (tuỳ chọn) chỉ lấy params trainable
        visu_param = [p for p in visu_param if p.requires_grad]
        text_param = [p for p in text_param if p.requires_grad]
        rest_param = [p for p in rest_param if p.requires_grad]

        optimizer = torch.optim.AdamW(
            [
                {'params': rest_param, 'lr': args.lr},
                {'params': visu_param, 'lr': args.lr / 10.0},
                {'params': text_param, 'lr': args.lr / 10.0},
            ],
            lr=args.lr, weight_decay=1e-4
        )

        sum_visu = sum(p.nelement() for p in visu_param)
        sum_text = sum(p.nelement() for p in text_param)
        sum_rest = sum(p.nelement() for p in rest_param)
        print('visu, text, fusion module parameters:', sum_visu, sum_text, sum_rest)

    else:
        # Không tune BERT: có thể freeze text model (đã set trong MGVLF) hoặc để nguyên
        visu_param = list(model.module.visumodel.parameters())
        visu_ids = {id(p) for p in visu_param}

        rest_param = [p for p in model.parameters() if id(p) not in visu_ids]

        visu_param = [p for p in visu_param if p.requires_grad]
        rest_param = [p for p in rest_param if p.requires_grad]

        optimizer = torch.optim.AdamW(
            [
                {'params': rest_param, 'lr': args.lr},
                {'params': visu_param, 'lr': args.lr},
            ],
            lr=args.lr, weight_decay=1e-4
        )

        sum_visu = sum(p.nelement() for p in visu_param)
        sum_text_total = sum(p.nelement() for p in model.module.textmodel.parameters())
        sum_rest = sum(p.nelement() for p in rest_param)
        # fusion ~ rest; nếu text bị freeze, nó nằm trong rest về mặt tổng số param ALL,
        # nhưng không ảnh hưởng training vì requires_grad=False sẽ không vào optimizer.
        print('visu, text(total), fusion(rest) parameters:', sum_visu, sum_text_total, sum_rest)

    # ==== sanity checks để tránh sót/đúp ====
    all_ids = {id(p) for p in model.parameters() if p.requires_grad}
    grouped_ids = set()
    for g in optimizer.param_groups:
        for p in g['params']:
            grouped_ids.add(id(p))

    assert all_ids == grouped_ids, \
        f"Param grouping mismatch: all_grad={len(all_ids)} grouped={len(grouped_ids)}"

    return model, optimizer



def train_epoch(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    l1_losses = AverageMeter()
    GIoU_losses = AverageMeter()
    acc5 = AverageMeter(); acc6 = AverageMeter(); acc7 = AverageMeter(); acc8 = AverageMeter(); acc9 = AverageMeter()
    meanIoU = AverageMeter(); inter_area = AverageMeter(); union_area = AverageMeter()

    model.train()
    end = time.time()

    for batch_idx, (imgs, masks, word_id, word_mask, gt_bbox) in enumerate(train_loader):
        imgs = imgs.cuda()
        masks = masks.cuda()
        masks = (masks[:, :, :, 0] == 255).bool()       # True = padding
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        gt_bbox = gt_bbox.cuda()
        image = Variable(imgs)
        masks = Variable(masks)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        gt_bbox = Variable(gt_bbox)
        gt_bbox = torch.clamp(gt_bbox, min=0, max=args.size - 1)

        pred_bbox = model(image, masks, word_id, word_mask)

        # loss
        loss = 0.
        giou = GIoU_Loss(pred_bbox * (args.size - 1), gt_bbox, args.size - 1)
        loss += giou
        gt_bbox_ = xyxy2xywh(gt_bbox)
        l1 = Reg_Loss(pred_bbox, gt_bbox_ / (args.size - 1))
        loss += l1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), imgs.size(0))
        l1_losses.update(l1.item(), imgs.size(0))
        GIoU_losses.update(giou.item(), imgs.size(0))

        # metrics
        pred_xyxy = torch.cat([pred_bbox[:, :2] - (pred_bbox[:, 2:] / 2),
                               pred_bbox[:, :2] + (pred_bbox[:, 2:] / 2)], dim=1) * (args.size - 1)
        iou, interArea, unionArea = bbox_iou(pred_xyxy.data.cpu(), gt_bbox.data.cpu(), x1y1x2y2=True)
        cumInterArea = np.sum(np.array(interArea.data.cpu().numpy()))
        cumUnionArea = np.sum(np.array(unionArea.data.cpu().numpy()))
        accu5 = np.mean((iou.data.cpu().numpy() > 0.5).astype(float))
        accu6 = np.mean((iou.data.cpu().numpy() > 0.6).astype(float))
        accu7 = np.mean((iou.data.cpu().numpy() > 0.7).astype(float))
        accu8 = np.mean((iou.data.cpu().numpy() > 0.8).astype(float))
        accu9 = np.mean((iou.data.cpu().numpy() > 0.9).astype(float))

        meanIoU.update(torch.mean(iou).item(), imgs.size(0))
        inter_area.update(cumInterArea); union_area.update(cumUnionArea)
        acc5.update(accu5, imgs.size(0)); acc6.update(accu6, imgs.size(0))
        acc7.update(accu7, imgs.size(0)); acc8.update(accu8, imgs.size(0)); acc9.update(accu9, imgs.size(0))

        # time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            lang_lr = optimizer.param_groups[2]['lr'] if len(optimizer.param_groups) > 2 else optimizer.param_groups[-1]['lr']
            print_str = (f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}]\t'
                         f'acc@0.5: {acc5.avg:.4f}\tacc@0.6: {acc6.avg:.4f}\tacc@0.7: {acc7.avg:.4f}\t'
                         f'acc@0.8: {acc8.avg:.4f}\tacc@0.9: {acc9.avg:.4f}\t'
                         f'meanIoU: {meanIoU.avg:.4f}\t'
                         f'cumuIoU: {inter_area.sum/union_area.sum:.4f}\t'
                         f'Loss: {losses.avg:.4f}\tL1: {l1_losses.avg:.4f}\tGIoU: {GIoU_losses.avg:.4f}\t'
                         f'vis_lr {optimizer.param_groups[0]["lr"]:.8f}\tlang_lr {lang_lr:.8f}')
            print(print_str); logging.info(print_str)

    return acc5.avg, acc6.avg, acc7.avg, acc8.avg, acc9.avg, meanIoU.avg, inter_area.sum/union_area.sum, losses.avg


@torch.no_grad()
def validate_epoch(val_loader, model, args):
    batch_time = AverageMeter()
    losses = AverageMeter(); l1_losses = AverageMeter(); GIoU_losses = AverageMeter()
    acc5 = AverageMeter(); acc6 = AverageMeter(); acc7 = AverageMeter(); acc8 = AverageMeter(); acc9 = AverageMeter()
    meanIoU = AverageMeter(); inter_area = AverageMeter(); union_area = AverageMeter()

    model.eval()
    end = time.time()
    print(datetime.datetime.now())

    for batch_idx, (imgs, masks, word_id, word_mask, bbox) in enumerate(val_loader):
        imgs = imgs.cuda()
        masks = masks.cuda()
        masks = (masks[:, :, :, 0] == 255).bool()      # True = padding
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()

        image = Variable(imgs); masks = Variable(masks)
        word_id = Variable(word_id); word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox, min=0, max=args.size - 1)

        pred_bbox = model(image, masks, word_id, word_mask)
        gt_bbox = bbox

        # loss
        loss = 0.
        giou = GIoU_Loss(pred_bbox * (args.size - 1), gt_bbox, args.size - 1); loss += giou
        gt_bbox_ = xyxy2xywh(gt_bbox)
        l1 = Reg_Loss(pred_bbox, gt_bbox_ / (args.size - 1)); loss += l1

        losses.update(loss.item(), imgs.size(0)); l1_losses.update(l1.item(), imgs.size(0)); GIoU_losses.update(giou.item(), imgs.size(0))

        # metrics
        pred_xyxy = torch.cat([pred_bbox[:, :2] - (pred_bbox[:, 2:] / 2),
                               pred_bbox[:, :2] + (pred_bbox[:, 2:] / 2)], dim=1) * (args.size - 1)
        iou, interArea, unionArea = bbox_iou(pred_xyxy.data.cpu(), gt_bbox.data.cpu(), x1y1x2y2=True)
        cumInterArea = np.sum(np.array(interArea.data.cpu().numpy()))
        cumUnionArea = np.sum(np.array(unionArea.data.cpu().numpy()))
        accu5 = np.mean((iou.data.cpu().numpy() > 0.5).astype(float))
        accu6 = np.mean((iou.data.cpu().numpy() > 0.6).astype(float))
        accu7 = np.mean((iou.data.cpu().numpy() > 0.7).astype(float))
        accu8 = np.mean((iou.data.cpu().numpy() > 0.8).astype(float))
        accu9 = np.mean((iou.data.cpu().numpy() > 0.9).astype(float))

        meanIoU.update(torch.mean(iou).item(), imgs.size(0))
        inter_area.update(cumInterArea); union_area.update(cumUnionArea)
        acc5.update(accu5, imgs.size(0)); acc6.update(accu6, imgs.size(0))
        acc7.update(accu7, imgs.size(0)); acc8.update(accu8, imgs.size(0)); acc9.update(accu9, imgs.size(0))

        # time
        batch_time.update(time.time() - end); end = time.time()

        if batch_idx % args.print_freq == 0:
            print_str = (f'[{batch_idx}/{len(val_loader)}]\tTime {batch_time.avg:.3f}\t'
                         f'acc@0.5: {acc5.avg:.4f}\tacc@0.6: {acc6.avg:.4f}\tacc@0.7: {acc7.avg:.4f}\t'
                         f'acc@0.8: {acc8.avg:.4f}\tacc@0.9: {acc9.avg:.4f}\t'
                         f'meanIoU: {meanIoU.avg:.4f}\t'
                         f'cumuIoU: {inter_area.sum/union_area.sum:.4f}\t'
                         f'Loss: {losses.avg:.4f}')
            print(print_str); logging.info(print_str)

    final_str = (f'acc@0.5: {acc5.avg:.4f}\tacc@0.6: {acc6.avg:.4f}\tacc@0.7: {acc7.avg:.4f}\t'
                 f'acc@0.8: {acc8.avg:.4f}\tacc@0.9: {acc9.avg:.4f}\t'
                 f'meanIoU: {meanIoU.avg:.4f}\t'
                 f'cumuIoU: {inter_area.sum/union_area.sum:.4f}')
    print(final_str); logging.info(final_str)
    return acc5.avg, acc6.avg, acc7.avg, acc8.avg, acc9.avg, meanIoU.avg, inter_area.sum/union_area.sum, losses.avg


def main():
    args = parse_args()

    print('----------------------------------------------------------------------')
    print('Args:', args)
    print('----------------------------------------------------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # seeds
    cudnn.benchmark = False; cudnn.deterministic = True
    random.seed(args.seed); np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 2); torch.cuda.manual_seed_all(args.seed + 3)

    # logs
    if args.savename == 'default':
        args.savename = f'MGVLF_batch{args.batch_size}_epoch{args.nb_epoch}_lr{args.lr}_seed{args.seed}'
    os.makedirs('./logs', exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=f"./logs/{args.savename}", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(str(sys.argv)); logging.info(str(args))

    # data & model
    train_loader, val_loader = build_loaders(args)
    model, optimizer = build_model(args)

    # train loop
    best_accu = -float('Inf')
    for epoch in range(args.nb_epoch):
        adjust_learning_rate(args, optimizer, epoch)
        _ = train_epoch(train_loader, model, optimizer, epoch, args)
        v_metrics = validate_epoch(val_loader, model, args)

        acc_new = v_metrics[0]
        is_best = acc_new >= best_accu
        best_accu = max(acc_new, best_accu)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': acc_new,
            'optimizer': optimizer.state_dict(),
        }, is_best, args, filename=args.savename)

    print('\nBest Accu: %f\n' % best_accu)
    logging.info('\nBest Accu: %f\n' % best_accu)


if __name__ == "__main__":
    main()
