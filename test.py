#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test/Eval for MGVLF
"""
import argparse
import os, sys, time, datetime, logging
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor, Normalize

from data_loader import RSVGDataset
from models.model import MGVLF
from utils.utils import AverageMeter, bbox_iou
from utils.checkpoint import load_resume


def parse_args():
    parser = argparse.ArgumentParser(description='MGVLF Test')
    parser.add_argument('--size', default=640, type=int, help='image size')
    parser.add_argument('--images_path', type=str, default='./DIOR_RSVG/JPEGImages')
    parser.add_argument('--anno_path', type=str, default='./DIOR_RSVG/Annotations')
    parser.add_argument('--time', default=40, type=int, help='max language length')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='checkpoint to evaluate')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str)
    parser.add_argument('--savename', default='test_log', type=str)

    # * DETR / backbone / transformer (giống train)
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


def build_loader(args):
    input_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    test_dataset = RSVGDataset(images_path=args.images_path, anno_path=args.anno_path,
                               split='test', testmode=True, imsize=args.size,
                               transform=input_transform, max_query_len=args.time)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, drop_last=True, num_workers=args.workers)
    print('testset:', len(test_dataset))
    return test_loader


@torch.no_grad()
def test_epoch(test_loader, model, args):
    batch_time = AverageMeter()
    acc5 = AverageMeter(); acc6 = AverageMeter(); acc7 = AverageMeter(); acc8 = AverageMeter(); acc9 = AverageMeter()
    meanIoU = AverageMeter(); inter_area = AverageMeter(); union_area = AverageMeter()

    model.eval()
    end = time.time()

    for batch_idx, (imgs, masks, word_id, word_mask, bbox, ratio, dw, dh, im_id, phrase) in enumerate(test_loader):
        imgs = imgs.cuda()
        masks = masks.cuda()
        masks = (masks[:, :, :, 0] == 255).bool()   # True = padding
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()

        image = Variable(imgs); masks = Variable(masks)
        word_id = Variable(word_id); word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox, min=0, max=args.size - 1)

        pred_bbox = model(image, masks, word_id, word_mask)
        pred_bbox = torch.cat([pred_bbox[:, :2] - (pred_bbox[:, 2:] / 2),
                               pred_bbox[:, :2] + (pred_bbox[:, 2:] / 2)], dim=1)
        pred_bbox = pred_bbox * (args.size - 1)

        # revert to original image scale
        pred_bbox = pred_bbox.data.cpu()
        target_bbox = bbox.data.cpu()
        pred_bbox[:, 0], pred_bbox[:, 2] = (pred_bbox[:, 0] - dw) / ratio, (pred_bbox[:, 2] - dw) / ratio
        pred_bbox[:, 1], pred_bbox[:, 3] = (pred_bbox[:, 1] - dh) / ratio, (pred_bbox[:, 3] - dh) / ratio
        target_bbox[:, 0], target_bbox[:, 2] = (target_bbox[:, 0] - dw) / ratio, (target_bbox[:, 2] - dw) / ratio
        target_bbox[:, 1], target_bbox[:, 3] = (target_bbox[:, 1] - dh) / ratio, (target_bbox[:, 3] - dh) / ratio

        # clamp by original image size
        top, bottom = round(float(dh[0]) - 0.1), args.size - round(float(dh[0]) + 0.1)
        left, right = round(float(dw[0]) - 0.1), args.size - round(float(dw[0]) + 0.1)
        img_np = imgs[0, :, top:bottom, left:right].data.cpu().numpy().transpose(1, 2, 0)

        ratio_scalar = float(ratio)
        new_shape = (round(img_np.shape[1] / ratio_scalar), round(img_np.shape[0] / ratio_scalar))
        img_np = cv2.resize(img_np, new_shape, interpolation=cv2.INTER_CUBIC)
        img_np = Variable(torch.from_numpy(img_np.transpose(2, 0, 1)).cuda().unsqueeze(0))

        pred_bbox[:, :2], pred_bbox[:, 2], pred_bbox[:, 3] = \
            torch.clamp(pred_bbox[:, :2], min=0), \
            torch.clamp(pred_bbox[:, 2], max=img_np.shape[3]), \
            torch.clamp(pred_bbox[:, 3], max=img_np.shape[2])
        target_bbox[:, :2], target_bbox[:, 2], target_bbox[:, 3] = \
            torch.clamp(target_bbox[:, :2], min=0), \
            torch.clamp(target_bbox[:, 2], max=img_np.shape[3]), \
            torch.clamp(target_bbox[:, 3], max=img_np.shape[2])

        # IoU & metrics (batch_size=1 ở test mặc định)
        iou, interArea, unionArea = bbox_iou(pred_bbox, target_bbox, x1y1x2y2=True)
        cumInterArea = np.sum(np.array(interArea.data.cpu().numpy()))
        cumUnionArea = np.sum(np.array(unionArea.data.cpu().numpy()))
        accu5 = float((iou.data.cpu().numpy() > 0.5).astype(float).mean())
        accu6 = float((iou.data.cpu().numpy() > 0.6).astype(float).mean())
        accu7 = float((iou.data.cpu().numpy() > 0.7).astype(float).mean())
        accu8 = float((iou.data.cpu().numpy() > 0.8).astype(float).mean())
        accu9 = float((iou.data.cpu().numpy() > 0.9).astype(float).mean())

        meanIoU.update(torch.mean(iou).item(), imgs.size(0))
        inter_area.update(cumInterArea); union_area.update(cumUnionArea)
        acc5.update(accu5, imgs.size(0)); acc6.update(accu6, imgs.size(0))
        acc7.update(accu7, imgs.size(0)); acc8.update(accu8, imgs.size(0)); acc9.update(accu9, imgs.size(0))

        batch_time.update(time.time() - end); end = time.time()

        if batch_idx % 50 == 0:
            print_str = (f'[{batch_idx}/{len(test_loader)}]\tTime {batch_time.avg:.3f}\t'
                         f'acc@0.5: {acc5.avg:.4f}\tacc@0.6: {acc6.avg:.4f}\tacc@0.7: {acc7.avg:.4f}\t'
                         f'acc@0.8: {acc8.avg:.4f}\tacc@0.9: {acc9.avg:.4f}\t'
                         f'meanIoU: {meanIoU.avg:.4f}\t'
                         f'cumuIoU: {inter_area.sum/union_area.sum:.4f}')
            print(print_str); logging.info(print_str)

    final_str = (f'acc@0.5: {acc5.avg:.4f}\tacc@0.6: {acc6.avg:.4f}\tacc@0.7: {acc7.avg:.4f}\t'
                 f'acc@0.8: {acc8.avg:.4f}\tacc@0.9: {acc9.avg:.4f}\t'
                 f'meanIoU: {meanIoU.avg:.4f}\t'
                 f'cumuIoU: {inter_area.sum/union_area.sum:.4f}')
    print(final_str); logging.info(final_str)


def main():
    args = parse_args()

    print('----------------------------------------------------------------------')
    print('Args:', args)
    print('----------------------------------------------------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.makedirs('./logs', exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=f"./logs/{args.savename}", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(str(sys.argv)); logging.info(str(args))

    test_loader = build_loader(args)

    # Model
    model = MGVLF(bert_model=args.bert_model, tunebert=True, args=args)
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        model = load_resume(model, args, logging)
    else:
        raise ValueError("--resume (checkpoint) is required for testing.")

    # Run test
    test_epoch(test_loader, model, args)


if __name__ == "__main__":
    main()
