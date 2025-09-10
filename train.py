# main.py
"""
Train + Validate for MGVLF (all-in-one baseline+improve)
"""
import argparse
import os, sys, time, datetime, logging, random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib as mpl
mpl.use('Agg')

from torchvision.transforms import Compose, ToTensor, Normalize

from data_loader import RSVGDataset
from models.model import MGVLF
from utils.loss import Reg_Loss, GIoU_Loss, bbox_xywh_to_xyxy, iou_xyxy, quality_loss
from utils.utils import AverageMeter, xyxy2xywh, bbox_iou  # adjust_learning_rate không dùng nữa
from utils.checkpoint import save_checkpoint, load_pretrain, load_resume
from utils.schedule import WarmupCosine
from utils.ema import EMA


def parse_args():
    parser = argparse.ArgumentParser(description='MGVLF Train/Val')
    parser.add_argument('--size', default=640, type=int, help='image size')
    parser.add_argument('--images_path', type=str, default='/kaggle/input/dior-rsvg/DIOR_RSVG/JPEGImages')
    parser.add_argument('--anno_path', type=str, default='/kaggle/input/dior-rsvg/DIOR_RSVG/Annotations')
    parser.add_argument('--time', default=40, type=int, help='max language length')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--nb_epoch', default=150, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_dec', default=0.1, type=float)  # giữ để tương thích log, không dùng
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH')
    parser.add_argument('--print_freq', '-p', default=100, type=int)
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

    # ===== Improvements (all-in-one) =====
    parser.add_argument('--use_ema', action='store_true', help='Use EMA for eval')
    parser.add_argument('--num_retrieval', type=int, default=4, help='K retrieval tokens pooled at fusion output')
    parser.add_argument('--use_coord', action='store_true', help='Inject 8-ch coord prior into visual feature')
    # parser.add_argument('--prune_ratio', type=float, default=1.0)  # để sau

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
    model = MGVLF(bert_model=args.bert_model, tunebert=args.tunebert, args=args).cuda()
    
    # nếu bạn thật sự có >1 GPU:
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model).cuda()

    if args.pretrain:
        model = load_pretrain(model, args, logging)
    if args.resume:
        model = load_resume(model, args, logging)

    num_params = sum(p.nelement() for p in model.parameters())
    print('Num of parameters:', num_params)
    logging.info('Num of parameters:%d' % int(num_params))

    # ==== group params (SO SÁNH THEO IDENTITY) ====
    if args.tunebert:
        visu_param = list(model.visumodel.parameters())
        text_param = list(model.textmodel.parameters())
        visu_ids = {id(p) for p in visu_param}
        text_ids = {id(p) for p in text_param}
        rest_param = [p for p in model.parameters() if id(p) not in visu_ids and id(p) not in text_ids]

        visu_param = [p for p in visu_param if p.requires_grad]
        text_param = [p for p in text_param if p.requires_grad]
        rest_param = [p for p in rest_param if p.requires_grad]

        optimizer = torch.optim.AdamW(
            [
                {'params': rest_param, 'lr': args.lr},
                {'params': visu_param, 'lr': args.lr / 10.0},
                {'params': text_param, 'lr': args.lr / 10.0},
            ],
            lr=args.lr, weight_decay=3e-4
        )

        sum_visu = sum(p.nelement() for p in visu_param)
        sum_text = sum(p.nelement() for p in text_param)
        sum_rest = sum(p.nelement() for p in rest_param)
        print('visu, text, fusion module parameters:', sum_visu, sum_text, sum_rest)
    else:
        visu_param = list(model.visumodel.parameters())
        visu_ids = {id(p) for p in visu_param}
        rest_param = [p for p in model.parameters() if id(p) not in visu_ids]

        visu_param = [p for p in visu_param if p.requires_grad]
        rest_param = [p for p in rest_param if p.requires_grad]

        optimizer = torch.optim.AdamW(
            [
                {'params': rest_param, 'lr': args.lr},
                {'params': visu_param, 'lr': args.lr},
            ],
            lr=args.lr, weight_decay=3e-4
        )

        sum_visu = sum(p.nelement() for p in visu_param)
        sum_text_total = sum(p.nelement() for p in model.textmodel.parameters())
        sum_rest = sum(p.nelement() for p in rest_param)
        print('visu, text(total), fusion(rest) parameters:', sum_visu, sum_text_total, sum_rest)

    # ==== sanity checks để tránh sót/đúp ====
    all_ids = {id(p) for p in model.parameters() if p.requires_grad}
    grouped_ids = set()
    for g in optimizer.param_groups:
        for p in g['params']:
            grouped_ids.add(id(p))
    assert all_ids == grouped_ids, f"Param grouping mismatch: all_grad={len(all_ids)} grouped={len(grouped_ids)}"

    return model, optimizer


def train_epoch(train_loader, model, optimizer, epoch, args, sched=None, ema=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    l1_losses = AverageMeter()
    GIoU_losses = AverageMeter()
    acc5 = AverageMeter(); acc6 = AverageMeter(); acc7 = AverageMeter(); acc8 = AverageMeter(); acc9 = AverageMeter()
    meanIoU = AverageMeter(); inter_area = AverageMeter(); union_area = AverageMeter()

    model.train()
    end = time.time()

    for batch_idx, (imgs, masks, word_id, word_mask, gt_bbox) in enumerate(train_loader):
        imgs = imgs.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)
        # True = padding
        masks = (masks[:, :, :, 0] == 255).bool()
        word_id = word_id.cuda(non_blocking=True)
        word_mask = word_mask.cuda(non_blocking=True)
        gt_bbox = gt_bbox.cuda(non_blocking=True)

        image = Variable(imgs)
        masks = Variable(masks)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        gt_bbox = Variable(gt_bbox)
        gt_bbox = torch.clamp(gt_bbox, min=0, max=args.size - 1)

        # forward + aux (qhat)
        pred_bbox, aux = model(image, masks, word_id, word_mask, return_aux=True)
        qhat = aux["qhat"]  # (B,1) in [0,1]

        # --- losses ---
        # GIoU dùng pixel
        giou = GIoU_Loss(pred_bbox * (args.size - 1), gt_bbox, args.size - 1)
        # L1 dùng norm (giữ như code gốc)
        gt_bbox_xywh = xyxy2xywh(gt_bbox)
        l1 = Reg_Loss(pred_bbox, gt_bbox_xywh / (args.size - 1))

        # IoU target cho quality head & weighting — dùng NORM hệ (cùng hệ với L1)
        pred_xyxy_norm = bbox_xywh_to_xyxy(pred_bbox)                  # (B,4) norm
        gt_xyxy_norm   = bbox_xywh_to_xyxy(gt_bbox_xywh / (args.size - 1))
        with torch.no_grad():
            iou_t = iou_xyxy(pred_xyxy_norm, gt_xyxy_norm)            # (B,1)

        # trọng số chất lượng nhẹ
        qw = (iou_t + 0.5)  # ∈ [0.5, 1.5]
        main_loss = (l1 + giou) * qw.mean()
        q_loss = quality_loss(qhat, iou_t, weight=0.2)

        loss = main_loss + q_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if sched is not None:
            sched.step()
        if ema is not None:
            ema.update(model)

        # logging loss
        losses.update(loss.item(), imgs.size(0))
        l1_losses.update(l1.item(), imgs.size(0))
        GIoU_losses.update(giou.item(), imgs.size(0))

        # ---- metrics (giữ nguyên logic cũ) ----
        pred_xyxy_px = torch.cat([pred_bbox[:, :2] - (pred_bbox[:, 2:] / 2),
                                  pred_bbox[:, :2] + (pred_bbox[:, 2:] / 2)], dim=1) * (args.size - 1)
        iou, interArea, unionArea = bbox_iou(pred_xyxy_px.data.cpu(), gt_bbox.data.cpu(), x1y1x2y2=True)
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
            # nếu có 3 nhóm param, nhóm thứ 3 thường là text
            try:
                lang_lr = optimizer.param_groups[2]['lr']
            except Exception:
                lang_lr = optimizer.param_groups[-1]['lr']
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
def validate_epoch(val_loader, model, args, ema_obj=None):
    batch_time = AverageMeter()
    losses = AverageMeter(); l1_losses = AverageMeter(); GIoU_losses = AverageMeter()
    acc5 = AverageMeter(); acc6 = AverageMeter(); acc7 = AverageMeter(); acc8 = AverageMeter(); acc9 = AverageMeter()
    meanIoU = AverageMeter(); inter_area = AverageMeter(); union_area = AverageMeter()

    # dùng EMA nếu có
    eval_model = ema_obj.ema if (ema_obj is not None) else model
    eval_model.eval()

    end = time.time()
    print(datetime.datetime.now())

    for batch_idx, (imgs, masks, word_id, word_mask, bbox) in enumerate(val_loader):
        imgs = imgs.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)
        masks = (masks[:, :, :, 0] == 255).bool()      # True = padding
        word_id = word_id.cuda(non_blocking=True)
        word_mask = word_mask.cuda(non_blocking=True)
        bbox = bbox.cuda(non_blocking=True)

        image = Variable(imgs); masks = Variable(masks)
        word_id = Variable(word_id); word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox, min=0, max=args.size - 1)
        gt_bbox = bbox

        # ---- TTA: pred gốc + flip ngang ----
        pred1 = eval_model(image, masks, word_id, word_mask)             # (B,4)
        image_flip = torch.flip(image, dims=[-1])
        pred2 = eval_model(image_flip, masks, word_id, word_mask)
        pred2[:, 0] = 1.0 - pred2[:, 0]  # cx -> 1 - cx
        pred_bbox = 0.5 * (pred1 + pred2)

        # ---- loss (để theo dõi, không tối ưu) ----
        loss = 0.
        giou = GIoU_Loss(pred_bbox * (args.size - 1), gt_bbox, args.size - 1); loss += giou
        gt_bbox_ = xyxy2xywh(gt_bbox)
        l1 = Reg_Loss(pred_bbox, gt_bbox_ / (args.size - 1)); loss += l1

        losses.update(loss.item(), imgs.size(0)); l1_losses.update(l1.item(), imgs.size(0)); GIoU_losses.update(giou.item(), imgs.size(0))

        # ---- metrics (giữ như cũ) ----
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

    # ===== Scheduler (Warmup→Cosine) & EMA =====
    steps_per_epoch = len(train_loader)
    total_steps = args.nb_epoch * max(1, steps_per_epoch)
    warmup_steps = max(100, int(0.02 * total_steps))  # ~2%
    sched = WarmupCosine(optimizer, base_lr=args.lr, warmup_steps=warmup_steps, total_steps=total_steps)
    ema = EMA(model, decay=0.999) if args.use_ema else None

    # train loop
    best_accu = -float('Inf')
    global_step = 0
    for epoch in range(args.nb_epoch):
        _ = train_epoch(train_loader, model, optimizer, epoch, args, sched=sched, ema=ema)
        v_metrics = validate_epoch(val_loader, model, args, ema_obj=ema if args.use_ema else None)

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
