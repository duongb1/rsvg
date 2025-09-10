import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F

def save_checkpoint(state, is_best, args, filename='default'):
    if filename=='default':
        filename = 'MGVLF_batch%d_epoch%d_lr%d_seed%d' % (args.batch_size, args.nb_epoch,args.lr, args.seed)

    checkpoint_name = './saved_models/%s_checkpoint.pth.tar'%(filename)
    best_name = './saved_models/%s_model_best.pth.tar'%(filename)
    torch.save(state, checkpoint_name)
    if is_best:
        shutil.copyfile(checkpoint_name, best_name)

def load_pretrain(model, args, logging):
    if os.path.isfile(args.pretrain):
        checkpoint = torch.load(args.pretrain, map_location="cpu")

        # linh hoạt lấy state_dict
        if "state_dict" in checkpoint:
            pretrained_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            pretrained_dict = checkpoint["model"]
        else:
            # nếu file chỉ là raw state_dict
            pretrained_dict = checkpoint

        model_dict = model.state_dict()
        # lọc các key khớp shape
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }

        if len(pretrained_dict) == 0:
            print(f"=> WARNING: no matching keys in {args.pretrain}")
            logging.info(f"=> WARNING: no matching keys in {args.pretrain}")
        else:
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"=> loaded pretrain model at {args.pretrain} "
                  f"({len(pretrained_dict)} keys matched)")
            logging.info(f"=> loaded pretrain model at {args.pretrain} "
                         f"({len(pretrained_dict)} keys matched)")

        del checkpoint
        torch.cuda.empty_cache()
    else:
        print(f"=> no pretrained file found at '{args.pretrain}'")
        logging.info(f"=> no pretrained file found at '{args.pretrain}'")
    return model


def load_resume(model, args, logging):
    if os.path.isfile(args.resume):
        print(("=> loading checkpoint '{}'".format(args.resume)))
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        print(("=> loaded checkpoint (epoch {}) Loss{}"
              .format(checkpoint['epoch'], best_loss)))
        logging.info("=> loaded checkpoint (epoch {}) Loss{}"
              .format(checkpoint['epoch'], best_loss))
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    else:
        print(("=> no checkpoint found at '{}'".format(args.resume)))
        logging.info(("=> no checkpoint found at '{}'".format(args.resume)))
    return model