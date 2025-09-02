import os
import shutil
import torch


def save_checkpoint(state: dict, is_best: bool, args, filename: str = "default"):
    if filename == "default":
        filename = f"MGVLF_batch{args.batch_size}_epoch{args.nb_epoch}_lr{args.lr}_seed{args.seed}"
    os.makedirs("./saved_models", exist_ok=True)
    ckpt_path = f"./saved_models/{filename}_checkpoint.pth.tar"
    best_path = f"./saved_models/{filename}_model_best.pth.tar"
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copyfile(ckpt_path, best_path)


def load_pretrain(model, args, logging):
    if os.path.isfile(args.pretrain):
        checkpoint = torch.load(args.pretrain, map_location="cpu")
        pretrained = checkpoint.get("state_dict", checkpoint)
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in pretrained.items() if k in model_dict and v.shape == model_dict[k].shape}
        if not pretrained:
            print(f"=> no matching keys in pretrain file: {args.pretrain}")
            logging.info(f"=> no matching keys in pretrain file: {args.pretrain}")
            return model
        model_dict.update(pretrained)
        model.load_state_dict(model_dict, strict=False)
        print(f"=> loaded pretrain model: {args.pretrain} ({len(pretrained)} keys)")
        logging.info(f"=> loaded pretrain model: {args.pretrain} ({len(pretrained)} keys)")
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print(f"=> no pretrained file found at '{args.pretrain}'")
        logging.info(f"=> no pretrained file found at '{args.pretrain}'")
    return model


def load_resume(model, args, logging):
    if os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        logging.info(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location="cpu")
        args.start_epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("best_loss", None)
        state = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state, strict=False)
        print(f"=> loaded checkpoint (epoch {args.start_epoch}) best={best_loss}")
        logging.info(f"=> loaded checkpoint (epoch {args.start_epoch}) best={best_loss}")
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print(f"=> no checkpoint found at '{args.resume}'")
        logging.info(f"=> no checkpoint found at '{args.resume}'")
    return model
