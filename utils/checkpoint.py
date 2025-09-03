import os
import shutil
import torch


def save_checkpoint(state: dict, is_best: bool, args, filename: str = "default"):
    """Lưu checkpoint và copy ra bản best nếu cần"""
    if filename == "default":
        filename = f"MGVLF_b{args.batch_size}_e{args.nb_epoch}_lr{args.lr}_seed{args.seed}"
    os.makedirs("./saved_models", exist_ok=True)

    ckpt_path = f"./saved_models/{filename}_checkpoint.pth.tar"
    best_path = f"./saved_models/{filename}_model_best.pth.tar"

    torch.save(state, ckpt_path)
    if is_best:
        shutil.copyfile(ckpt_path, best_path)


def load_pretrain(model, args, logging):
    """Nạp pretrain model (chỉ khớp các key trùng và cùng shape)"""
    if not os.path.isfile(args.pretrain):
        msg = f"=> no pretrained file found at '{args.pretrain}'"
        print(msg); logging.info(msg)
        return model

    checkpoint = torch.load(
        args.pretrain,
        map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    pretrained = checkpoint.get("state_dict", checkpoint)
    model_dict = model.state_dict()

    # filter các key hợp lệ
    matched = {k: v for k, v in pretrained.items()
               if k in model_dict and v.shape == model_dict[k].shape}

    if not matched:
        msg = f"=> no matching keys in pretrain file: {args.pretrain}"
        print(msg); logging.info(msg)
        return model

    model_dict.update(matched)
    model.load_state_dict(model_dict, strict=False)

    msg = f"=> loaded pretrain model: {args.pretrain} ({len(matched)} keys matched)"
    print(msg); logging.info(msg)

    del checkpoint
    torch.cuda.empty_cache()
    return model


def load_resume(model, args, logging):
    """Resume từ checkpoint đầy đủ"""
    if not os.path.isfile(args.resume):
        msg = f"=> no checkpoint found at '{args.resume}'"
        print(msg); logging.info(msg)
        return model, 0, None

    msg = f"=> loading checkpoint '{args.resume}'"
    print(msg); logging.info(msg)

    checkpoint = torch.load(
        args.resume,
        map_location="cuda" if torch.cuda.is_available() else "cpu"
    )

    start_epoch = checkpoint.get("epoch", 0)
    best_loss = checkpoint.get("best_loss", None)
    state = checkpoint.get("state_dict", checkpoint)

    model.load_state_dict(state, strict=False)

    msg = f"=> loaded checkpoint (epoch {start_epoch}) best_loss={best_loss}"
    print(msg); logging.info(msg)

    del checkpoint
    torch.cuda.empty_cache()
    return model, start_epoch, best_loss
