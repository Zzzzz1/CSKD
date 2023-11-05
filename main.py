import os
import sys
import math
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.cuda import amp
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, ModelEma, get_state_dict

from cskd import build_dataset
from cskd import CSKDLoss
from cskd.models import create_model
from cskd.sampler import RASampler
from cskd.metric import AverageMeter, Accuracy
from cskd.utils import (
    load_config_from_file,
    save_checkpoint,
    load_checkpoint,
    load_model_ema,
    get_logger,
    Clock,
)


def train(
    local_rank,
    cfg,
    epoch,
    model,
    loader,
    mixup_func,
    criterion,
    loss_scaler,
    optimizer,
    model_ema,
    loss_meter,
    acc1_meter,
    acc5_meter,
    writer,
):
    logger = get_logger(cfg)
    if local_rank == 0:
        logger.info("start training...")

    model.train()
    for step, (inputs, target) in enumerate(loader):
        iteration = epoch * len(loader) + step
        inputs = inputs.to(local_rank, non_blocking=True)
        target = target.to(local_rank, non_blocking=True)
        if cfg.use_mixup:
            inputs, target = mixup_func(inputs, target)
        with amp.autocast():
            outputs = model(inputs)
            loss = criterion(inputs, outputs, target, epoch, cfg.epochs)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            logger.error(f"loss is {loss_value}, stop training")
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=cfg.clip_grad,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        if model_ema is not None:
            model_ema.update(model)

        loss_meter.update(loss_value)
        if not cfg.use_mixup:
            acc1_meter.update(outputs, target)
            acc5_meter.update(outputs, target)

        if (iteration + 1) % cfg.log_interval != 0:
            continue
        train_loss = loss_meter.sync_and_pop()
        if cfg.use_mixup:
            acc1, acc5 = 0.0, 0.0
        else:
            acc1, acc5 = acc1_meter.sync_and_pop(), acc5_meter.sync_and_pop()

        if local_rank != 0:
            continue
        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        info_str = f"[train] "
        info_str += f"epoch: {epoch + 1}/{cfg.epochs}, "
        info_str += f"step: {step + 1}/{len(loader)}, "
        info_str += f"iter: {iteration + 1}/{cfg.epochs * len(loader)}"
        info_str += "({:.2%}), ".format((iteration + 1)/(cfg.epochs * len(loader)))
        info_str += f"lr: {current_lr:7f}, "
        info_str += f"loss: {train_loss}"
        if not cfg.use_mixup:
            info_str += f", acc1: {acc1}, acc5: {acc5}"
        logger.parent.handlers.clear()
        logger.info(info_str)

        writer.add_scalar("loss/train", train_loss, iteration + 1)
        writer.add_scalar("lr/lr", current_lr, iteration + 1)
        if not cfg.use_mixup:
            writer.add_scalar("acc/train_acc1", acc1, iteration + 1)
            writer.add_scalar("acc/train_acc5", acc5, iteration + 1)
        writer.flush()


@torch.no_grad()
def evaluate(
    local_rank,
    cfg,
    train_loader,
    loader,
    epoch,
    model,
    loss_meter,
    acc1_meter,
    acc5_meter,
    writer,
):
    logger = get_logger(cfg)
    if local_rank == 0:
        logger.info("start evaluating...")

    model.eval()
    criterion = nn.CrossEntropyLoss()
    for step, (inputs, target) in enumerate(loader):
        inputs = inputs.to(local_rank, non_blocking=True)
        target = target.to(local_rank, non_blocking=True)

        with amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, target)

        loss_meter.update(loss.item())
        acc1_meter.update(outputs, target)
        acc5_meter.update(outputs, target)

        if (step + 1) % cfg.log_interval == 0:
            if local_rank == 0:
                logger.info(f"[val] step: {step + 1}/{len(loader)}")

    val_loss = loss_meter.sync_and_pop()
    acc1 = acc1_meter.sync_and_pop()
    acc5 = acc5_meter.sync_and_pop()

    if local_rank == 0:
        info_str = f"[val] "
        info_str += f"epoch: {epoch + 1}/{cfg.epochs}, "
        info_str += f"loss: {val_loss}, acc1: {acc1}, acc5: {acc5}"
        logger.info(info_str)

        iteration = (epoch + 1) * len(train_loader)
        writer.add_scalar("loss/val", val_loss, iteration + 1)
        writer.add_scalar("acc/val_acc1", acc1, iteration + 1)
        writer.add_scalar("acc/val_acc5", acc5, iteration + 1)
        writer.flush()

    
    return acc1, acc5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="config file")
    parser.add_argument("--eval-only", action="store_true", default=False, help="eval only")
    parser.add_argument("--ckpt", type=str, default="", help="ckpt to eval")
    args = parser.parse_args()
    config = load_config_from_file(args.config)
    cfg = config.instance()

    # init distributed process
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend=cfg.backend,
        init_method=cfg.dist_url,
        world_size=cfg.world_size,
        rank=local_rank,
    )
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        cfg.print()

    # ensure output dir
    train_writer, val_writer = None, None
    if local_rank == 0:
        os.makedirs(cfg.log_dir, exist_ok=True)
        train_writer = SummaryWriter(os.path.join(cfg.log_dir, "train.events"))
        val_writer = SummaryWriter(os.path.join(cfg.log_dir, "val.events"))
        clock = Clock()

    dist.barrier()
    logger = get_logger(cfg)
    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    train_acc1_meter = Accuracy()
    train_acc5_meter = Accuracy(topk=5)
    val_acc1_meter = Accuracy()
    val_acc5_meter = Accuracy(topk=5)

    if cfg.deit_loss_type != "none" and cfg.finetune and not cfg.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    # reproducibility
    seed = cfg.seed + local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    if local_rank == 0:
        logger.info("start load dataset")
    
    # dataset
    train_dataset = build_dataset(cfg, train=True)
    val_dataset = build_dataset(cfg, train=False)

    # data sampler
    if cfg.repeated_aug:
        train_sampler = RASampler(train_dataset, shuffle=True)
    else:
        train_sampler = data.DistributedSampler(train_dataset, shuffle=True)
    if cfg.dist_eval:
        if local_rank == 0:
            logger.warning(
                "Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )
        val_sampler = data.DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = data.SequentialSampler(val_dataset)

    # data loader
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size // cfg.world_size,
        sampler=train_sampler,
        num_workers=cfg.train_loader_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=cfg.val_batch_size // cfg.world_size
        if cfg.dist_eval
        else cfg.val_batch_size,
        sampler=val_sampler,
        num_workers=cfg.val_loader_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )
    if local_rank == 0:
        logger.info("finish load dataset")

    if local_rank == 0:
        logger.info("start load model")
    model = create_model(
        cfg.model,
        pretrained=False,
        num_classes=train_dataset.num_classes,
        drop_rate=cfg.dropout,
        drop_path_rate=cfg.drop_path,
        drop_block_rate=None,
    )

    # finetune
    if cfg.finetune:
        if local_rank == 0:
            logger.info("finetune")
        if cfg.finetune.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                cfg.finetune, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(cfg.finetune, map_location="cpu")
        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
            if (
                k in checkpoint_model
                and (not k in state_dict or checkpoint_model[k].shape != state_dict[k].shape)
            ):
                if local_rank == 0:
                    logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size)
        pos_tokens = pos_tokens.permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model["pos_embed"] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    if cfg.attn_only:
        if local_rank == 0:
            logger.info("finetune attn only")
        for name_p, p in model.named_parameters():
            if ".attn." in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            if local_rank == 0:
                logger.warning("no position encoding")
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            if local_rank == 0:
                logger.warning("no patch embed")

    model.to(local_rank)

    model_ema = None
    if cfg.model_ema:
        model_ema = ModelEma(
            model,
            decay=cfg.model_ema_decay,
            device="cpu" if cfg.model_ema_on_cpu else "",
        )

    model_without_ddp = model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    model_without_ddp = model.module

    if args.eval_only:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        acc1, acc5 = evaluate(local_rank, cfg, train_loader, val_loader, 0, model, val_loss_meter, val_acc1_meter, val_acc5_meter, val_writer)
        return
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if local_rank == 0:
        logger.info("finish load model")
        logger.info(f"number of params: {n_parameters}")

    mixup_func = None
    if cfg.use_mixup:
        mixup_func = Mixup(
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            cutmix_minmax=cfg.cutmin_minmax,
            prob=cfg.mixup_prob,
            switch_prob=cfg.mixup_switch_prob,
            mode=cfg.mixup_prob,
            label_smoothing=cfg.label_smoothing,
            num_classes=train_dataset.num_classes,
        )

    optimizer = create_optimizer(cfg, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(cfg, optimizer)

    criterion = LabelSmoothingCrossEntropy()
    if cfg.use_mixup:
        criterion = SoftTargetCrossEntropy()
    elif cfg.label_smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=cfg.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    teacher_model = None
    if cfg.deit_loss_type != "none":
        if not cfg.teacher_weight:
            if local_rank == 0:
                logger.error("need to specify teacher_weight when using distillation")
        if local_rank == 0:
            logger.info(f"create teacher model: {cfg.teacher_model}")
        teacher_model = create_model(
            cfg.teacher_model,
            pretrained=False,
            num_classes=train_dataset.num_classes,
            global_pool="avg",
        )
        if cfg.teacher_weight.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                cfg.teacher_weight, map_location="cpu", check_hash=True
            )
        else:
            ckpt_name = os.path.basename(cfg.teacher_weight)
            ckpt_dir = os.path.dirname(cfg.teacher_weight)
            checkpoint = load_checkpoint(ckpt_dir, ckpt_name)
        teacher_model.load_state_dict(checkpoint["model"])
        teacher_model.to(local_rank)
        teacher_model.eval()

    criterion = CSKDLoss(
        cfg,
        criterion,
        teacher_model
    )

    start_epoch = 0
    best_val_acc = 0
    save_best = False
    if cfg.continue_training:
        checkpoint = load_checkpoint(cfg.log_dir, "latest")
        if checkpoint:
            if local_rank == 0:
                logger.info("resume from checkpoint")
            start_epoch = checkpoint["epoch"]
            best_val_acc = checkpoint["best_val_acc"]
            model_without_ddp.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            loss_scaler.load_state_dict(checkpoint["scaler"])
            if cfg.model_ema:
                load_model_ema(model_ema, checkpoint["model_ema"])
            if local_rank == 0:
                logger.info("successfully loaded the checkpoint from latest")
        elif start_epoch == 0:
            pass
        else:
            if local_rank == 0:
                logger.info("unable to load checkpoint from latest")
        del checkpoint

    if local_rank == 0:
        if start_epoch < cfg.epochs:
            logger.info(f"start from epoch {start_epoch + 1}")

    if local_rank == 0:
        clock.start()
    for epoch in range(start_epoch, cfg.epochs):
        train_loader.sampler.set_epoch(epoch)
        # fmt: off
        train(local_rank, cfg, epoch, model, train_loader, mixup_func, criterion, loss_scaler, optimizer, model_ema, train_loss_meter, train_acc1_meter, train_acc5_meter, train_writer)
        lr_scheduler.step(epoch)
        acc1, acc5 = evaluate(local_rank, cfg, train_loader, val_loader, epoch, model, val_loss_meter, val_acc1_meter, val_acc5_meter, val_writer)
        # fmt: on
        if local_rank == 0:
            if acc1 > best_val_acc:
                best_val_acc = acc1
                save_best = True
            epoch += 1
            checkpoint = {
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "scaler": loss_scaler.state_dict(),
                "model_ema": get_state_dict(model_ema),
            }

            if epoch % cfg.save_latest_checkpoint_interval == 0:
                if save_checkpoint(cfg.log_dir, "latest", checkpoint):
                    logger.info("successfully saved the checkpoint to latest")
                else:
                    logger.warning("unable to save checkpoint to latest")
            if epoch % cfg.save_checkpoint_interval == 0:
                if save_checkpoint(cfg.log_dir, f"epoch_{epoch}", checkpoint):
                    logger.info(f"successfully saved the checkpoint to epoch_{epoch}")
                else:
                    logger.warning(f"unable to save checkpoint to epoch_{epoch}")
            if save_best:
                if save_checkpoint(cfg.log_dir, "best", checkpoint):
                    logger.info("successfully saved the checkpoint to best")
                else:
                    logger.warning("unable to save checkpoint to best")
            del checkpoint
            save_best = False

            log_info = f"total time spent on training and evaluation "
            log_info += f"(epoch {epoch}): {clock.lap()} sec, best acc: {best_val_acc}"
            logger.info(log_info)

    if local_rank == 0:
        clock.stop()
        log_info = f"finish training, "
        log_info += f"total time: {clock.get()} sec, best acc: {best_val_acc}"
        logger.info(log_info)


if __name__ == "__main__":
    main()
