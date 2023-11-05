import os
import torch

from cskd.config import ConfigBase


class Config(ConfigBase):
    # base
    exp_name = os.path.splitext(os.path.basename(__file__))[0]

    # model
    model = "deit_base_distilled_patch16_224"
    input_size = 224
    dropout = 0.0
    drop_path = 0.0
    model_ema = True
    model_ema_decay = 0.99996
    model_ema_on_cpu = False

    # deit
    teacher_model = "regnety_160"
    teacher_weight = "https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth"
    deit_loss_type = "hard"  # none, soft, hard
    deit_alpha = 0.5
    deit_tau = 1.0
    # cskd
    cskd_decay_func = "linear"
    cskd_loss_type = "hard"
    cksd_loss_weight = 1.0

    # optimizer
    opt = "adamw"
    opt_eps = 1e-8
    opt_betas = None
    clip_grad = None
    momentum = 0.9
    weight_decay = 0.05

    # learning rate schedule
    sched = "cosine"
    lr = 2.5e-4
    lr_noise = None
    lr_noise_pct = 0.67
    lr_noise_std = 1.0
    warmup_lr = 1e-6
    min_lr = 1e-5
    decay_epochs = 30
    warmup_epochs = 5
    cooldown_epochs = 10
    patience_epochs = 10
    decay_rate = 0.1

    # augmentation
    color_jitter = 0.3
    auto_augment = "rand-m9-mstd0.5-inc1"
    label_smoothing = 0.1
    train_interpolation = "bicubic"
    repeated_aug = True
    # random erasing
    random_erasing_prob = 0.25
    random_erasing_mode = "pixel"
    random_erasing_count = 1
    # mixup
    use_mixup = True
    mixup_prob = 1.0
    mixup_alpha = 0.8
    mixup_switch_prob = 0.5
    mixup_mode = "batch"
    cutmix_alpha = 1.0
    cutmin_minmax = None

    # finetune
    finetune = ""
    attn_only = False

    # dataset
    dataset_name = "imagenet"
    seed = 0
    train_loader_workers = 8
    val_loader_workers = 8
    pin_memory = False

    # train
    continue_training = True
    epochs = 300
    train_batch_size = 256
    val_batch_size = 256
    weight_decay = 0.05

    # eval
    dist_eval = True

    # save
    save_latest_checkpoint_interval = 1
    save_checkpoint_interval = 5
    log_interval = 10

    # distributed
    ngpus_per_node = torch.cuda.device_count()
    dist_url = "tcp://localhost:12345"
    world_size = ngpus_per_node
    backend = "nccl"

    # path
    log_dir = os.path.join(os.getcwd(), "outputs", dataset_name, model, exp_name)
    log_path = os.path.join(log_dir, "worklog.log")


if __name__ == "__main__":
    cfg = Config.instance()
    cfg.print()
