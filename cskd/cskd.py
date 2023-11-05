import math
import torch
from torch.nn import functional as F

from .config import ConfigBase

__all__ = ["CSKDLoss"]


class CSKDLoss(torch.nn.Module):
    """
    Cumulative Spatial Knowledge Distillation Loss
    """
    def __init__(
        self, 
        cfg: ConfigBase,
        criterion: torch.nn.Module,
        teacher: torch.nn.Module,
    ):
        super().__init__()
        self.cfg = cfg
        self.criterion = criterion
        self.teacher = teacher

    def forward(
        self,
        inputs,
        outputs,
        labels,
        epoch,
        max_epoch
    ):
        if not isinstance(outputs, torch.Tensor):
            outputs, stu_deit_logits, stu_dense_logits = outputs
        loss_base = self.criterion(outputs, labels)
        if self.cfg.deit_loss_type == 'none':
            # no distill loss
            return loss_base

        with torch.no_grad():
            tea_dense_logits = self.teacher(inputs)
            tea_global_logits = tea_dense_logits.mean(dim=(2,3))
        loss_deit = self.get_loss_deit(stu_deit_logits, tea_global_logits)
        loss_cskd = self.get_loss_cskd(stu_dense_logits, tea_dense_logits, 
                    tea_global_logits, epoch, max_epoch)
        alpha = self.cfg.deit_alpha
        loss = loss_base * (1 - alpha) + loss_deit * alpha + \
            loss_cskd * self.cfg.cksd_loss_weight
        return loss

    def align_stu_logits(self, stu_dense_logits):
        N, M, C = stu_dense_logits.shape
        stu_dense_logits = stu_dense_logits.permute(0, 2, 1).reshape(N, C, 14, 14)
        stu_dense_logits = F.avg_pool2d(stu_dense_logits, kernel_size=2, stride=2)
        return stu_dense_logits

    def get_decay_ratio(self, epoch, max_epoch):
        x = epoch / max_epoch
        if self.cfg.cskd_decay_func == 'linear':
            return 1 - x
        elif self.cfg.cskd_decay_func == 'x2':
            return (1 - x) ** 2
        elif self.cfg.cskd_decay_func == 'cos':
            return math.cos(math.pi * 0.5 * x)
        else:
            raise NotImplementedError(self.cfg.cskd_decay_func)

    def get_loss_deit(
        self,
        stu_deit_logits,
        tea_global_logits,
    ):
        # deit loss
        if self.cfg.deit_loss_type == 'soft':
            T = self.cfg.deit_tau
            loss_deit = F.kl_div(
                F.log_softmax(stu_deit_logits / T, dim=1),
                F.log_softmax(tea_global_logits / T, dim=1),
                reduction='sum',
                log_target=True,
            ) * (T * T) / stu_deit_logits.numel()
        elif self.cfg.deit_loss_type == 'hard':
            loss_deit = F.cross_entropy(stu_deit_logits, tea_global_logits.argmax(dim=1))
        else:
            raise NotImplementedError(self.cfg.deit_loss_type)
        return loss_deit

    def get_loss_cskd(
        self,
        stu_dense_logits,
        teacher_dense_logits,
        teacher_global_logits,
        epoch,
        max_epoch,
    ):
        stu_dense_logits = self.align_stu_logits(stu_dense_logits)
        decay_ratio = self.get_decay_ratio(epoch, max_epoch)
        N, C = teacher_global_logits.shape
        teacher_logits = decay_ratio * teacher_dense_logits + \
            (1 - decay_ratio) * teacher_global_logits.reshape(N, C, 1, 1)
        # cskd loss
        if self.cfg.cskd_loss_type == "hard":
            loss_cskd = F.cross_entropy(
                stu_dense_logits, 
                teacher_logits.argmax(dim=1)
                )
        elif self.cfg.cskd_loss_type == "soft":
            T = self.cfg.deit_tau
            loss_cskd = F.kl_div(
                F.log_softmax(stu_dense_logits / T, dim=1),
                F.log_softmax(teacher_logits / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / stu_dense_logits.size(0)
        else:
            raise NotImplementedError(self.cfg.cskd_loss_type)
        return loss_cskd
