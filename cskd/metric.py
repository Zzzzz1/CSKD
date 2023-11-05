import json
import torch
import torch.distributed as dist

__all__ = ["AverageMeter", "Accuracy", "AccuracyImgNetReal"]


class _BaseMetric:
    def __init__(self, ndigits=5, eps=1e-9):
        self.count = 0
        self.total = 0
        self.ndigits = ndigits
        self.eps = eps

    def update(self, count_value, total_value):
        self.count += count_value
        self.total += total_value

    def get(self):
        return round(self.count / max(self.total, self.eps), self.ndigits)

    def clear(self):
        self.count = 0
        self.total = 0

    def pop(self):
        average_value = self.get()
        self.clear()
        return average_value

    def sync(self):
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = t[0]
        self.total = t[1]

    def sync_and_pop(self):
        self.sync()
        return self.pop()


class AverageMeter(_BaseMetric):
    def __init__(self, ndigits=5, eps=1e-9):
        super().__init__(ndigits, eps)

    def update(self, count_value, total_value=1):
        self.count += count_value
        self.total += total_value


class Accuracy(_BaseMetric):
    def __init__(self, topk=1, ndigits=5, eps=1e-9):
        super().__init__(ndigits, eps)
        self.topk = topk

    @torch.no_grad()
    def update(self, predict, target):
        _, pred = torch.topk(predict, self.topk, dim=1)
        pred.t_()
        correct = pred == target.reshape(1, -1).expand_as(pred)
        self.count += correct.reshape(-1).int().sum().item()
        self.total += target.size(0)


class AccuracyImgNetReal(_BaseMetric):

    def __init__(self, topk=1, ndigits=5, eps=1e-9):
        super().__init__(ndigits, eps)
        real_json='/home/zhaoborui/reassessed-imagenet/real.json'
        with open(real_json) as real_labels:
            real_labels = json.load(real_labels)
        self.real_labels = real_labels
        self.topk = topk

    @torch.no_grad()
    def update(self, output, index):
        _, pred_batch = output.topk(self.topk, 1, True, True)
        pred_batch = pred_batch.cpu().numpy()
        for i, pred in enumerate(pred_batch):
            idx = index[i]
            if self.real_labels[idx]:
                self.count += any([p in self.real_labels[idx] for p in pred[:self.topk]])
                self.total += 1
