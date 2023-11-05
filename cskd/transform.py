from random import random, uniform
from PIL import ImageFilter, ImageOps
from torch import nn
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
import torchvision.transforms.functional as F

__all__ = ["build_transform"]

TINYIMAGENET_MEAN = (0.485, 0.456, 0.406)
TINYIMAGENET_STD = (0.229, 0.224, 0.225)


class RandomGaussianBlur(nn.Module):
    def __init__(self, sigma=(0.1, 2.0), p=0.5):
        super().__init__()
        self.sigma = sigma
        self.p = p

    def forward(self, image):
        if random() < self.p:
            image = image.filter(ImageFilter.GaussianBlur(uniform(*self.sigma)))
        return image

    def __repr__(self):
        s = f"sigma={self.sigma}, p={self.p}"
        return f"{self.__class__.__name__}({s})"


class RandomSolarize(nn.Module):
    def __init__(self, threshold=128, p=0.5):
        super().__init__()
        self.threshold = threshold
        self.p = p

    def forward(self, image):
        if random() < self.p:
            image = ImageOps.solarize(image, threshold=self.threshold)
        return image

    def __repr__(self):
        s = f"threshold={self.threshold}, p={self.p}"
        return f"{self.__class__.__name__}({s})"


class RandomGrayScale(nn.Module):
    def __init__(self, num_output_channels=1, p=0.5):
        super().__init__()
        self.num_output_channels = num_output_channels
        self.p = p

    def forward(self, image):
        if random() < self.p:
            image = F.rgb_to_grayscale(
                image, num_output_channels=self.num_output_channels
            )
        return image

    def __repr__(self):
        s = f"num_output_channels={self.num_output_channels}, p={self.p}"
        return f"{self.__class__.__name__}({s})"


def build_transform(config, train):
    transform = None
    if config.dataset_name in ("imagenet", "cifar10", "cifar100", "cars", "inat19"):
        if train:
            transform = create_transform(
                input_size=config.input_size,
                is_training=True,
                color_jitter=config.color_jitter,
                auto_augment=config.auto_augment,
                interpolation=config.train_interpolation,
                re_prob=config.random_erasing_prob,
                re_mode=config.random_erasing_mode,
                re_count=config.random_erasing_count,
            )
        else:
            resize_size = int(256 / 224 * config.input_size)
            transform = transforms.Compose(
                [
                    transforms.Resize(resize_size, F.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(config.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ]
            )
    elif config.dataset_name == "tinyimagenet":
        if train:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(config.input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(config.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
                ]
            )
    else:
        raise NotImplementedError(f"{config.datasetname} is not implemented")
    return transform
