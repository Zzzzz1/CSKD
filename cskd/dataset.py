import os
from torchvision.datasets import ImageFolder
from .transform import build_transform


__all__ = ["build_dataset"]

data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/imagenet')

class ImageNet(ImageFolder):
    num_classes = 1000

def build_dataset(config, train):
    transform = build_transform(config, train)
    if config.dataset_name == "imagenet":
        train_folder = os.path.join(data_folder, 'train')
        val_folder = os.path.join(data_folder, 'val')
        folder = train_folder if train else val_folder
        dataset = ImageNet(folder, transform=transform)
    else:
        raise NotImplementedError(f"{config.datasetname} is not implemented")
    return dataset
