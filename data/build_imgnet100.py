from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

import logging

logging.getLogger("PIL").setLevel(logging.WARNING)
import os

os.environ["HF_USE_SOFTFILELOCK"] = "true"

from datasets.utils.logging import set_verbosity_error

set_verbosity_error()
# Constants
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


class HFImagenet100:
    def __init__(self, split, transform=None):
        self.dataset = load_dataset("clane9/imagenet-100", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # print(item)
        image = item["image"].convert("RGB")
        label = item["label"]
        # print(type(image))
        if self.transform:
            image = self.transform(image)
        return image, label


def build_transform(is_train, img_size):
    resize_im = img_size > 32
    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(int((256 / 224) * img_size)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
    return transform


def build_hf_loader(config):
    # Dataset splits
    train_transform = build_transform(is_train=True, img_size=config.DATA.IMG_SIZE)
    val_transform = build_transform(is_train=False, img_size=config.DATA.IMG_SIZE)

    train_dataset = HFImagenet100(split="train", transform=train_transform)
    val_dataset = HFImagenet100(split="validation", transform=val_transform)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )
    mixed_fn = None

    return train_dataset, val_dataset, train_loader, val_loader, mixed_fn


class Config:
    class DATA:
        IMG_SIZE = 224
        BATCH_SIZE = 64
        NUM_WORKERS = 4
        PIN_MEMORY = True


config = Config()
if __name__ == "__main__":
    train_dataset, val_dataset, train_loader, val_loader = build_hf_loader(config)

    for idx, (samples, targets) in enumerate(train_loader):
        print(samples.shape, targets.shape)
        # break
