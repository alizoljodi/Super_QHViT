# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import numpy as np
import torch.distributed as dist
import torch.utils
from torchvision import datasets, transforms
import sys
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp
import torchvision.transforms as transforms, torchvision, matplotlib.pyplot as plt

# from on_device_ai.tools.data.imagenet import build_imagenet_dataset
sys.path.append(
    "/mimer/NOBACKUP/groups/naiss2023-5-522/ali/mixed_precision_one_shot_training/QNASViT_cifar10/data"
)

from cached_image_folder import CachedImageFolder
from samplers import SubsetRandomSampler
from PIL import Image, ImageFilter, ImageOps
import math
import random
import torchvision.transforms.functional as tf

'''class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""
    def __call__(self, x):
        return ImageOps.solarize(x)'''


def build_loader(config):
    if not os.path.isdir("./cifar_data"):
        os.mkdir("./cifar_data")
    trainset = torchvision.datasets.CIFAR10(
        root="./cifar_data/train",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        ),
    )

    valset = torchvision.datasets.CIFAR10(
        root="./cifar_data/val",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        ),
    )
    testset = torchvision.datasets.CIFAR10(
        root="./cifar_data/test",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        ),
    )
    traindir = os.path.join("./cifar_data", "train")
    valdir = os.path.join("./cifar_data", "val")

    config.defrost()

    # pre-process the train dataset by removing 2 image per-class
    # valdir = os.path.join(config.DATA.DATA_PATH, "val")

    # train_transform = build_transform(True, config)
    # test_transform = build_transform(False, config)
    # dataset_train = datasets.ImageFolder(traindir, train_transform)
    # dataset_val = datasets.ImageFolder(valdir, test_transform)

    config.freeze()
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset"
    )
    # dataset_val, _ = build_dataset(is_train=False, config=config)
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset"
    )

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    # sampler_train = torch.utils.data.DistributedSampler(trainset)

    # let's use distributed sampler for the val dataset as well
    # data loading is general slow for cloud jobs
    sampler_val = None
    persistent_workers = True
    # if config.workflow_run_id:
    #    sampler_val = torch.utils.data.DistributedSampler(valset)

    data_loader_train = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        # pin_memory=config.DATA.PIN_MEMORY,
        # drop_last=True,
        # collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=config.DATA.PREFETCH_FACTOR,
    )

    data_loader_val = torch.utils.data.DataLoader(
        valset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        # pin_memory=config.DATA.PIN_MEMORY,
        # drop_last=False,
        # collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=config.DATA.PREFETCH_FACTOR,
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = (
        config.AUG.MIXUP > 0
        or config.AUG.CUTMIX > 0.0
        or config.AUG.CUTMIX_MINMAX is not None
    )
    if False == True:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB,
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING,
            num_classes=config.MODEL.NUM_CLASSES,
        )

    return trainset, valset, data_loader_train, data_loader_val, mixup_fn


"""def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=32,
            is_training=True,
            
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(32, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)"""


if __name__ == "__main__":
    config = get_config(args)
    (
        dataset_train,
        dataset_val,
        data_loader_train,
        data_loader_val,
        mixup_fn,
    ) = build_loader(config)
