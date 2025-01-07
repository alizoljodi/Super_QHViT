from torchvision import transforms
from torch import FloatTensor, div
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode
import pickle, torch
import numpy as np
from torch.utils.data.distributed import DistributedSampler

from matplotlib import pyplot as plt

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp
from timm.data.random_erasing import RandomErasing


# Constants
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


class ImageNetDataset(Dataset):
    def __init__(self, dataset, labels, transform=None, normalize=None):
        super(ImageNetDataset, self).__init__()
        assert len(dataset) == len(labels)
        self.dataset = dataset
        self.labels = labels
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.transform:
            data = self.transform(data)

        data = div(data.type(FloatTensor), 255)
        if self.normalize:
            data = self.normalize(data)

        return data, self.labels[idx]


def build_transform(is_train, img_size):
    resize_im = img_size > 32
    if is_train:
        transform = transforms.Compose(
            [
                transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
                transforms.RandAugment(num_ops=2, magnitude=0),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            ]
        )
    return transform


def build_hf_tiny_imagenet_loader(config):
    # Dataset splits
    train_transform = build_transform(is_train=True, img_size=config.DATA.IMG_SIZE)
    val_transform = build_transform(is_train=False, img_size=config.DATA.IMG_SIZE)
    with open(
        "/mimer/NOBACKUP/groups/naiss2023-5-522/ali/mixed_precision_one_shot_training/TinyIMGNT/data/train_dataset.pkl",
        "rb",
    ) as f:
        train_data, train_labels = pickle.load(f)
    with open(
        "/mimer/NOBACKUP/groups/naiss2023-5-522/ali/mixed_precision_one_shot_training/TinyIMGNT/data/val_dataset.pkl",
        "rb",
    ) as f:
        val_data, val_labels = pickle.load(f)

    train_dataset = ImageNetDataset(
        train_data,
        train_labels.type(torch.LongTensor),
        train_transform,
        normalize=transforms.Compose(
            [
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                )
            ]
        ),
    )
    val_dataset = ImageNetDataset(
        val_data,
        val_labels.type(torch.LongTensor),
        train_transform,
        normalize=transforms.Compose(
            [
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        ),
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=config.DATA.BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        sampler=DistributedSampler(train_dataset),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(val_dataset),
    )
    mixup_fn = None

    mixup_active = (
        config.AUG.MIXUP > 0
        or config.AUG.CUTMIX > 0.0
        or config.AUG.CUTMIX_MINMAX is not None
    )
    if mixup_active:
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

    random_erase = RandomErasing(probability=0.25, mode="pixel")

    return train_dataset, val_dataset, train_loader, val_loader, mixup_fn, random_erase


class Config:
    class DATA:
        IMG_SIZE = 384
        BATCH_SIZE = 64
        NUM_WORKERS = 4
        PIN_MEMORY = True


config = Config()

if __name__ == "__main__":
    (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
        _,
    ) = build_hf_tiny_imagenet_loader(config)

    for idx, (samples, targets) in enumerate(train_loader):
        print(samples.shape, targets.shape)
        plt.imshow(samples[1].permute(1, 2, 0))
        # plt.show()
        plt.savefig(
            "/mimer/NOBACKUP/groups/naiss2023-5-522/ali/mixed_precision_one_shot_training/TinyIMGNT/data/test_images/"
            + str(idx)
            + "test.png"
        )
        # sys.exit(0)
        # Uncomment below to only test the first batch
        # break
