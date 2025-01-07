from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import logging
from matplotlib import pyplot as plt

logging.getLogger("PIL").setLevel(logging.WARNING)
import os

os.environ["HF_USE_SOFTFILELOCK"] = "true"
from datasets.utils.logging import set_verbosity_error

set_verbosity_error()

# Constants
IMAGENET_DEFAULT_MEAN = [0.4802, 0.4481, 0.3976]
IMAGENET_DEFAULT_STD = [0.2302, 0.2265, 0.2262]


class HFTinyImagenet:
    def __init__(self, split, transform=None):
        self.dataset = load_dataset("Maysee/tiny-imagenet", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


def build_transform(is_train, img_size):
    resize_im = img_size > 32
    if is_train:
        transform = transforms.Compose(
            [
                # transforms.RandomCrop(64,padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
    return transform


def build_hf_tiny_imagenet_loader(config):
    # Dataset splits
    train_transform = build_transform(is_train=True, img_size=config.DATA.IMG_SIZE)
    val_transform = build_transform(is_train=False, img_size=config.DATA.IMG_SIZE)

    train_dataset = HFTinyImagenet(split="train", transform=train_transform)
    val_dataset = HFTinyImagenet(split="valid", transform=val_transform)

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
    mixed_up = None

    return train_dataset, val_dataset, train_loader, val_loader, mixed_up


class Config:
    class DATA:
        IMG_SIZE = 64
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
