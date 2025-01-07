from datasets import load_dataset, load_from_disk
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class ImageNet100ArrowDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        """
        Args:
            hf_dataset: Hugging Face Dataset object.
            transform: Transformations to apply to the images.
        """
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]  # Adjust this key based on the dataset schema
        label = sample["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


def build_imagenet100_arrow_loader(batch_size=64, num_workers=8, img_size=224):
    # Load the dataset from Arrow format
    train_dataset = load_from_disk(
        "/mimer/NOBACKUP/groups/naiss2023-5-522/ali/mixed_precision_one_shot_training/imagenet-100/scripts/cache/imagenet-100_224/train",
        split="train",
    )
    val_dataset = load_from_disk(
        "/mimer/NOBACKUP/groups/naiss2023-5-522/ali/mixed_precision_one_shot_training/imagenet-100/scripts/cache/imagenet-100_224/validation",
        split="validation",
    )

    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Wrap datasets for PyTorch
    train_data = ImageNet100ArrowDataset(train_dataset, transform=transform)
    val_data = ImageNet100ArrowDataset(val_dataset, transform=transform)

    # Data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    build_imagenet100_arrow_loader()
