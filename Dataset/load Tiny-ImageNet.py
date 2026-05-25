import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class TinyImageNetValDataset(Dataset):
    def __init__(self, val_dir, class_to_idx, transform=None):
        """
        val_dir: tiny-imagenet-200/val
        class_to_idx: train_dataset.class_to_idx
        """
        self.val_dir = val_dir
        self.image_dir = os.path.join(val_dir, "images")
        self.transform = transform
        self.class_to_idx = class_to_idx

        annotation_file = os.path.join(val_dir, "val_annotations.txt")

        self.samples = []

        with open(annotation_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split("\t")
                img_name = parts[0]
                class_name = parts[1]

                img_path = os.path.join(self.image_dir, img_name)
                label = self.class_to_idx[class_name]

                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_TinyImageNet(
    TinyImageNet_PATH,
    batch_size=128,
    workers=8,
    pin_memory=True,
    img_size=64
):

    traindir = os.path.join(TinyImageNet_PATH, "train")
    valdir = os.path.join(TinyImageNet_PATH, "val")

    print("traindir =", traindir)
    print("valdir   =", valdir)

    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizer
    ])

    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalizer
    ])

    train_dataset = datasets.ImageFolder(
        root=traindir,
        transform=train_transform
    )

    val_dataset = TinyImageNetValDataset(
        val_dir=valdir,
        class_to_idx=train_dataset.class_to_idx,
        transform=val_transform
    )

    print("num_classes   =", len(train_dataset.classes))
    print("train_dataset =", len(train_dataset))
    print("val_dataset   =", len(val_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, train_dataset, val_dataset
