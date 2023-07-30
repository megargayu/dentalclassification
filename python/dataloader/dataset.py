import os

from torchvision import datasets

from dataloader.transforms import data_transforms

dataset_path = os.path.join(os.path.dirname(__file__), "../dataset/")

image_datasets = {
  "train": datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=data_transforms["train"]),
  "validation": datasets.ImageFolder(os.path.join(dataset_path, "valid"), transform=data_transforms["test"]),
  "test": datasets.ImageFolder(os.path.join(dataset_path, "test"), transform=data_transforms["test"]),
}
