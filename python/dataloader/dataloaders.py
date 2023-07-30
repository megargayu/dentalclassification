from torch.utils.data import DataLoader

from dataloader.dataset import image_datasets

def create_dataloaders(batch_size):
  dataloaders = {
    "train": DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=True),
    "validation": DataLoader(image_datasets["validation"], batch_size=batch_size, shuffle=False),
    "test": DataLoader(image_datasets["test"], batch_size=batch_size, shuffle=False)
  }

  return dataloaders
