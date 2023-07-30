# TODO: convert `./notebooks` into python files

import numpy as np

import torch
from torch import nn, optim

from dataloader.dataloaders import create_dataloaders
from mobile.optimize import optimize_model, save_mobile_model
from models.models import models

from analysis.macs import get_macs

from train.train import train_model
from train.runphase import run_phase

from modelio.loadmodel import load_model
from modelio.savemodel import save_model

# Set random seed
SEED = 101
torch.manual_seed(SEED)
np.random.seed(SEED)

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Pick model
model_data = models[input(f"Which model? ({', '.join(list(models.keys()))}) > ")]

# Initialize model
model = model_data["model"]().to(device)

# Initialize dataloaders
dataloaders = create_dataloaders(model_data["batch_size"])

# Loss function & optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train or load?
should_train = input("Train or load model? (train, load) > ").lower() == "train"

if should_train:
  # Train the model
  accuracies, losses = train_model(dataloaders, model, loss_fn, optimizer, device, num_epochs=model_data["epochs"])
else:
  # Load the model from a path
  path = input("Path of model to load > ")
  name = input("Name of model to load > ")

  data = load_model(model, path, name, device)
  accuracies = data["accuracies"]
  losses = data["losses"]
  

# Test the model
print("Testing model...")
run_phase(dataloaders, model, loss_fn, optimizer, device, "test")
print("MACs used:", get_macs(model))

# Save model?
should_save = input("Save model? (Y/n) > ").lower() == "y"

if should_save:
  path = input("Path of output model > ")
  name = input("Name of output model > ")

  save_model(model, path, name, accuracies, losses)

# Convert to mobile version?
should_convert_mobile = input("Convert to mobile? (Y/n) > ").lower() == "y"

if should_convert_mobile:
  path = input("Path + name + .ptl of output mobile model > ")

  traced_script_module_optimized = optimize_model(model)
  save_mobile_model(traced_script_module_optimized, path)

print("Use files in the ./figures folder to get the code for generating the figures used in the paper!")
