import json
import torch
import os

def load_model(model, path, name, device):
    model.load_state_dict(torch.load(os.path.join(path, name + ".pt"), map_location=device))

    with open(os.path.join(path, name + ".json"), "r") as fin:
        data = json.load(fin)

    return data
