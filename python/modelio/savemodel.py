import torch
import json
import os

def save_model(model, path, name, accuracies, losses):
    name = os.path.join(path, name)

    # Save the model
    torch.save(model.state_dict(), name + ".pt")

    # Fix train & validation curves for accuracies and losses to make them serializable
    acc_fixed = {item: torch.stack(accuracies[item]).tolist() 
                        if not isinstance(accuracies[item][0], float) else accuracies[item] 
                        for item in accuracies.keys()}

    loss_fixed = losses

    # Dump!
    with open(name + ".json", "w") as fout:
        json.dump({
            "accuracies": acc_fixed,
            "losses": loss_fixed
        }, fout)
