import torch.nn as nn

from torchvision import models

class MobileNetModified(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.mobilenet_v3_small(pretrained=True)

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.post = nn.Sequential(
            nn.Linear(1000, 2)
        )

    def forward(self, inputs):
        x = self.model(inputs)
        x = self.post(x)
        return x