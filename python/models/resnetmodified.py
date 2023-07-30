import torch.nn as nn

from torchvision import models

class ResNetModified(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet34(pretrained=True)

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(512, 2)
        )

    def forward(self, inputs):
        x = self.model(inputs)
        return x
