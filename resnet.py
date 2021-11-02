from torchvision import models
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = models.resnet18(False)
        self.model.fc = nn.Sequential(
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x