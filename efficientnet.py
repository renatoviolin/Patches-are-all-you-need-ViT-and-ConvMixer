import torch.nn as nn
import timm


class EfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = timm.create_model('tf_efficientnetv2_b0', num_classes=num_classes, pretrained=False)

    def forward(self, x):
        x = self.model(x)
        return x
