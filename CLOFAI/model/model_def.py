import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision.models.efficientnet import EfficientNet_B0_Weights

class ModifiedEfficientNet(nn.Module):
    def __init__(self):
        super(ModifiedEfficientNet, self).__init__()
        base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_features = base_model.classifier[1].in_features
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # copy all layers except the last one
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
