import torch
import torch.nn as nn
from torchvision import models

class NeuroLensResNet18(nn.Module):
    """
    ResNet18 classifier for 2.5D MRI brain slices.
    """
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.3):
        super(NeuroLensResNet18, self).__init__()
        
        # Load torchvision ResNet18
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)

        # Replace final classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def build_model(num_classes=2, pretrained=True, device="cpu"):
    model = NeuroLensResNet18(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    return model

if __name__ == "__main__":
    model = build_model()
    dummy_input = torch.randn(4, 3, 224, 224)
    out = model(dummy_input)
    print("Model architecture loaded successfully.")
    print(f"Input shape: {dummy_input.shape} -> Output shape: {out.shape}")
