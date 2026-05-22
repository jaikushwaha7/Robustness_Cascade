import torch
import torch.nn as nn
import torchvision.models as models


# ─────────────────────────────────────────────
# SMALL MODEL (MS) — exact from paper Appendix C.1
# ─────────────────────────────────────────────

class SmallCNN(nn.Module):
    """
    Small model MS used in image classification experiments.
    Architecture taken verbatim from Appendix C.1 of the paper.

    Input : (B, 3, 32, 32)  — CIFAR-10 / CIFAR-100
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 32x32 → 16x16
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                         dilation=1, ceil_mode=False),

            # Block 2: 16x16 → 8x8
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                         dilation=1, ceil_mode=False),
        )

        # After two MaxPool2d(2,2): 32 → 16 → 8
        # Feature map: 32 channels × 8 × 8 = 2048
        self.classifier = nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)   # flatten → (B, 2048)
        return self.classifier(x)


# ─────────────────────────────────────────────
# LARGE MODEL (ML) — ResNet-18
# ─────────────────────────────────────────────

def get_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    Large model ML used in CIFAR-10 / CIFAR-100 experiments.

    Note: The paper trains ML from scratch on the target dataset,
    so pretrained=False by default. Set pretrained=True if you
    want to start from ImageNet weights and fine-tune instead.

    Args:
        num_classes : number of output classes
        pretrained  : whether to load ImageNet weights
    """
    '''weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model'''
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ─────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_s = SmallCNN(num_classes=10).to(device)
    model_l = get_resnet18(num_classes=10).to(device)

    dummy = torch.randn(4, 3, 32, 32).to(device)   # batch of 4 CIFAR images

    out_s = model_s(dummy)
    out_l = model_l(dummy)

    print(f"MS output shape : {out_s.shape}")   # → (4, 10)
    print(f"ML output shape : {out_l.shape}")   # → (4, 10)

    total_s = sum(p.numel() for p in model_s.parameters())
    total_l = sum(p.numel() for p in model_l.parameters())
    print(f"MS params : {total_s:,}")            # ~135K
    print(f"ML params : {total_l:,}")            # ~11.2M
