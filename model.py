import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetBackbone(nn.Module):

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = efficientnet_b0(weights=weights)

        self.feature_extractor = net.features
        self.out_channels = net.classifier[1].in_features

    def forward(self, x):
        return self.feature_extractor(x)


class RCNN(nn.Module):
    def __init__(
        self, num_classes: int, hidden_size: int = 256, pretrained: bool = True
    ):
        super().__init__()
        self.cnn = EfficientNetBackbone(pretrained=pretrained)

        self.pool = nn.AdaptiveAvgPool2d((1, None))

        self.rnn = nn.LSTM(
            input_size=self.cnn.out_channels,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        f = self.cnn(x)  # [B, C, H, W]
        f = self.pool(f)  # [B, C, 1, W]
        f = f.squeeze(2).permute(0, 2, 1)  # [B, W, C]

        r, _ = self.rnn(f)  # [B, W, 2*hidden]

        # CTC logits
        logits = self.fc(r)  # [B, W, num_classes]
        return logits.permute(1, 0, 2)  # [T, B, C] для CTC
