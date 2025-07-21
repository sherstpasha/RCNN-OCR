import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from torchvision.models import ResNet18_Weights, ResNet50_Weights


class MSFBackbone(nn.Module):
    """Multi-scale Feature Fusion wrapper for a CNN backbone"""

    def __init__(self, stage1, stage2, stage3, stage4, stage5, use_msf: bool = True):
        super().__init__()
        self.stage1, self.stage2 = stage1, stage2
        self.stage3, self.stage4 = stage3, stage4
        self.stage5 = stage5
        self.use_msf = use_msf
        if use_msf:
            out_ch = self.stage5[-1].conv3.out_channels
            self.fuse_conv = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        if self.use_msf:
            up = F.interpolate(
                x4, size=x5.shape[2:], mode="bilinear", align_corners=False
            )
            cat = torch.cat([up, x5], dim=1)
            return self.fuse_conv(cat)
        return x5


class ConvNeXtMSF(nn.Module):
    """MSF fusion for ConvNeXt features_only backbone"""

    def __init__(self, features_model, use_msf: bool = True):
        super().__init__()
        # features_only returns list of feature maps
        self.features = features_model
        self.use_msf = use_msf
        if use_msf:
            ch4 = self.features.feature_info[-2]["num_chs"]
            ch5 = self.features.feature_info[-1]["num_chs"]
            self.fuse_conv = nn.Conv2d(ch4 + ch5, ch5, kernel_size=1)

    def forward(self, x):
        feats = self.features(x)
        f4, f5 = feats[-2], feats[-1]
        if self.use_msf:
            up = F.interpolate(
                f4, size=f5.shape[2:], mode="bilinear", align_corners=False
            )
            cat = torch.cat([up, f5], dim=1)
            return self.fuse_conv(cat)
        return f5


class OCRCTC(nn.Module):
    """
    OCR model using CTC loss.

    Supported backbones:
      - 'res18'       ResNet-18
      - 'res50'       ResNet-50
      - 'convnextv2'  ConvNeXtV2 Tiny

    Args:
      imgH: Image height (unused internally)
      imgW: Image width  (unused internally)
      num_classes: Number of output CTC classes (including blank)
      backbone: Which backbone to use
      pretrained: Load ImageNet weights
      msf: Multi-scale feature fusion on/off
      max_seq_len: fixed output temporal length (None for dynamic)
    """

    def __init__(
        self,
        imgH: int,
        imgW: int,
        num_classes: int,
        backbone: str = "res18",
        pretrained: bool = True,
        msf: bool = True,
        max_seq_len: int = None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        # choose backbone
        if backbone == "res18":
            base = models.resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            # adjust first conv to 1 channel
            orig = base.conv1
            base.conv1 = nn.Conv2d(
                1,
                orig.out_channels,
                orig.kernel_size,
                orig.stride,
                orig.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    base.conv1.weight[:] = orig.weight.sum(dim=1, keepdim=True)
            base.maxpool = nn.Identity()
            cnn_out = 512
            # extract stages
            s1 = nn.Sequential(*list(base.children())[:4])
            s2, s3, s4, s5 = base.layer1, base.layer2, base.layer3, base.layer4
            self.cnn = MSFBackbone(s1, s2, s3, s4, s5, use_msf=msf)

        elif backbone == "res50":
            base = models.resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
            orig = base.conv1
            base.conv1 = nn.Conv2d(
                1,
                orig.out_channels,
                orig.kernel_size,
                orig.stride,
                orig.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    base.conv1.weight[:] = orig.weight.sum(dim=1, keepdim=True)
            base.maxpool = nn.Identity()
            cnn_out = 2048
            s1 = nn.Sequential(*list(base.children())[:4])
            s2, s3, s4, s5 = base.layer1, base.layer2, base.layer3, base.layer4
            self.cnn = MSFBackbone(s1, s2, s3, s4, s5, use_msf=msf)

        elif backbone == "convnextv2":
            feats = timm.create_model(
                "convnextv2_tiny", pretrained=pretrained, features_only=True, in_chans=1
            )
            cnn_out = feats.feature_info[-1]["num_chs"]
            self.cnn = ConvNeXtMSF(feats, use_msf=msf)

        else:
            raise ValueError(f"Unknown backbone {backbone}")

        # sequence pooling
        if self.max_seq_len:
            self.pool = nn.AdaptiveAvgPool2d((1, self.max_seq_len))
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, None))

        # recurrent CTC head
        self.rnn = nn.LSTM(
            cnn_out, 256, num_layers=2, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
          x: Input tensor of shape (B, 1, H, W)
        Returns:
          Tensor of shape (T, B, C) for CTC loss and decoding
        """
        f = self.cnn(x)
        f = self.pool(f)  # B, C, 1, W
        B, C, _, W = f.shape
        f = f.view(B, C, W).permute(0, 2, 1)  # B, W, C
        r, _ = self.rnn(f)
        logits = self.fc(r)  # B, W, classes
        return logits.permute(1, 0, 2)  # T, B, C
