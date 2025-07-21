import torch
import torch.nn as nn
from kornia.geometry.transform import get_tps_transform, warp_image_tps
from torch import Tensor
from typing import Optional


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def build_fiducial_points(num_fiducial: int) -> torch.Tensor:
    assert num_fiducial % 2 == 0, "num_fiducial должно быть чётным"
    half = num_fiducial // 2
    xs = torch.linspace(-1.0, 1.0, steps=half)
    top = torch.stack([xs, torch.full_like(xs, -1.0)], dim=1)
    bot = torch.stack([xs, torch.full_like(xs, +1.0)], dim=1)
    return torch.cat([top, bot], dim=0)


class ResNet(nn.Module):
    """Full ResNet implementation for OCR (uses BasicBlock)."""

    def __init__(
        self, input_channel: int, output_channel: int, block, layers: list[int]
    ):
        super().__init__()
        self.output_channel_block = [
            output_channel // 4,
            output_channel // 2,
            output_channel,
            output_channel,
        ]
        self.inplanes = output_channel // 8
        # initial conv layers
        self.conv0_1 = nn.Conv2d(
            input_channel, output_channel // 16, 3, 1, 1, bias=False
        )
        self.bn0_1 = nn.BatchNorm2d(output_channel // 16)
        self.conv0_2 = nn.Conv2d(
            output_channel // 16, self.inplanes, 3, 1, 1, bias=False
        )
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # stage1
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(
            self.output_channel_block[0],
            self.output_channel_block[0],
            3,
            1,
            1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])
        # stage2
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.layer2 = self._make_layer(
            block, self.output_channel_block[1], layers[1], stride=1
        )
        self.conv2 = nn.Conv2d(
            self.output_channel_block[1],
            self.output_channel_block[1],
            3,
            1,
            1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])
        # stage3
        self.maxpool3 = nn.MaxPool2d((2, 1), (2, 1), padding=(0, 0))
        self.layer3 = self._make_layer(
            block, self.output_channel_block[2], layers[2], stride=1
        )
        self.conv3 = nn.Conv2d(
            self.output_channel_block[2],
            self.output_channel_block[2],
            3,
            1,
            1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])
        # stage4
        self.layer4 = self._make_layer(
            block, self.output_channel_block[3], layers[3], stride=1
        )
        self.conv4_1 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            2,
            (2, 1),
            (0, 1),
            bias=False,
        )
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            2,
            1,
            0,
            bias=False,
        )
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion, 1, stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn0_1(self.conv0_1(x)))
        x = self.relu(self.bn0_2(self.conv0_2(x)))
        x = self.maxpool1(x)
        x = self.relu(self.bn1(self.conv1(self.layer1(x))))
        x = self.maxpool2(x)
        x = self.relu(self.bn2(self.conv2(self.layer2(x))))
        x = self.maxpool3(x)
        x = self.relu(self.bn3(self.conv3(self.layer3(x))))
        x = self.layer4(x)
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        return x


class TPS_STN(nn.Module):
    """Thin-Plate Spline Spatial Transformer"""

    def __init__(self, num_fiducial: int, H: int, W: int):
        super().__init__()
        self.num_fiducial = num_fiducial
        feat_h, feat_w = H // 8, W // 8
        self.loc_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((feat_h, feat_w)),
            nn.Flatten(),
            nn.Linear(128 * feat_h * feat_w, 256),
            nn.ReLU(),
            nn.Linear(256, num_fiducial * 2),
        )
        ctrl = build_fiducial_points(num_fiducial)
        self.register_buffer("target_control_points", ctrl)
        fc = self.loc_net[-1]
        nn.init.zeros_(fc.weight)
        fc.bias.data.copy_(ctrl.view(-1))

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        delta = self.loc_net(x).view(B, self.num_fiducial, 2) * 0.1
        dst_pts = self.target_control_points.unsqueeze(0).expand(B, -1, -1)
        src_pts = dst_pts + delta
        kernel_w, affine_w = get_tps_transform(src_pts, dst_pts)
        return warp_image_tps(x, src_pts, kernel_w, affine_w, align_corners=True)


class TRBA(nn.Module):
    """
    TRBA OCR model with ResNet backbone and optional TPS STN
    """

    def __init__(
        self,
        img_height: int,
        img_width: int,
        num_classes: int,
        transform: Optional[str] = None,
    ):
        super().__init__()
        # Spatial transformer
        self.stn = TPS_STN(20, img_height, img_width) if transform == "tps" else None
        # ResNet backbone from FAN paper

        self.cnn = ResNet(1, 512, BasicBlock, [1, 2, 5, 3])
        cnn_out = 512
        # Pool height
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        # LSTM sequence modeling
        self.rnn = nn.LSTM(
            cnn_out, 256, 2, bidirectional=True, dropout=0.3, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.3)
        # Classifier
        self.embedding = nn.Linear(512, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        if self.stn is not None:
            x = self.stn(x)
        f = self.cnn(x)
        f = self.adaptive_pool(f)
        B, C, _, Wp = f.shape
        f = f.view(B, C, Wp).permute(0, 2, 1)
        r, _ = self.rnn(f)
        r = self.layer_norm(r)
        r = self.dropout(r)
        o = self.embedding(r)
        return o.permute(1, 0, 2)
