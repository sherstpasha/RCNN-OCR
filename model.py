# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from kornia.geometry.transform import get_tps_transform, warp_image_tps


def build_fiducial_points(num_fiducial: int) -> torch.Tensor:
    """
    Генерирует num_fiducial точек, равномерно распределённых
    по двум линиям y=-1 (верх) и y=+1 (низ).
    Требуется, чтобы num_fiducial был чётным.
    """
    assert num_fiducial % 2 == 0, "num_fiducial должно быть чётным"
    half = num_fiducial // 2
    xs = torch.linspace(-1.0, 1.0, steps=half)
    top = torch.stack([xs, torch.full_like(xs, -1.0)], dim=1)
    bot = torch.stack([xs, torch.full_like(xs, +1.0)], dim=1)
    return torch.cat([top, bot], dim=0)  # shape = [num_fiducial, 2]


class AffineSTN(nn.Module):
    """Аффинный Spatial Transformer"""

    def __init__(self):
        super().__init__()
        self.loc_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
        )
        # инициализация под identity
        self.loc_net[-1].weight.data.zero_()
        self.loc_net[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.loc_net(x).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x


class TPS_STN(nn.Module):
    """Thin-Plate Spline Spatial Transformer"""

    def __init__(self, num_fiducial: int, img_height: int, img_width: int):
        super().__init__()
        self.num_fiducial = num_fiducial
        self.img_size = (img_height, img_width)
        feat_h, feat_w = img_height // 8, img_width // 8
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
        # Задаём target control points и инициализируем bias под identity
        ctrl = build_fiducial_points(num_fiducial)  # shape (K,2)
        self.register_buffer("target_control_points", ctrl)
        # последний Linear: (in=256, out=K*2)
        fc = self.loc_net[-1]
        nn.init.zeros_(fc.weight)  # обнуляем веса
        fc.bias.data.copy_(ctrl.view(-1))  # bias = [x1,y1,x2,y2,...]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1) Predict source control points (theta) [B, K, 2]
        2) target_control_points is [K,2], expand to [B,K,2]
        3) get_tps_transform(src, dst) -> (kernel_weights, affine_weights)
        4) warp_image_tps(image, src_pts, kernel_weights, affine_weights)
        """
        B, C, H, W = x.shape
        source_pts = self.loc_net(x).view(B, self.num_fiducial, 2)
        target_pts = self.target_control_points.unsqueeze(0).expand(B, -1, -1)
        # Получаем веса ядра и аффинные веса
        kernel_weights, affine_weights = get_tps_transform(source_pts, target_pts)
        # Применяем TPS‐преобразование
        x = warp_image_tps(
            x, source_pts, kernel_weights, affine_weights, align_corners=True
        )
        return x


class CRNN(nn.Module):
    """
    CRNN with optional STN/TPS transformer before the CNN backbone.
    Args:
        img_height: высота входного изображения
        img_width: максимальная ширина входного изображения
        num_classes: число классов + blank
        backbone: название CNN-бэкбона ('resnet50')
        pretrained: флаг для ImageNet-весов
        transform: 'none'|'affine'|'tps'
    """

    def __init__(
        self,
        img_height: int,
        img_width: int,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        transform: str = "none",
    ):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        # выбрать трансформер
        if transform == "affine":
            self.stn = AffineSTN()
        elif transform == "tps":
            self.stn = TPS_STN(
                num_fiducial=20, img_height=img_height, img_width=img_width
            )
        else:
            self.stn = None

        # CNN backbone
        if backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            cnn = models.resnet50(weights=weights)
            orig = cnn.conv1
            conv1 = nn.Conv2d(
                1,
                orig.out_channels,
                kernel_size=orig.kernel_size,
                stride=orig.stride,
                padding=orig.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    conv1.weight.data = orig.weight.data.sum(dim=1, keepdim=True)
            cnn.conv1 = conv1
            self.cnn = nn.Sequential(*list(cnn.children())[:-2])
            cnn_out = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        # ——————— Усиленный RNN-блок ———————
        self.rnn = nn.LSTM(
            input_size=cnn_out,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,  # dropout между слоями LSTM
            batch_first=True,
        )
        # LayerNorm по последнему измерению (2*hidden_size)
        self.layer_norm = nn.LayerNorm(256 * 2)
        # Дополнительный Dropout после LSTM (по времени)
        self.dropout_rnn = nn.Dropout(0.3)
        # —————————————
        self.embedding = nn.Linear(256 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stn is not None:
            x = self.stn(x)
        f = self.cnn(x)  # [B, C, H', W']
        f = self.adaptive_pool(f)  # [B, C, 1, W']
        f = f.squeeze(2)  # [B, C, W']
        f = f.permute(0, 2, 1)  # [B, W', C]
        r, _ = self.rnn(f)  # [B, W', 2*H]
        r = self.layer_norm(r)  # выравниваем дистрибуцию
        r = self.dropout_rnn(r)  # дополнительный dropout
        o = self.embedding(r)  # [B, W', num_classes]
        return o.permute(1, 0, 2)  # [W', B, num_classes]
