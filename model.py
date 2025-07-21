import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from kornia.geometry.transform import get_tps_transform, warp_image_tps


def build_fiducial_points(num_fiducial: int) -> torch.Tensor:
    assert num_fiducial % 2 == 0, "num_fiducial должно быть чётным"
    half = num_fiducial // 2
    xs = torch.linspace(-1.0, 1.0, steps=half)
    top = torch.stack([xs, torch.full_like(xs, -1.0)], dim=1)
    bot = torch.stack([xs, torch.full_like(xs, +1.0)], dim=1)
    return torch.cat([top, bot], dim=0)


class AffineSTN(nn.Module):
    """Аффинный Spatial Transformer"""

    def __init__(self):
        super().__init__()
        self.loc_net = nn.Sequential(
            nn.Conv2d(1, 8, 7),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 10, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
        )
        self.loc_net[-1].weight.data.zero_()
        self.loc_net[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x):
        theta = self.loc_net(x).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)


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

    def forward(self, x):
        B, C, H, W = x.shape
        delta = self.loc_net(x).view(B, self.num_fiducial, 2) * 0.1
        dst_pts = self.target_control_points.unsqueeze(0).expand(B, -1, -1)
        src_pts = dst_pts + delta  # src_pts ≈ dst_pts ± 0.1

        kernel_w, affine_w = get_tps_transform(src_pts, dst_pts)
        return warp_image_tps(x, src_pts, kernel_w, affine_w, align_corners=True)


class MSFResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # --- Scale A --- (без maxpool)
        resnet_a = models.resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        resnet_a.maxpool = nn.Identity()
        self.scaleA = nn.Sequential(*list(resnet_a.children())[:-2])

        # --- Scale B --- (обычная, с maxpool)
        resnet_b = models.resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.scaleB = nn.Sequential(*list(resnet_b.children())[:-2])

        # Преобразуем входные conv1, чтобы принять grayscale
        for net in [self.scaleA, self.scaleB]:
            conv1 = net[0]
            new_conv = nn.Conv2d(
                1,
                conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=(1, 1),
                padding=conv1.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    new_conv.weight[:] = conv1.weight.sum(dim=1, keepdim=True)
            net[0] = new_conv

    def freeze_layers(self, layer_names: list[str]):
        for lname in layer_names:
            module = self
            for part in lname.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break
            if isinstance(module, nn.Module):
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x):
        fA = self.scaleA(x)  # меньше downsampling
        fB = self.scaleB(x)  # обычный путь

        if fA.shape[2:] != fB.shape[2:]:
            fB = F.interpolate(
                fB, size=fA.shape[2:], mode="bilinear", align_corners=False
            )

        return fA + fB  # или torch.cat([fA, fB], dim=1) если нужно больше признаков


class VGG_FeatureExtractor(nn.Module):
    """VGG-style из оригинального CRNN (Shi et al. 2016)"""

    def __init__(self, in_ch=1, out_ch=512):
        super().__init__()
        oc = [out_ch // 8, out_ch // 4, out_ch // 2, out_ch]
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, oc[0], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(oc[0], oc[1], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(oc[1], oc[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(oc[2], oc[2], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(oc[2], oc[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(oc[3]),
            nn.ReLU(True),
            nn.Conv2d(oc[3], oc[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(oc[3]),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(oc[3], oc[3], 2, 1, 0),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


class CRNN(nn.Module):
    """
    CRNN с CTC + Attention:
      - backbone: 'resnet' или 'vgg'
      - transform: 'none'|'affine'|'tps'
    """

    def __init__(
        self,
        img_height: int,
        img_width: int,
        num_ctc_classes: int,
        num_attn_classes: int,
        backbone: str = "resnet",
        pretrained: bool = True,
        transform: str = "none",
        freeze_cnn_layers: list[str] = None,
    ):
        super().__init__()
        # STN
        if transform == "affine":
            self.stn = AffineSTN()
        elif transform == "tps":
            self.stn = TPS_STN(20, img_height, img_width)
        else:
            self.stn = None

        # CNN
        if backbone == "resnet":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=weights)
            orig = net.conv1
            net.conv1 = nn.Conv2d(
                1,
                orig.out_channels,
                kernel_size=orig.kernel_size,
                stride=(1, 1),
                padding=orig.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    net.conv1.weight[:] = orig.weight.sum(dim=1, keepdim=True)
            net.maxpool = nn.Identity()
            self.cnn = nn.Sequential(*list(net.children())[:-2])
            cnn_out = 512

        elif backbone == "msfresnet":
            self.cnn = MSFResNet(pretrained=pretrained)
            cnn_out = 512  # если используете add; если concat — поставьте 1024

        elif backbone == "vgg":
            self.cnn = VGG_FeatureExtractor(in_ch=1, out_ch=512)
            cnn_out = 512

        else:
            raise ValueError(f"Unsupported backbone '{backbone}'")

        if freeze_cnn_layers:
            if backbone == "msfresnet" and hasattr(self.cnn, "freeze_layers"):
                self.cnn.freeze_layers(freeze_cnn_layers)
            else:
                for name, module in self.cnn.named_children():
                    if name in freeze_cnn_layers:
                        for param in module.parameters():
                            param.requires_grad = False

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        # RNN-энкодер
        self.rnn = nn.LSTM(
            input_size=cnn_out,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3,
        )
        self.layer_norm = nn.LayerNorm(256 * 2)
        self.dropout = nn.Dropout(0.3)

        # CTC-голова
        self.ctc_fc = nn.Linear(256 * 2, num_ctc_classes)

        # Attention-декодер
        self.token_embedding = nn.Embedding(num_attn_classes, 256)
        self.decoder_rnn = nn.LSTM(256 + 256, 256, num_layers=2, batch_first=True)
        self.attn_q = nn.Linear(256, 256)
        self.attn_k = nn.Linear(256 * 2, 256)
        self.attn_v = nn.Linear(256 * 2, 256)
        self.attn_out = nn.Linear(256, 256)
        self.decoder_fc = nn.Linear(256, num_attn_classes)

    def freeze_layers(self, layer_names: list[str]):
        for lname in layer_names:
            if not hasattr(self, lname.split(".")[0]):
                continue
            module = self
            for part in lname.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break
            if isinstance(module, nn.Module):
                for p in module.parameters():
                    p.requires_grad = False

    def forward(self, x, decoder_inputs=None):
        if self.stn is not None:
            x = self.stn(x)
        f = self.cnn(x)
        f = self.adaptive_pool(f)
        B, C, _, Wp = f.shape
        f = f.view(B, C, Wp).permute(0, 2, 1)

        r, _ = self.rnn(f)
        r = self.layer_norm(r)
        r = self.dropout(r)

        # CTC
        ctc_logits = self.ctc_fc(r)
        ctc_out = ctc_logits.permute(1, 0, 2)

        # Attention
        attn_out = None
        if decoder_inputs is not None:
            B, L = decoder_inputs.size()
            emb = self.token_embedding(decoder_inputs)
            outputs = []
            K = self.attn_k(r)
            V = self.attn_v(r)
            h, c = None, None
            for t in range(L):
                q = self.attn_q(emb[:, t]).unsqueeze(1)
                scores = (q @ K.transpose(1, 2)) / (r.size(-1) ** 0.5)
                alpha = F.softmax(scores, dim=-1)
                ctx = alpha @ V
                din = torch.cat([emb[:, t].unsqueeze(1), ctx], dim=-1)
                # Инициализация скрытого состояния при первом вызове
                if h is None or c is None:
                    out, (h, c) = self.decoder_rnn(din)
                else:
                    out, (h, c) = self.decoder_rnn(din, (h, c))
                out = self.attn_out(out)
                logits = self.decoder_fc(out)
                outputs.append(logits)
            attn_out = torch.cat(outputs, dim=1)

        return ctc_out, attn_out
