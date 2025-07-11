# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from kornia.geometry.transform import get_tps_transform, warp_image_tps
from vocab import SPECIAL_TOKENS


class CNNEncoder(nn.Module):
    def __init__(self, backbone="resnet", pretrained=True):
        super().__init__()
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
            self.cnn = nn.Sequential(*list(net.children())[:-2])  # remove avgpool/fc
            self.out_channels = 512
        else:
            raise NotImplementedError("Only resnet supported in this version")

    def forward(self, x):  # x: [B,1,H,W]
        f = self.cnn(x)     # [B,C,H',W']
        f = torch.mean(f, dim=2)  # global average over height → [B,C,W']
        f = f.permute(2, 0, 1)    # → [W',B,C]
        return f
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [T, 1, D]
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: [T, B, D]
        return x + self.pe[:x.size(0)]


class TransformerOCRModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        backbone: str = "resnet",
    ):
        super().__init__()
        self.encoder_cnn = CNNEncoder(backbone=backbone)
        self.d_model = d_model

        self.encoder_proj = nn.Linear(self.encoder_cnn.out_channels, d_model)
        self.decoder_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, imgs, tgt_inp):
        # imgs: [B, 1, H, W]
        # tgt_inp: [B, T] (token IDs with <sos>)

        memory = self.encoder_cnn(imgs)  # [S, B, C]
        memory = self.encoder_proj(memory)  # [S, B, D]
        memory = self.pos_encoder(memory)

        tgt = self.decoder_embed(tgt_inp).permute(1, 0, 2)  # [T, B, D]
        tgt = self.pos_decoder(tgt)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0), tgt.device)  # [T,T]

        out = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
        )  # [T, B, D]
        out = self.output_layer(out)  # [T, B, vocab]
        return out


def build_fiducial_points(num_fiducial: int) -> torch.Tensor:
    assert num_fiducial % 2 == 0, "num_fiducial должно быть чётным"
    half = num_fiducial // 2
    xs = torch.linspace(-1.0, 1.0, steps=half)
    top = torch.stack([xs, torch.full_like(xs, -1.0)], dim=1)
    bot = torch.stack([xs, torch.full_like(xs, +1.0)], dim=1)
    return torch.cat([top, bot], dim=0)  # [K,2]


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
        # инициализация bias под identity
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


class VGG_FeatureExtractor(nn.Module):
    """VGG-style из оригинального CRNN (Shi et al. 2016)"""

    def __init__(self, in_ch=1, out_ch=512):
        super().__init__()
        oc = [out_ch // 8, out_ch // 4, out_ch // 2, out_ch]  # [64,128,256,512]
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
    CRNN с опциями:
      - backbone: 'resnet' (ResNet-18) или 'vgg'
      - transform: 'none'|'affine'|'tps'
    """

    def __init__(
        self,
        img_height: int,
        img_width: int,
        num_classes: int,
        backbone: str = "resnet",
        pretrained: bool = True,
        transform: str = "none",
    ):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width

        # Spatial Transformer
        if transform == "affine":
            self.stn = AffineSTN()
        elif transform == "tps":
            self.stn = TPS_STN(20, img_height, img_width)
        else:
            self.stn = None

        # Выбираем фиче-экстрактор
        if backbone == "resnet":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=weights)
            # первый conv → 1-канал, stride=(1,1)
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
            # убираем агрессивный пул
            net.maxpool = nn.Identity()
            self.cnn = nn.Sequential(*list(net.children())[:-2])
            cnn_out = 512

        elif backbone == "vgg":
            self.cnn = VGG_FeatureExtractor(in_ch=1, out_ch=512)
            cnn_out = 512

        else:
            raise ValueError(f"Unsupported backbone '{backbone}'")

        # Свёртки → AdaptiveAvgPool2d((1,None)) оставляет только размер по ширине W'
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        # RNN-блок
        self.rnn = nn.LSTM(
            input_size=cnn_out,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(256 * 2)
        self.dropout_rnn = nn.Dropout(0.3)

        # классификатор
        self.embedding = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        # x: [B,1,H,W]
        if self.stn is not None:
            x = self.stn(x)
        f = self.cnn(x)  # [B,C,H',W']
        f = self.adaptive_pool(f)  # [B,C,1,W']
        B, C, _, Wp = f.shape
        f = f.view(B, C, Wp).permute(0, 2, 1)  # [B, W', C]
        r, _ = self.rnn(f)  # [B, W', 512]
        r = self.layer_norm(r)
        r = self.dropout_rnn(r)
        o = self.embedding(r)  # [B, W', classes]
        return o.permute(1, 0, 2)  # [W',B,classes]
