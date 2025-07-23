import torch
import torch.nn as nn
from kornia.geometry.transform import get_tps_transform, warp_image_tps
from torch import Tensor
from typing import Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


def build_fiducial_points(num_fiducial: int) -> torch.Tensor:
    assert num_fiducial % 2 == 0, "num_fiducial must be even"
    half = num_fiducial // 2
    xs = torch.linspace(-1.0, 1.0, steps=half)
    top = torch.stack([xs, torch.full_like(xs, -1.0)], dim=1)
    bot = torch.stack([xs, torch.full_like(xs, +1.0)], dim=1)
    return torch.cat([top, bot], dim=0)


class ResNet(nn.Module):
    # unchanged stage definitions
    def __init__(self, input_channel: int, output_channel: int, block, layers: list[int]):
        super().__init__()
        self.output_channel_block = [
            output_channel // 4,
            output_channel // 2,
            output_channel,
            output_channel,
        ]
        self.inplanes = output_channel // 8
        # initial conv layers
        self.conv0_1 = nn.Conv2d(input_channel, output_channel // 16, 3, 1, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(output_channel // 16)
        self.conv0_2 = nn.Conv2d(output_channel // 16, self.inplanes, 3, 1, 1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # stage1
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])
        # stage2
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[1], 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])
        # stage3
        self.maxpool3 = nn.MaxPool2d((2, 1), (2, 1), padding=(0, 0))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[2], 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])
        # stage4
        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3], 2, (2, 1), (0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3], 2, 1, 0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
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
    # unchanged TPS_STN code
    def __init__(self, num_fiducial: int, H: int, W: int):
        super().__init__()
        feat_h, feat_w = H // 8, W // 8
        self.loc_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((feat_h, feat_w)), nn.Flatten(),
            nn.Linear(128 * feat_h * feat_w, 256), nn.ReLU(),
            nn.Linear(256, num_fiducial * 2),
        )
        ctrl = build_fiducial_points(num_fiducial)
        self.register_buffer("target_control_points", ctrl)
        fc = self.loc_net[-1]
        nn.init.zeros_(fc.weight)
        fc.bias.data.copy_(ctrl.view(-1))

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        delta = self.loc_net(x).view(B, self.loc_net[-1].out_features // 2, 2) * 0.1
        dst = self.target_control_points.unsqueeze(0).expand(B, -1, -1)
        src = dst + delta
        kw, aw = get_tps_transform(src, dst)
        return warp_image_tps(x, src, kw, aw, align_corners=True)


# Attention modules (Bahdanau additive attention)
class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # batch_H: [B, S, F]
        proj_H = self.i2h(batch_H)
        proj_prev = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(proj_H + proj_prev))  # [B, S, 1]
        alpha = torch.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)
        inp = torch.cat([context, char_onehots], dim=1)
        return self.rnn(inp, prev_hidden), alpha


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.cell = AttentionCell(input_size, hidden_size, num_classes)
        self.gen = nn.Linear(hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes

    def _onehot(self, ids):
        B = ids.size(0)
        oh = torch.zeros(B, self.num_classes, device=device)
        return oh.scatter_(1, ids.unsqueeze(1), 1)

    def forward(self, batch_H, text, is_train=True):
        B, S, F = batch_H.size()
        T = text.size(1)
        outputs = torch.zeros(B, T, self.hidden_size, device=device)
        hidden = (
            torch.zeros(B, self.hidden_size, device=device),
            torch.zeros(B, self.hidden_size, device=device)
        )
        if is_train:
            for i in range(T):
                ch = self._onehot(text[:, i])
                hidden, _ = self.cell(hidden, batch_H, ch)
                outputs[:, i, :] = hidden[0]
            return self.gen(outputs)
        # inference
        preds = torch.zeros(B, T, self.num_classes, device=device)
        prev = torch.zeros(B, dtype=torch.long, device=device)
        for i in range(T):
            ch = self._onehot(prev)
            hidden, _ = self.cell(hidden, batch_H, ch)
            logits = self.gen(hidden[0])
            preds[:, i, :] = logits
            _, prev = logits.max(1)
        return preds


class TRBA(nn.Module):
    """
    TRBA OCR model with ResNet backbone, optional TPS STN and attention decoder
    """

    def __init__(
        self,
        img_height: int,
        img_width: int,
        num_classes: int,
        transform: Optional[str] = None,
        use_attention: bool = False,
        att_hidden_size: int = 256,
        att_max_length: Optional[int] = None,
    ):
        super().__init__()
        self.stn = TPS_STN(20, img_height, img_width) if transform == "tps" else None
        self.cnn = ResNet(1, 512, BasicBlock, [1, 2, 5, 3])
        cnn_out = 512
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(cnn_out, 256, 2, bidirectional=True, batch_first=True, dropout=0.3)
        self.layer_norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.3)
        self.use_attention = use_attention
        self.num_classes = num_classes
        self.att_max_length = att_max_length
        if use_attention:
            self.att = Attention(input_size=512, hidden_size=att_hidden_size, num_classes=num_classes)
        else:
            self.fc = nn.Linear(512, num_classes)

    def forward(
        self,
        x: Tensor,
        text: Optional[Tensor] = None,
        is_train: bool = True,
    ) -> Tensor:
        if self.stn is not None:
            x = self.stn(x)
        f = self.cnn(x)
        f = self.adaptive_pool(f)
        B, C, _, Wp = f.shape
        feat = f.view(B, C, Wp).permute(0, 2, 1)  # [B, S, F]
        rnn_out, _ = self.rnn(feat)
        rnn_out = self.dropout(self.layer_norm(rnn_out))
        if self.use_attention:
            assert text is not None, "`text` must be provided for attention mode"
            logits = self.att(rnn_out, text, is_train)
            # logits: [B, T, C] -> [T, B, C]
            return logits.permute(1, 0, 2)
        # CTC path
        logits = self.fc(rnn_out)  # [B, S, C]
        return logits.permute(1, 0, 2)  # [S, B, C]
