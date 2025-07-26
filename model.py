import torch
import torch.nn as nn
from kornia.geometry.transform import get_tps_transform, warp_image_tps
from torch import Tensor
from typing import Optional, List, Tuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    # unchanged TPS_STN code
    def __init__(self, num_fiducial: int, H: int, W: int):
        super().__init__()
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
        delta = self.loc_net(x).view(B, self.loc_net[-1].out_features // 2, 2) * 0.1
        dst = self.target_control_points.unsqueeze(0).expand(B, -1, -1)
        src = dst + delta
        kw, aw = get_tps_transform(src, dst)
        return warp_image_tps(x, src, kw, aw, align_corners=True)


# Attention modules (Bahdanau additive attention)
class AttentionCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        embed_size: int = 128,
        att_dropout: float = 0.1,
    ):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)
        self.embed = nn.Embedding(num_classes, embed_size)
        self.att_dropout = nn.Dropout(att_dropout)
        self.rnn = nn.LSTMCell(input_size + embed_size, hidden_size)

    def forward(
        self,
        batch_H: torch.Tensor,  # [B, S, F]
        prev_state: tuple,  # (h, c)
        prev_ids: torch.LongTensor,  # [B]
        mask: Optional[torch.BoolTensor],  # [B, S]
    ):
        device = batch_H.device
        proj_H = self.i2h(batch_H)  # [B, S, H]
        proj_prev = self.h2h(prev_state[0]).unsqueeze(1)  # [B, 1, H]

        # аддитивное внимание
        score_in = torch.tanh(proj_H + proj_prev)  # [B, S, H]
        score_in = self.att_dropout(score_in)
        e = self.score(score_in)  # [B, S, 1]
        if mask is not None:
            e = e.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(e, dim=1)  # [B, S, 1]
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # [B, F]

        ch_emb = self.embed(prev_ids.to(device))  # [B, embed_size]
        inp = torch.cat([context, ch_emb], dim=1)  # [B, F+embed_size]
        return self.rnn(inp, prev_state), alpha


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        embed_size: int = 128,
        att_dropout: float = 0.1,
        max_length: int = 100,
        sos_idx: int = 0,
        eos_idx: int = None,
        pad_idx: int = None,
    ):
        super().__init__()
        self.cell = AttentionCell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            embed_size=embed_size,
            att_dropout=att_dropout,
        )
        self.generator = nn.Linear(hidden_size, num_classes)
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx if eos_idx is not None else num_classes - 2
        self.pad_idx = pad_idx if pad_idx is not None else num_classes - 1
        self.max_length = max_length

    def forward(
        self,
        batch_H: torch.Tensor,  # [B, S, F]
        text: torch.LongTensor,  # [B, T] (с SOS и PAD)
        mask: Optional[torch.BoolTensor],  # [B, S]
        is_train: bool = True,
    ):
        B, T = text.size()
        h = torch.zeros(B, self.cell.rnn.hidden_size, device=batch_H.device)
        c = torch.zeros(B, self.cell.rnn.hidden_size, device=batch_H.device)
        outputs = []

        if is_train:
            for t in range(1, T):
                prev_ids = text[:, t - 1]
                (h, c), _ = self.cell(batch_H, (h, c), prev_ids, mask)
                logits = self.generator(h)  # [B, num_classes]
                outputs.append(logits)
            outputs = torch.stack(outputs, dim=1)  # [B, T-1, C]
            targets = text[:, 1:]  # [B, T-1]
            return outputs, targets

        # Inference (greedy)
        # инференс: генерируем до EOS или пока не исчерпаем max_length
        generated = []
        prev_ids = torch.full(
            (B,), self.sos_idx, dtype=torch.long, device=batch_H.device
        )
        for _ in range(self.max_length):
            (h, c), _ = self.cell(batch_H, (h, c), prev_ids, mask)
            logits = self.generator(h)
            next_ids = logits.argmax(dim=1)
            generated.append(next_ids)
            prev_ids = next_ids
            # если все батчевые элементы уже предсказали EOS — выходим
            if (next_ids == self.eos_idx).all():
                break
        # сконвертируем в тензор и допадим до единой длины
        outputs = torch.stack(generated, dim=1)  # [B, L<=max_length]
        if outputs.size(1) < self.max_length:
            pad = outputs.new_full((B, self.max_length - outputs.size(1)), self.pad_idx)
            outputs = torch.cat([outputs, pad], dim=1)
        return outputs


class TRBA(nn.Module):
    def __init__(
        self,
        alphabet: list[str],  # Новый аргумент
        img_height: int,
        img_width: int,
        transform: Optional[str] = None,
        use_attention: bool = False,
        att_hidden_size: int = 256,
        att_embed_size: int = 128,
        att_dropout: float = 0.1,
        att_max_length: Optional[int] = None,
    ):
        super().__init__()
        # определяем токены
        self.alphabet = alphabet
        num_chars = len(alphabet)
        self.SOS_IDX = 0
        self.CHAR_START = 1
        self.EOS_IDX = self.CHAR_START + num_chars
        self.PAD_IDX = self.EOS_IDX + 1
        num_classes = self.PAD_IDX + 1

        self.stn = TPS_STN(20, img_height, img_width) if transform == "tps" else None
        self.cnn = ResNet(1, 512, BasicBlock, [1, 2, 5, 3])  # без изменений
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(
            512, 256, 2, bidirectional=True, batch_first=True, dropout=0.3
        )
        self.layer_norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.3)

        self.use_attention = use_attention
        if use_attention:
            self.att_decoder = AttentionDecoder(
                input_size=512,
                hidden_size=att_hidden_size,
                num_classes=num_classes,
                embed_size=att_embed_size,
                att_dropout=att_dropout,
                max_length=att_max_length or img_width // 4,
                sos_idx=self.SOS_IDX,
                eos_idx=self.EOS_IDX,
                pad_idx=self.PAD_IDX,
            )
        else:
            self.fc = nn.Linear(512, num_classes)

    def forward(
        self,
        x: Tensor,
        text: Optional[Tensor] = None,
        is_train: bool = True,
    ):
        # 1) Сначала STN (если есть)
        if self.stn is not None:
            x = self.stn(x)

        # 2) CNN → адаптивный пул → преобразуем в [B, S, F] для RNN
        f = self.cnn(x)
        f = self.adaptive_pool(f)  # [B, C, 1, Wp]
        B, C, _, Wp = f.shape
        feat = f.view(B, C, Wp).permute(0, 2, 1)  # [B, S, F]
        rnn_out, _ = self.rnn(feat)  # [B, S, 2*H]
        rnn_out = self.layer_norm(rnn_out)
        rnn_out = self.dropout(rnn_out)

        # 3) Ветка с attention
        if self.use_attention:
            assert text is not None, "`text` должен быть передан в режиме attention"
            # создаём маску из единиц (никаких блокировок)
            att_mask = torch.ones(
                B, rnn_out.size(1), device=rnn_out.device, dtype=torch.bool
            )

            if is_train:
                # на обучении получаем (логиты, targets)
                outputs, targets = self.att_decoder(
                    rnn_out, text, att_mask, is_train=True
                )
                return outputs, targets
            else:
                # на инференсе получаем только логиты
                outputs = self.att_decoder(rnn_out, text, att_mask, is_train=False)
                return outputs

        # 4) А если без attention — CTC‐ветка
        logits = self.fc(rnn_out)  # [B, S, C]
        return logits.permute(1, 0, 2)  # [S, B, C]


def build_attention_inputs(
    seqs: List[torch.Tensor],
    sos_idx: int,
    eos_idx: int,
    pad_idx: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds text_input and targets for attention:
    - text_input: [B, max_len+2] with SOS at 0, EOS after sequence, PAD elsewhere
    - targets: [B, max_len+1] from first char to EOS inclusive
    """
    B = len(seqs)
    max_L = max(s.size(0) for s in seqs)
    text_input = torch.full((B, max_L + 2), pad_idx, dtype=torch.long, device=device)
    text_input[:, 0] = sos_idx
    for i, s in enumerate(seqs):
        L = s.size(0)
        text_input[i, 1 : 1 + L] = s.to(device)
        text_input[i, 1 + L] = eos_idx
    targets = text_input[:, 1 : 1 + max_L + 1]
    return text_input, targets


def decode_attention(
    seqs: List[torch.Tensor],
    pred_idxs: torch.Tensor,
    alphabet: List[str],
    sos_idx: int,
    pad_idx: int,
) -> Tuple[List[str], List[str]]:
    """
    Decode attention outputs and build hypotheses and references.
    Removes SOS, PAD and repeats.
    """
    hyps, refs = [], []
    for seq, pred in zip(seqs, pred_idxs.tolist()):
        # hypothesis
        hyp_chars = []
        prev = None
        for idx in pred:
            if idx in (sos_idx, pad_idx) or idx == prev:
                prev = idx
                continue
            # adjust for alphabet offset
            if 1 <= idx <= len(alphabet):
                hyp_chars.append(alphabet[idx - 1])
            prev = idx
        hyps.append("".join(hyp_chars))
        # reference
        true_idxs = seq.tolist()
        ref_chars = [alphabet[j - 1] for j in true_idxs if j > 0]
        refs.append("".join(ref_chars))
    return hyps, refs
