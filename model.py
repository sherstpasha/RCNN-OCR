import torch
import torch.nn as nn
from resnet31 import ResNet31


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x: [B, T, input_size]
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(x)  # [B, T, 2*hidden_size]
        output = self.linear(recurrent)  # [B, T, output_size]
        return output


class RCNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super().__init__()

        self.cnn = ResNet31(in_channels=3, out_channels=512)
        self.pool = nn.AdaptiveAvgPool2d((1, None))

        self.rnn = nn.Sequential(
            BidirectionalLSTM(self.cnn.out_channels, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes),
        )

    def forward(self, x):
        # CNN features
        f = self.cnn(x)  # [B, C, H, W]
        f = self.pool(f)  # [B, C, 1, W]
        f = f.squeeze(2).permute(0, 2, 1)  # [B, W, C]

        # RNN features
        out = self.rnn(f)  # [B, W, num_classes]

        # For CTC: [T, B, C]
        return out.permute(1, 0, 2)
