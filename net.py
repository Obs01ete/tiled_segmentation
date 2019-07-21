import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size-1)//2,
        )
        self._bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._bn(x)
        x = F.relu(x)
        return x


class SegmentationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (16, 32, 16)
        self._conv1 = Conv(3, channels[0], 3)
        self._conv2 = Conv(channels[0], channels[1], 3)
        self._pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self._conv3 = Conv(channels[1], channels[2], 3)
        self._upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self._conv4 = nn.Conv2d(
            in_channels=channels[2],
            out_channels=2,
            kernel_size=5,
            padding=(5-1)//2,
        )

        self._seg_loss = nn.modules.loss.CrossEntropyLoss()

    def forward(self, x):
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._pool(x)
        x = self._conv3(x)
        x = self._upsample(x)
        x = self._conv4(x)
        return x

    def loss(self, pred_tensor, gt_tensor):
        segmentation_loss = self._seg_loss(
            pred_tensor,
            gt_tensor
        )
        return segmentation_loss

    def decode(self, pred_tensor, threshold=0.5):
        sm = F.softmax(pred_tensor, dim=1)
        return (sm[:, 0, :, :] > threshold).long()
