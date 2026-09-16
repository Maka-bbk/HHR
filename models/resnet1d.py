# -*- coding: utf-8 -*-

"""
1D-ResNet backbone for Happy-CGCD × USC-HAD.

Input:
    x: [B, C, T]

Example:
    [B, 6, 256]

Output:
    feat: [B, feat_dim]

Example:
    [B, 256]

This backbone is used to replace image ViT for HAR time-series data.
"""

from typing import List, Optional

import torch
import torch.nn as nn


def conv1d_3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    dilation: int = 1,
) -> nn.Conv1d:
    """
    1D version of 3x3 convolution.

    For time series:
        kernel_size = 3
        padding = dilation

    This keeps the temporal length roughly unchanged when stride=1.
    """

    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


class BasicBlock1D(nn.Module):
    """
    Basic residual block for 1D time-series data.

    Input:
        x: [B, in_channels, T]

    Output:
        out: [B, out_channels, T'].

    If stride=1 and channel number does not change:
        residual path is identity.

    If stride>1 or channel number changes:
        residual path uses 1x1 Conv1d for downsampling.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv1 = conv1d_3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv1d_3x3(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=1,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if dropout > 0:
            self.dropout = nn.Dropout(p=float(dropout))
        else:
            self.dropout = nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """
    1D-ResNet backbone for HAR sensor windows.

    Input:
        x: [B, C, T]

    Default USC-HAD V0:
        x: [B, 6, 256]

    Output:
        feat: [B, feat_dim]

    Default:
        feat: [B, 256]
    """

    def __init__(
        self,
        in_channels: int = 6,
        feat_dim: int = 256,
        base_channels: int = 64,
        layers: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        if layers is None:
            layers = [2, 2, 2]

        self.in_channels = int(in_channels)
        self.feat_dim = int(feat_dim)
        self.base_channels = int(base_channels)
        self.layers = layers
        self.dropout_rate = float(dropout)

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.base_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(self.base_channels),
            nn.ReLU(inplace=True),
        )

        self.current_channels = self.base_channels

        self.layer1 = self._make_layer(
            out_channels=self.base_channels,
            num_blocks=layers[0],
            stride=1,
            dropout=self.dropout_rate,
        )

        self.layer2 = self._make_layer(
            out_channels=self.base_channels * 2,
            num_blocks=layers[1],
            stride=2,
            dropout=self.dropout_rate,
        )

        self.layer3 = self._make_layer(
            out_channels=self.base_channels * 4,
            num_blocks=layers[2],
            stride=2,
            dropout=self.dropout_rate,
        )

        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.proj = nn.Linear(
            in_features=self.base_channels * 4,
            out_features=self.feat_dim,
        )

        self._init_weights()

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
        dropout: float,
    ) -> nn.Sequential:
        blocks = []

        blocks.append(
            BasicBlock1D(
                in_channels=self.current_channels,
                out_channels=out_channels,
                stride=stride,
                dropout=dropout,
            )
        )

        self.current_channels = out_channels

        for _ in range(1, num_blocks):
            blocks.append(
                BasicBlock1D(
                    in_channels=self.current_channels,
                    out_channels=out_channels,
                    stride=1,
                    dropout=dropout,
                )
            )

        return nn.Sequential(*blocks)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature before projection head.

        Input:
            x: [B, C, T]

        Output:
            feat: [B, feat_dim]
        """

        if x.ndim != 3:
            raise ValueError(
                f"ResNet1D expects input shape [B, C, T], but got {tuple(x.shape)}"
            )

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)

        feat = self.proj(x)

        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def build_resnet1d(
    in_channels: int = 6,
    feat_dim: int = 256,
    base_channels: int = 64,
    dropout: float = 0.0,
) -> ResNet1D:
    """
    Helper function for building ResNet1D.

    This function is optional.
    train_happy.py can directly use ResNet1D(...).
    """

    model = ResNet1D(
        in_channels=in_channels,
        feat_dim=feat_dim,
        base_channels=base_channels,
        layers=[2, 2, 2],
        dropout=dropout,
    )

    return model


if __name__ == "__main__":
    x = torch.randn(4, 6, 256)
    model = ResNet1D(in_channels=6, feat_dim=256)
    feat = model(x)

    print("input shape:", x.shape)
    print("feature shape:", feat.shape)