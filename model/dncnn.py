# import torch
import torch.nn as nn

# import torch.nn.functional as F
# from base import BaseModel


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17, activation="relu", last_activ=False):
        super().__init__()

        if activation == "relu":
            activation_layer = nn.ReLU(inplace=True)
        elif activation == "leaky":
            activation_layer = nn.LeakyReLU(inplace=True)
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=channels,
                out_channels=features,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        layers.append(activation_layer)
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(features))
            layers.append(activation_layer)
        layers.append(
            nn.Conv2d(
                in_channels=features,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out

