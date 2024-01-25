import torch.nn as nn
import torch.nn.functional as F


def NormLayer(dim, config):
    norm_type = config["model"]["norm_type"]
    if norm_type == "batch":
        return nn.BatchNorm2d(dim)
    elif norm_type == "group":
        return nn.GroupNorm(min(32, dim), dim)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(dim)
    elif norm_type == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown or incompatible norm type {norm_type}")


def ConvLayer(dim_in, dim_out, config):
    return nn.Conv2d(
        dim_in,
        dim_out,
        kernel_size=config["model"]["kernel_size"],
        stride=config["model"]["stride"],
    )


class ResidualBlock(nn.Module):
    """Pre-activation Residual Block. https://arxiv.org/pdf/1603.05027.pdf"""

    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.activation = nn.LeakyReLU(0.3)
        self.conv1 = ConvLayer(dim_in, dim_out, config)
        self.norm1 = NormLayer(dim_out, config)
        self.conv2 = ConvLayer(dim_out, dim_out, config)
        self.norm2 = NormLayer(dim_out, config)
        self.shortcut = (
            nn.Identity() if dim_in == dim_out else nn.Conv2d(dim_in, dim_out, 1)
        )

    def padding(self, x):
        x = F.pad(x, (0, 0, 1, 1), "reflect")  # reflect padding on Y
        x = F.pad(x, (1, 1, 0, 0), "circular")  # circular padding on X
        return x

    def forward(self, x):
        out = self.activation(x)

        # First convolutional layer
        out = self.padding(out)
        out = self.conv1(out)
        out = self.norm1(out)
    
        out = self.activation(out)

        # Second convolutional layer
        out = self.padding(out)
        out = self.conv2(out)
        out = self.norm2(out)

        # Shortcut connection
        out = out + self.shortcut(x)
        return out
