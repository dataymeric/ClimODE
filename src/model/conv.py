import torch.nn as nn
import torch.nn.functional as F


class CustomRefPad2d(nn.Module):
    def __init__(self, padding):
        super(CustomRefPad2d, self).__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, self.padding, mode="reflect")


class CustomCircPad2d(nn.Module):
    def __init__(self, padding):
        super(CustomCircPad2d, self).__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, self.padding, mode="circular")


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
    """Pre-activation Residual Block. https://arxiv.org/pdf/1603.05027.pdf
    $f_{conv}$
    """

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
        self.padder_reflect = CustomRefPad2d((0, 0, 1, 1))
        self.padder_circular = CustomCircPad2d((1, 1, 0, 0))

    def forward(self, x):
        out = self.activation(x)

        # First convolutional layer
        out = self.padder_reflect(out)
        out = self.padder_circular(out)

        out = self.conv1(out)
        out = self.norm1(out)

        out = self.activation(out)

        # Second convolutional layer
        out = self.padder_reflect(out)
        out = self.padder_circular(out)
        out = self.conv2(out)
        out = self.norm2(out)

        # Shortcut connection
        out = out + self.shortcut(x)
        return out


class ClimateResNet2D(nn.Module):
    def __init__(self, in_channels, layers_length, layers_hidden_size, config):
        super().__init__()
        layers_cnn = []
        for length, hidden_size in zip(layers_length, layers_hidden_size):
            layers_cnn = layers_cnn + self.make_layer(
                in_channels, hidden_size, length, config
            )
            in_channels = hidden_size
        self.layer_cnn = nn.Sequential(*layers_cnn)

    @staticmethod
    def make_layer(in_channels, out_channels, reps, config):
        return [ResidualBlock(in_channels, out_channels, config)] + [
            ResidualBlock(out_channels, out_channels, config) for _ in range(reps - 1)
        ]

    def forward(self, x):
        x = x.float()
        x = self.layer_cnn(x)
        return x
