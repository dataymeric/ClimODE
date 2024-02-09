import torch
from torch import nn
from conv import Climate_ResNet_2D

"""
WIP
"""


class Emission_model(nn.Module):
    """
    Equivalent of noise_net_contrib() using a class format.
    """

    def __init__(self, config, time_pos_embedding):
        super().__init__()
        self.sub_config = config["model"]["emission_model"]
        self.model = Climate_ResNet_2D(
            self.sub_config["in_channels"],
            self.sub_config["layers_length"],
            self.sub_config["layers_hidden_size"],
            config,
        )
        self.time_pos_embedding = time_pos_embedding

    def forward(
        self,
        t,
        x,
    ):
        """
        WIP, not tested yet.
        """
        # Dim ? Je connais pas la dim de x yet
        x = torch.cat([x, self.time_pos_embedding[t]], dim=1)
        x = self.model(x)
        # From original code, not sure if it's correct
        mean = x + x[:, :, : self.sub_config["out_types"]]
        std = nn.Softplus()(x[:, :, self.sub_config["out_types"] :])
        return mean, std
