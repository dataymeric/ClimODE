import torch
from model.conv import ClimateResNet2D
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint


class EmissionModel(nn.Module):
    """
    Equivalent of noise_net_contrib() using a class format.
    """

    def __init__(self, config, time_pos_embedding):
        super().__init__()
        self.sub_config = config["model"]["EmissionModel"]
        self.model = ClimateResNet2D(
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


class AttentionModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden_channels = in_channels // 2
        self.query = self.make_layer(
            in_channels, in_channels // 8, hidden_channels, stride=1, padding=True
        )
        self.key = self.make_layer(
            in_channels, in_channels // 8, hidden_channels, stride=2
        )
        self.value = self.make_layer(
            in_channels, out_channels, hidden_channels, stride=2
        )
        self.post_map = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))

    @staticmethod
    def make_layer(in_channels, out_channels, hidden_channels, stride, padding=False):
        def get_block(in_channels, out_channels, stride, padding):
            if padding:
                block = [
                    nn.ReflectionPad2d((0, 0, 1, 1)),
                    nn.CircularPad2d((1, 1, 0, 0)),
                ]
            else:
                block = []

            block += [
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride),
                nn.LeakyReLU(0.3),
            ]
            return block

        return nn.Sequential(
            *get_block(in_channels, hidden_channels, stride, padding),
            *get_block(hidden_channels, out_channels, stride, padding),
            *get_block(out_channels, out_channels, 1, padding),
        )

    def forward(self, x):
        """OK

        Parameters
        ----------
        x : _type_
            shape code origin: (1, 64, 32, 64)

        Returns
        -------
        _type_
            _description_
        """
        # On flatten sur la latitude et la longitude
        q = self.query(x).flatten(-2, -1)  # (1, 64, 32, 64) -> (1, 64, 2048)
        k = self.key(x).flatten(-2, -1)  # (1, 8, 3, 13) -> (1, 8, 65)
        v = self.value(x).flatten(-2, -1)  # (1, 10, 3, 13) -> (1, 10, 65)
        # ic(q.shape, k.shape, v.shape)
        # Il doit y avoir moyen de mieux faire, le contiguous est salle je pense
        attention_beta = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=1)
        attention_beta = torch.bmm(v, attention_beta.transpose(1, 2))
        attention_beta = attention_beta.view(1, -1, 32, 64).contiguous()
        # ic(self.post_map(attention_beta).shape) # (1, 10, 32, 64)
        return self.post_map(attention_beta)
        """
        size = x.size()
        x = x.float()
        q, k, v = (
            self.query(x).flatten(-2, -1),
            self.key(x).flatten(-2, -1),
            self.value(x).flatten(-2, -1),
        )
        beta = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=1)
        o = torch.bmm(v, beta.transpose(1, 2))
        o = self.post_map(o.view(-1, self.out_ch, size[-2], size[-1]).contiguous())
        return o
        """


class VelocityModel(nn.Module):
    """
    Equivalent of $f_\theta$ in the paper.
    """

    def __init__(self, config, time_pos_embedding):
        super().__init__()
        sub_config = config["model"]["VelocityModel"]
        self.time_pos_embedding = time_pos_embedding
        self.local_model = ClimateResNet2D(
            sub_config["local"]["in_channels"],
            sub_config["local"]["layers_length"],
            sub_config["local"]["layers_hidden_size"],
            config,
        )
        self.global_model = AttentionModel(
            sub_config["global"]["in_channels"],
            sub_config["global"]["out_channels"],
        )  # input original code ([1, 64, 32, 64])
        self.gamma = nn.Parameter(torch.tensor([sub_config["gamma"]]))

    def forward(self, t, x):
        """
        OK tested
        Input must directly have all the parameters concatenated.
        x: shape: (32, 64, 15) -> (32, 64, 10) + (32, 64, 5)
        """

        # Obligé de cat avant puis uncat ici car odeint ne peut pas split ces param je pense
        # pour le coup un tensors dict ici serait plus propre mais plus le temps
        # Si on passe en (batch, timestep, année, ...,...), il faudra rajouter un :
        past_velocity = x[:, :, :10]  # v in original code
        past_velocity_x = past_velocity[:, :, :5]
        past_velocity_y = past_velocity[:, :, 5:]
        past_velocity_grad_x = torch.gradient(past_velocity_x, dim=-2)[0]
        past_velocity_grad_y = torch.gradient(past_velocity_y, dim=-3)[0]

        x_0 = x[:, :, 10:]  # ds in original code
        x_0_grad_x = torch.gradient(x_0, dim=-2)[0]  # sur la dim de la logitude (64)
        x_0_grad_y = torch.gradient(x_0, dim=-3)[0]  # sur la dim de la latitude (32)
        nabla_u = torch.cat([x_0_grad_x, x_0_grad_y], dim=-1)  # (32,64,2*5)

        t_emb = t.view(1, 1, 1).expand(32, 64, 1)
        t = int(t.item()) * 100
        # ic(
        #     x.shape,
        #     nabla_u.shape,
        #     self.time_pos_embedding[t].shape,
        #     past_velocity.shape,
        #     x_0.shape,
        # )

        x = torch.cat([t_emb, x, nabla_u, self.time_pos_embedding[t]], dim=-1)
        # Unsquueze for simulate a batch of 1
        # and inverting the last dimension to the match CNN style conv (sorry j'aurai pu le faire avant j'ai merdé tant pis TODO)
        x = x.view(1, 64, 32, 64)
        dv = self.local_model(x)
        dv += self.gamma * self.global_model(x)
        dv = dv.squeeze().view(32, 64, -1)  # (32,64,10)

        adv1 = past_velocity_x * x_0_grad_x + past_velocity_y * x_0_grad_y
        adv2 = x_0 * (past_velocity_grad_x + past_velocity_grad_y)

        # ic(dv.shape, adv1.shape, adv2.shape)
        dvs = torch.cat([dv, adv1 + adv2], dim=-1)
        return dvs


class ClimODE(nn.Module):
    def __init__(self, config, time_pos_embedding):
        super(ClimODE, self).__init__()
        self.config = config
        self.device = config["device"]
        self.freq = config["freq"]
        self.velocity_model = VelocityModel(config, time_pos_embedding)
        self.emission_model = EmissionModel(config, time_pos_embedding)
        self.time_pos_embedding = time_pos_embedding

    def forward(self, t, x):
        """
        WIP, not tested yet.

        # TODO: Add the initial condition as input
        """

        # Calcul of news timesteps
        init_time = t[0].item() * self.freq
        final_time = t[-1].item() * self.freq
        steps_val = final_time - init_time
        ode_t = (1 / 100) * torch.linspace(
            init_time, final_time, steps=int(steps_val) + 1
        ).to(
            self.device
        )  # Je sais pas pourquoi 0.01

        # Solvings ODE
        x = odeint(self.velocity_model, x, t, method="euler")
        mean, std = self.emission_model(t, x)
        return mean, std
