import torch
from icecream import ic
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint_adjoint as odeint

from model.conv import ClimateResNet2D


class EmissionModel(nn.Module):
    """
    Equivalent of noise_net_contrib() using a class format.
    """

    def __init__(self, config, time_pos_embedding):
        super().__init__()
        self.config = config
        self.sub_config = config["model"]["EmissionModel"]
        self.model = ClimateResNet2D(
            self.sub_config["in_channels"],
            self.sub_config["layers_length"],
            self.sub_config["layers_hidden_size"],
            config,
        )
        self.time_pos_embedding = time_pos_embedding.view(-1, 38, 32, 64)
        self.nb_var_time_dep = config["nb_variable_time_dependant"]

    def forward(
        self,
        t,
        x,
    ):
        original_x = x

        # for each time step in t, we generate the next 8 time steps

        t = t.view(-1, 1).expand(-1, self.config['pred_length'])
        t = t + torch.arange(0, self.config['pred_length'], device=t.device).view(1,
                                                                                  -1).flatten()

        tpe = self.time_pos_embedding[t]
        tpe = tpe.view(
            self.config['bs'],
            self.config['pred_length'],
            -1,
            *tpe.shape[-2:]
        ).to(x.device)

        x = torch.cat([x, tpe], dim=-3)
        x = x.view(-1, *x.shape[2:])
        x = self.model(x).view(
            self.config['bs'],
            self.config['pred_length'],
            -1,
            *original_x.shape[-2:]
        )

        mean = original_x + x[:, :, : self.nb_var_time_dep]
        std = F.softmax(x[:, :, self.nb_var_time_dep:], dim=-3)

        return mean, std


class AttentionModel(nn.Module):
    def __init__(self, in_channels, out_channels, bs):
        super().__init__()
        self.bs = bs
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
        """
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
        attention_beta = attention_beta.view(self.bs, -1, 32, 64)

        output = self.post_map(attention_beta)
        return output


class VelocityModel(nn.Module):
    """
    Equivalent of $f_\theta$ in the paper.
    """

    def __init__(self, config, time_pos_embedding):
        super().__init__()
        self.config = config
        sub_config = config["model"]["VelocityModel"]
        self.time_pos_embedding = time_pos_embedding.view(-1, 38, 32, 64)
        self.local_model = ClimateResNet2D(
            sub_config["local"]["in_channels"],
            sub_config["local"]["layers_length"],
            sub_config["local"]["layers_hidden_size"],
            config,
        )
        self.global_model = AttentionModel(
            sub_config["global"]["in_channels"],
            sub_config["global"]["out_channels"],
            config["bs"],
        )  # input original code ([1, 64, 32, 64])
        self.gamma = nn.Parameter(torch.tensor([sub_config["gamma"]]))

    def update_time(self, t):
        self.t = t

    def forward(self, t, x):
        """
        OK
        Input must directly have all the parameters concatenated.
        x: shape: (12, 8, 5, 32, 64) et (12, 8, 5, 2, 32, 64)
        """

        (x_0, vel) = x

        past_velocity_x = vel[:, :, 0]
        past_velocity_y = vel[:, :, 1]
        past_velocity_grad_x = torch.gradient(past_velocity_x, dim=-1)[0]
        past_velocity_grad_y = torch.gradient(past_velocity_y, dim=-2)[0]

        x_0_grad_x = torch.gradient(x_0, dim=-1)[0]  # sur la dim de la logitude (64)
        x_0_grad_y = torch.gradient(x_0, dim=-2)[0]  # sur la dim de la latitude (32)
        nabla_u = torch.cat([x_0_grad_x, x_0_grad_y], dim=-3)  # (batch, 2*5,32,64)

        t = self.t

        # adapt following shape to genere
        # torch.Size([12, 1, 32, 64])
        t_emb = t.view(self.config["bs"], 1, 1, 1).expand(self.config["bs"], 1, 32, 64)
        t_emb = t_emb.to(x_0.device)

        # torch.Size([12, 38, 32, 64])
        time_pos_embedding = self.time_pos_embedding[t.to(torch.long)].to(x_0.device)

        # torch.Size([12, 64, 32, 64])
        vel = vel.view(self.config["bs"], -1, 32, 64)

        # torch.Size([12, 64, 32, 64])
        x = torch.cat([t_emb, x_0, vel, nabla_u, time_pos_embedding], dim=-3)

        dv = self.local_model(x)

        dv += self.gamma * self.global_model(x)

        adv1 = past_velocity_x * x_0_grad_x + past_velocity_y * x_0_grad_y
        adv2 = x_0 * (past_velocity_grad_x + past_velocity_grad_y)

        dvs = torch.cat([dv, adv1 + adv2], dim=1)

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

    def forward(self, data, vel, t):
        """
        OK
        """

        ode_t = 0.1 * torch.linspace(
            0, 8, steps=8
        ).to(self.device)

        x = (data, vel)

        # Solvings ODE
        self.velocity_model.update_time(t)

        ic(data.device, vel.device, ode_t.device)

        data, vel = odeint(self.velocity_model, x, ode_t, method="euler")

        # ode return the time as the first dimension,
        # we want the batch as the first dimension
        data = data.transpose(0, 1)
        vel = vel.transpose(0, 1)

        mean, std = self.emission_model(t, data)
        return mean, std
