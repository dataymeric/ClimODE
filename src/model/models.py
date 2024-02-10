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
        self.time_pos_embedding = time_pos_embedding.view(-1, 38, 32, 64)
        self.nb_var_time_dep = config["nb_variable_time_dependant"]

    def forward(
        self,
        t,
        x,
    ):
        # ic(x.shape, self.time_pos_embedding.shape)  # [8, 5, 32, 64]
        original_x = x
        x = torch.cat([x, self.time_pos_embedding[t]], dim=-3)
        x = self.model(x)
        # ic(
        #     x.shape, # [8, 10, 32, 64]
        #     x[:, : self.nb_var_time_dep].shape, original_x.shape
        # )
        # From original code, idk what he is doing here
        mean = original_x + x[:, : self.nb_var_time_dep]
        std = F.softmax(x[:, self.nb_var_time_dep :], dim=-3)
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
        attention_beta = attention_beta.view(1, -1, 32, 64).contiguous()
        # ic(self.post_map(attention_beta).shape) # (1, 10, 32, 64)
        return self.post_map(attention_beta)


class VelocityModel(nn.Module):
    """
    Equivalent of $f_\theta$ in the paper.
    """

    def __init__(self, config, time_pos_embedding):
        super().__init__()
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
        )  # input original code ([1, 64, 32, 64])
        self.gamma = nn.Parameter(torch.tensor([sub_config["gamma"]]))

    def forward(self, t, x):
        """
        OK
        Input must directly have all the parameters concatenated.
        x: shape: (15,32,64) -> (10, 32,64,10) + (5,32,64)
        """

        # Obligé de cat avant puis uncat ici car odeint ne peut pas split ces param je pense
        # pour le coup un tensors dict ici serait plus propre mais plus le temps
        # Si on passe en (batch, timestep, année, ...,...), il faudra rajouter un :
        past_velocity = x[:10]  # v in original code
        past_velocity_x = past_velocity[:5]
        past_velocity_y = past_velocity[5:]
        past_velocity_grad_x = torch.gradient(past_velocity_x, dim=-2)[0]
        past_velocity_grad_y = torch.gradient(past_velocity_y, dim=-3)[0]

        x_0 = x[10:, :, :]  # ds in original code
        x_0_grad_x = torch.gradient(x_0, dim=-1)[0]  # sur la dim de la logitude (64)
        x_0_grad_y = torch.gradient(x_0, dim=-2)[0]  # sur la dim de la latitude (32)
        nabla_u = torch.cat([x_0_grad_x, x_0_grad_y], dim=-3)  # (2*5,32,64)

        t_emb = t.view(1, 1, 1).expand(1, 32, 64)
        t = int(t.item()) * 100

        x = torch.cat([t_emb, x, nabla_u, self.time_pos_embedding[t]], dim=-3)
        # Unsquueze for simulate a batch of 1
        x = x.unsqueeze(0)
        # ic(
        #     x.shape, # [1, 64, 32, 64]
        #     nabla_u.shape, # [10, 32, 64]
        #     self.time_pos_embedding[t].view(-1, 32, 64).shape, # [38, 32, 64]
        #     past_velocity.shape, # [10, 32, 64]
        #     x_0.shape, # [5, 32, 64]
        # )
        dv = self.local_model(x)
        dv += self.gamma * self.global_model(x)
        dv = dv.squeeze().view(-1, 32, 64)  # (10, 32, 64)

        adv1 = past_velocity_x * x_0_grad_x + past_velocity_y * x_0_grad_y
        adv2 = x_0 * (past_velocity_grad_x + past_velocity_grad_y)

        # ic(
        #   v.shape, # [10, 32, 64]
        #   adv1.shape, # [5, 32, 64]
        #   adv2.shape, # [5, 32, 64]
        # )
        dvs = torch.cat([dv, adv1 + adv2], dim=0)
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
        OK
        """

        # Calcul of news timesteps
        init_time = t[0].item() * self.freq
        final_time = t[-1].item() * self.freq
        steps_val = final_time - init_time
        ode_t = 0.01 * torch.linspace(
            init_time, final_time, steps=int(steps_val) + 1
        ).to(self.device)  # Je sais pas pourquoi 0.01

        # Solvings ODE
        x = odeint(self.velocity_model, x, ode_t, method="euler")
        # ic(x.shape) # torch.Size([43, 15, 32, 64])
        x = x[
            :, -5:
        ]  # On récupère que les données de la prédiction uniquement, pas des past velocities si je comprends bien ???
        # Nan je sais pas
        # ic(x.shape) # [43, 5, 32, 64]
        x = x[::6]  # idk pourquoi on fait ça, je crois qu'on rediscretise en 8 morceaux
        # ic(x.shape) # ([8, 5, 32, 64])
        mean, std = self.emission_model(t, x)
        return mean, std
