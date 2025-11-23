import torch
from torch import nn
import math


# from geolab.models.components.position_encoders import FourierFeatures


class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, omega, is_first, is_last):

        super().__init__()
        self.is_last = is_last
        self.in_features = in_features
        self.out_features = out_features
        self.omega = omega
        self.is_first = is_first
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        self.init_weights(in_features=in_features)

    def forward(self, x):
        wx_b = self.linear(x)
        if not self.is_last:
            out = torch.sin(self.omega * wx_b)
            return out
        return wx_b

    def init_weights(self, in_features: int):
        """Initialization recommended by SIREN paper."""
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                bound = math.sqrt(6 / in_features) / self.omega
                self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)  # Required for symmetry


class SirenNet(nn.Module):
    def __init__(self,
                 N_in_features,
                 N_out_features,
                 N_hidden_features,
                 N_hidden_layers,
                 first_omega=30,
                 hidden_omega=30,
                 position_encoder_type=None,
                 mapping_dim=None,
                 scale=1.0):

        super().__init__()

        self.position_encoder_type = position_encoder_type

        # Create position encoder if specified
        if position_encoder_type is not None:
            if mapping_dim is None:
                raise ValueError("mapping_dim must be specified when using position encoder")
            self.position_encoder = FourierFeatures(
                input_dimension=N_in_features,
                mapping_dimension=mapping_dim,
                scale=scale,
                type=position_encoder_type,
                trainable=False
            )
            # Update input features for the network
            network_input_features = mapping_dim
        else:
            self.position_encoder = None
            network_input_features = N_in_features

        self.net = self._build_network(network_input_features, N_out_features,
                                       N_hidden_features, N_hidden_layers,
                                       first_omega, hidden_omega)

    def _build_network(self, N_in_features, N_out_features,
                       N_hidden_features, N_hidden_layers,
                       first_omega, hidden_omega):
        net = nn.Sequential()

        first_layer = SirenLayer(N_in_features, N_hidden_features, omega=first_omega, is_first=True, is_last=False)
        net.add_module('first_layer', first_layer)

        for i in range(N_hidden_layers):
            hidden_layer = SirenLayer(N_hidden_features, N_hidden_features, omega=hidden_omega, is_first=False,
                                      is_last=False)
            net.add_module(f'hidden_layer_{i + 1}', hidden_layer)

        last_layer = SirenLayer(N_hidden_features, N_out_features, omega=hidden_omega, is_first=False, is_last=True)
        net.add_module('last_layer', last_layer)

        return net

    def forward(self, x):
        if self.position_encoder is not None:
            x = self.position_encoder(x)
        return self.net(x)