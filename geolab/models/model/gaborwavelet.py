import torch
from torch import nn
import math
from geolab.models.components.position_encoders import FourierFeatures


class RealGaborLayer(nn.Module):
    def __init__(self, in_features, out_features, bias, omega_0, scale_0, is_first, init_type):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        self.scale_0 = scale_0
        self.is_first = is_first
        self.freqs = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.scale = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.init_freqs_weights()
        self.init_scale_weights(init_type)

    def init_freqs_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.freqs.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega_0
                self.freqs.weight.uniform_(-bound, bound)

    def init_scale_weights(self, init_type):
        with torch.no_grad():
            if init_type == 'uniform':
                nn.init.xavier_uniform_(self.scale.weight)
            elif init_type == 'normal':
                nn.init.xavier_normal_(self.scale.weight)
            else:
                raise ValueError(f"Invalid initialization type: {init_type}")

    def forward(self, x):
        omega_wx_b = self.omega_0 * self.freqs(x)
        scale_wx_b = self.scale_0 * self.scale(x)
        out = torch.cos(omega_wx_b)*torch.exp(-scale_wx_b**2)
        return out


class RealWireNet(nn.Module):
    def __init__(self, N_in_features, N_out_features, N_hidden_features,
                 N_hidden_layers, omega_0, scale_0, init_type, bias,
                 position_encoder_type=None, mapping_dim=None, scale=1.0):

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

        self.net = self._build_network(network_input_features, N_out_features, N_hidden_features,
                                       N_hidden_layers, omega_0, scale_0, init_type, bias)


    def _build_network(self, N_in_features, N_out_features, N_hidden_features,
                       N_hidden_layers, omega_0, scale_0, init_type, bias):

        net = nn.Sequential()

        first_layer = RealGaborLayer(N_in_features, N_hidden_features, bias=bias,
                                     omega_0=omega_0, scale_0=scale_0, is_first=True, init_type=init_type)
        net.add_module('first_layer', first_layer)

        for i in range(N_hidden_layers):
            hidden_layer = RealGaborLayer(N_hidden_features, N_hidden_features, bias=bias,
                                          omega_0=omega_0, scale_0=scale_0, is_first=False, init_type=init_type)
            net.add_module(f'hidden_layer_{i+1}', hidden_layer)

        last_layer = nn.Linear(N_hidden_features, N_out_features)
        nn.init.xavier_uniform_(last_layer.weight)

        net.add_module('last_layer', last_layer)

        return net

    def forward(self, x):
        if self.position_encoder is not None:
            x = self.position_encoder(x)
        return self.net(x)



class ComplexGaborLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0, scale_0, is_first, is_last,bias,trainable_omega, trainable_scale):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first
        self.is_last = is_last

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        self.omega_0 = nn.Parameter(omega_0 * torch.ones(1), requires_grad=trainable_omega)
        self.scale_0 = nn.Parameter(scale_0 * torch.ones(1), requires_grad=trainable_scale)

        self.layer = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.layer.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega_0
                self.layer.weight.uniform_(-bound, bound)

    def forward(self, x):
        wx_b = self.layer(x)
        if not self.is_last:
            spectral = 1j * self.omega_0 * wx_b
            spatial = -torch.abs(self.scale_0 * wx_b)**2
            out = torch.exp(spectral) * torch.exp(spatial)
            return out
        return wx_b.real


class ComplexWireNet(nn.Module):
    #TO DO wire still confuses me
    pass


