import torch
from torch import nn
import math






class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, omega, is_first, is_last):

        super().__init__()
        self.is_last = is_last
        self.in_features = in_features
        self.out_features = out_features
        self.omega = omega
        self.is_first = is_first
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.init_weights()

    def forward(self, x):
        wx_b = self.linear(x)
        if not self.is_last:
            out = torch.sin(self.omega * wx_b)
            return out
        return wx_b

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega
                self.linear.weight.uniform_(-bound, bound)



class SirenNet(nn.Module):
    def __init__(self,
                 N_in_features,
                 N_out_features,
                 N_hidden_features,
                 N_hidden_layers,
                 first_omega=30,
                 hidden_omega=30):

        super().__init__()

        self.net = self._build_network(N_in_features,N_out_features,
                                         N_hidden_features,N_hidden_layers,
                                         first_omega, hidden_omega)

    def _build_network(self, N_in_features,N_out_features,
                         N_hidden_features,N_hidden_layers,
                         first_omega, hidden_omega):
        net = nn.Sequential()

        first_layer = SirenLayer(N_in_features, N_hidden_features, omega=first_omega, is_first=True, is_last=False)
        net.add_module('first_layer', first_layer)

        for i in range(N_hidden_layers):
            hidden_layer = SirenLayer(N_hidden_features, N_hidden_features, omega=hidden_omega, is_first=False, is_last=False)
            net.add_module(f'hidden_layer_{i+1}', hidden_layer)

        last_layer = SirenLayer(N_hidden_features, N_out_features, omega=hidden_omega, is_first=False, is_last=True)
        net.add_module('last_layer', last_layer)

        return net


    def forward(self, x):
        return self.net(x)