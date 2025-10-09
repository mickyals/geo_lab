import torch
from torch import nn
import math



class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, omega, spread, is_first, is_last):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega = omega
        self.spread = spread
        self.is_first = is_first
        self.is_last = is_last
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        self.init_weights()
        self.init_biases()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega
                self.linear.weight.uniform_(-bound, bound)

    def init_biases(self):
        with torch.no_grad():
            self.linear.bias.uniform_(-self.spread, self.spread)


    def forward(self, x):
        wx_b = self.linear(x)
        if not self.is_last:
            alpha = torch.abs(wx_b) + 1
            out = torch.sin(self.omega * alpha * wx_b)
            return out
        return wx_b



class FinerNet(nn.Module):
    def __init__(self,
                 N_in_features,
                 N_out_features,
                 N_hidden_features,
                 N_hidden_layers,
                 first_omega,
                 hidden_omega,
                 spread):
        super().__init__()

        self.net = self._build_network(N_in_features,N_out_features,
                                     N_hidden_features,N_hidden_layers,
                                     first_omega, hidden_omega, spread)


    def _build_network(self, N_in_features,N_out_features,
                        N_hidden_features,N_hidden_layers,
                        first_omega, hidden_omega, spread):

        net = nn.Sequential()

        first_layer = FinerLayer(N_in_features, N_hidden_features, omega=first_omega,
                                 spread=spread, is_first=True, is_last=False)
        net.add_module('first_layer', first_layer)

        for i in range(N_hidden_layers):
            hidden_layer = FinerLayer(N_hidden_features, N_hidden_features, omega=hidden_omega,
                                      spread=spread, is_first=False, is_last=False)
            net.add_module(f'hidden_layer_{i+1}', hidden_layer)

        last_layer = FinerLayer(N_hidden_features, N_out_features, omega=hidden_omega,
                                spread=spread, is_first=False, is_last=True)
        net.add_module('last_layer', last_layer)

        return net

    def forward(self, x):
        return self.net(x)