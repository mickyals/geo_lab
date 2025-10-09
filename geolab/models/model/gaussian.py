import torch
from torch import nn



class GaussianLayer(nn.Module):
    def __init__(self, in_features, out_features, sigma, bias, is_last, init_type):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma
        self.is_last = is_last
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.init_weights(init_type)

    def init_weights(self, init_type):
        with torch.no_grad():
            if init_type == 'uniform':
                nn.init.xavier_uniform_(self.linear.weight)
            elif init_type == 'normal':
                nn.init.xavier_normal_(self.linear.weight)
            else:
                raise ValueError(f"Invalid initialization type: {init_type}")

    def forward(self, x):
        wx_b = self.linear(x)
        if not self.is_last:
            out = torch.exp(-(self.sigma * wx_b)**2)
            return out
        return wx_b



class GaussianNet(nn.Module):
    def __init__(self, N_in_features, N_out_features,
                 N_hidden_features, N_hidden_layers,
                 sigma, init_type):
        super().__init__()

        self.net = self._build_network(N_in_features, N_out_features,
                                     N_hidden_features, N_hidden_layers, sigma, init_type)


    def _build_network(self, N_in_features, N_out_features,
                        N_hidden_features, N_hidden_layers, sigma, init_type):
        net = nn.Sequential()

        first_layer = GaussianLayer(in_features=N_in_features, out_features=N_hidden_features,
                                    sigma=sigma, bias=True, is_last=False, init_type=init_type)
        net.add_module('first_layer', first_layer)

        for i in range(N_hidden_layers):
            hidden_layer = GaussianLayer(in_features=N_hidden_features, out_features=N_hidden_features,
                                         sigma=sigma, bias=True, is_last=False, init_type=init_type)
            net.add_module(f'hidden_layer_{i+1}', hidden_layer)

        last_layer = GaussianLayer(in_features=N_hidden_features, out_features=N_out_features,
                                   sigma=sigma, bias=True, is_last=True, init_type=init_type)
        net.add_module('last_layer', last_layer)

        return net

    def forward(self, x):
        return self.net(x)