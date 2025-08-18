# Base layer types: Linear, FourierLayer, SIRENLayer
import torch
import torch.nn as nn
from ..core.initializations import get_initializer
from ..core.activations import get_activation



class BaseLayer(nn.Module):
    def __init__(self, in_features, out_features, initializer, activation, is_last=False, initializer_kwargs=None, activation_kwargs=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.is_last = is_last
        self.initializer_kwargs = initializer_kwargs if initializer_kwargs is not None else {}
        self.initializer = get_initializer(initializer,  **self.initializer_kwargs)
        self.activation_kwargs = activation_kwargs if activation_kwargs is not None else {}
        self.activation = get_activation(activation, **self.activation_kwargs)
        self.initializer(self.linear)

    def forward(self, x):
        if self.is_last:
            return self.linear(x)
        return self.activation(self.linear(x))


class ResidualBaseBlock(nn.Module):
    def __init__(self, in_features, out_features, initializer, activation, residual_weight=None, initializer_kwargs=None, activation_kwargs=None):
        super().__init__()
        self.activation = get_activation(activation, **activation_kwargs)
        self.residual_weight = residual_weight

        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)

        # Only create a projection if needed
        if in_features != out_features:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = nn.Identity()

        # Initialize
        init_fn = get_initializer(initializer, **initializer_kwargs)
        init_fn(self.linear1)
        init_fn(self.linear2)
        if not isinstance(self.residual, nn.Identity):
            init_fn(self.residual)

    def forward(self, x):
        main_path = self.activation(self.linear1(x))
        main_path = self.activation(self.linear2(main_path))
        res = self.residual(x)

        if self.residual_weight is None:
            return main_path + res
        else:
            return (1 - self.residual_weight) * main_path + self.residual_weight * res


