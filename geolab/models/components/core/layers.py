"""
building the basal layer for any MLP with interchangeable activation functions and initializations
also residual base layer/block code for any MLP with interchangeable activation functions and initializations
basal layer for weight parametrization for parameternet to supply weights too
"""

import torch
import torch.nn as nn
from geolab.models.components.core.initializations import get_initializer
from geolab.models.components.core.activations import get_activation



class BaseLayer(nn.Module):
    def __init__(self, in_features, out_features, activation,
                 initialization, is_last,
                 activation_kwargs, initialization_kwargs):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation = get_activation(activation, **(activation_kwargs if activation_kwargs else {}))
        self.initialization = get_initializer(initialization, **(initialization_kwargs if initialization_kwargs else {}))
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features)
        self.initialization(self.linear)

    def forward(self, x):
        if self.is_last:
            return self.linear(x)
        return self.activation(self.linear(x))



class ResidualBlock(BaseLayer):
    """
    a block is defined as the number of layer between x and f(x) + x

    """
    def __init__(self, num_features, residual_weight,
                 activation, initialization,  initialization_kwargs, activation_kwargs):
        super().__init__()

        self.activation = get_activation(activation, **(activation_kwargs if activation_kwargs else {}))
        self.initialization = get_initializer(initialization, **(initialization_kwargs if initialization_kwargs else {}))
        self.num_features = num_features
        self.linear1 = nn.Linear(num_features, num_features)
        self.linear2 = nn.Linear(num_features, num_features)
        self.residual_weight = residual_weight
        self.initialization(self.linear1)
        self.initialization(self.linear2)


    def forward(self, x):

        identity =  x
        first = self.linear1(x)
        act1 = self.activation(first)
        second = self.residual_weight * self.linear2(act1)
        if self.residual_weight is not None:
            residual = ((1 - self.residual_weight) * identity) + (self.residual_weight * second )
            return self.activation(residual)
        else:
            residual = identity + second
            return self.activation(residual)





class DenseBlock(BaseLayer):
    def __init__(self, num_features, num_layers, activation,
                 initialization, initialization_kwargs, activation_kwargs):
        pass

class BaseParametrizationLayer(BaseLayer):
    def __init__(self, in_features, out_features, weights_and_biases, activation,
                 initialization, initialization_kwargs, activation_kwargs):
        pass