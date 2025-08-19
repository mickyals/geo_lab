# Complete SIREN network class, connects layers from core/layers.py

import torch
import torch.nn as nn
from ..core.layers import BaseLayer, ResidualBaseBlock
from ..core.embeddings import get_embedding


class SIREN(nn.Module):
    """
    Initializes a SIREN network.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features in each layer.
        out_features (int): Number of output features.
        num_layers (int): Number of layers in the network.
        omega (float, optional): Frequency parameter. Defaults to 30.
        initializer (str, optional): Initializer for the layers. Defaults to 'SIREN'.
        activation (str, optional): Activation function for the layers. Defaults to 'SINE'.
        residual_weight (float, optional): Weight for residual connections. Defaults to None.
        residual (bool, optional): Whether to use residual connections. Defaults to False.
        use_embedding (bool, optional): Whether to use an embedding layer. Defaults to False.
        embedding_type (str, optional): Type of embedding layer. Defaults to 'GAUSSIAN_POSITIONAL'.
        embedding_kwargs (dict, optional): Keyword arguments for the embedding layer. Defaults to None.
    """
    def __init__(self, in_features,
                 hidden_features,
                 out_features,
                 num_layers,
                 omega=30,
                 initializer='SIREN',
                 activation='SINE',
                 residual_weight=None,
                 residual=False,
                 use_embedding=False,
                 embedding_type='GAUSSIAN_POSITIONAL',
                 embedding_kwargs=None,
                 ):


        super().__init__()

        self.use_embedding = use_embedding
        embedding_kwargs = embedding_kwargs if embedding_kwargs is not None else {}

        # Initialize the embedding layer
        if use_embedding:
            self.embedding = get_embedding(embedding_type, **embedding_kwargs)
            in_dim = self.embedding.out_features
        else:
            in_dim = in_features

        layers = []

        # first layer with SIREN Initalization
        layers.append(BaseLayer(in_features=in_dim,
                                out_features=hidden_features,
                                initializer=initializer,
                                activation=activation,
                                activation_kwargs={'omega': omega},
                                initializer_kwargs={'in_features': in_dim,
                                                    'is_first': True,
                                                    'omega': omega}
                                ))

        # hidden layers
        for _ in range(1, num_layers - 1):
            if residual:
                layers.append(ResidualBaseBlock(in_features=hidden_features,
                                                out_features=hidden_features,
                                                initializer=initializer,
                                                activation=activation,
                                                activation_kwargs={'omega': omega},
                                                initializer_kwargs={'in_features': hidden_features,
                                                                    'is_first': False,
                                                                    'omega': omega},
                                                residual_weight=residual_weight,
                                                ))
            else:
                layers.append(BaseLayer(in_features=hidden_features,
                                        out_features=hidden_features,
                                        initializer=initializer,
                                        activation=activation,
                                        activation_kwargs={'omega': omega},
                                        initializer_kwargs={'in_features': hidden_features,
                                                            'is_first': False,
                                                            'omega': omega}
                                        ))

        # last layer
        layers.append(BaseLayer(in_features=hidden_features,
                                out_features=out_features,
                                initializer=initializer,
                                activation=activation,
                                is_last=True,
                                initializer_kwargs={'in_features': hidden_features,
                                                    'is_first': False,
                                                    'omega': omega}
                                ))

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        if self.use_embedding:
            x = self.embedding(x)
        return self.net(x)


SIREN_REGISTRY = {"siren": SIREN}