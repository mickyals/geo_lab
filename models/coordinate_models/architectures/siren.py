# Complete SIREN network class, connects layers from core/layers.py

import torch
import torch.nn as nn
from ..core.layers import BaseLayer, ResidualBaseBlock
from ..core.embeddings import get_embedding


class SIREN(nn.Module):
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

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.initializer = initializer
        self.activation = activation
        self.residual_weight = residual_weight
        self.residual = residual
        self.use_embedding = use_embedding
        self.embedding_type = embedding_type
        self.embedding_kwargs = embedding_kwargs if embedding_kwargs is not None else {}

        # Initialize the embedding layer
        if self.use_embedding:
            self.embedding = get_embedding(self.embedding_type, **self.embedding_kwargs)
            in_dim = self.embedding.out_features
        else:
            in_dim = self.in_features

        layers = []

        # first layer with SIREN Initalization
        layers.append(BaseLayer(in_features=in_dim,
                                out_features=hidden_features,
                                initializer=self.initializer,
                                activation=self.activation,
                                activation_kwargs={'omega': omega},
                                initializer_kwargs={'in_features': in_dim,
                                                    'is_first': True,
                                                    'omega': omega}
                                ))

        # hidden layers
        for _ in range(1, num_layers - 1):
            if self.residual:
                layers.append(ResidualBaseBlock(in_features=hidden_features,
                                                out_features=hidden_features,
                                                initializer=self.initializer,
                                                activation=self.activation,
                                                activation_kwargs={'omega': omega},
                                                initializer_kwargs={'in_features': hidden_features,
                                                                    'is_first': False,
                                                                    'omega': omega},
                                                residual_weight=self.residual_weight,
                                                ))
            else:
                layers.append(BaseLayer(in_features=hidden_features,
                                        out_features=hidden_features,
                                        initializer=self.initializer,
                                        activation=self.activation,
                                        activation_kwargs={'omega': omega},
                                        initializer_kwargs={'in_features': hidden_features,
                                                            'is_first': False,
                                                            'omega': omega}
                                        ))

        # last layer
        layers.append(BaseLayer(in_features=hidden_features,
                                out_features=out_features,
                                initializer=self.initializer,
                                activation=self.activation,
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


