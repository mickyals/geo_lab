import torch
import torch.nn as nn
from ..core.layers import BaseLayer, ResidualBaseBlock
from ..core.embeddings import get_embedding

class SpectralNetwork(nn.Module):
    def __init__(self,
                 r_half,
                 in_features,
                 hidden_features,
                 num_layers,
                 initializer,
                 activation,
                 initializer_kwargs=None,
                 activation_kwargs=None,
                 residual_weight=None,
                 residual=False,
                 use_embedding=False,
                 embedding_type='GAUSSIAN_POSITIONAL',
                 embedding_kwargs=None,
                 ):
        super().__init__()
        self.r_half = r_half
        self.use_embedding = use_embedding
        embedding_kwargs = embedding_kwargs if embedding_kwargs is not None else {}

        self.input = nn.Parameter(torch.randn(in_features))
        if use_embedding:
            self.embedding = get_embedding(embedding_type, **embedding_kwargs)
            in_dim = self.embedding.out_features
        else:
            in_dim = in_features

        layers = []
        # first layer allowing for custom initialization and activation
        layers.append(
            BaseLayer(
                in_features=in_dim,
                out_features=hidden_features,
                initializer=initializer,
                activation=activation,
                activation_kwargs=activation_kwargs,
                initializer_kwargs=initializer_kwargs,
        ))
        for _ in range(1, num_layers - 1):
            if residual:
                layers.append(
                    ResidualBaseBlock(
                        in_features = hidden_features,
                        out_features = hidden_features,
                        initializer = initializer,
                        activation = activation,
                        initializer_kwargs = initializer_kwargs,
                        activation_kwargs = activation_kwargs,
                        residual_weight = residual_weight
                    )
                )

            else:
                layers.append(
                    BaseLayer(
                        in_features = hidden_features,
                        out_features = hidden_features,
                        initializer = initializer,
                        activation = activation,
                        initializer_kwargs = initializer_kwargs,
                        activation_kwargs = activation_kwargs,
                    )
                )

        layers.append(
            BaseLayer(
                in_features = hidden_features,
                out_features = 2 * r_half,
                initializer = initializer,
                activation = activation,
                initializer_kwargs = initializer_kwargs,
                activation_kwargs = activation_kwargs,
                is_last = True
            )
        )

        self.net = nn.Sequential(*layers)


    def forward(self):
        if self.use_embedding:
            x = self.embedding(self.input)
        else:
            x = self.input
        out = self.net(x)

        raw_decay = out[..., :self.r_half]
        raw_freq = out[..., self.r_half:]

        decay = -2 * torch.sigmoid(raw_decay)
        freq = torch.sigmoid(raw_freq)

        return decay, freq


class InitialStateNetwork(nn.Module):
    def __init__(self,
                 r_half,
                 in_features,
                 hidden_features,
                 num_layers,
                 initializer,
                 activation,
                 initializer_kwargs=None,
                 activation_kwargs=None,
                 residual_weight=None,
                 residual=False,
                 use_embedding=False,
                 embedding_type='GAUSSIAN_POSITIONAL',
                 embedding_kwargs=None,
                 ):
        super().__init__()
        self.r_half = r_half
        self.use_embedding = use_embedding
        embedding_kwargs = embedding_kwargs if embedding_kwargs is not None else {}

        # latent input parameter
        self.input = nn.Parameter(torch.randn(in_features))

        if use_embedding:
            self.embedding = get_embedding(embedding_type, **embedding_kwargs)
            in_dim = self.embedding.out_features
        else:
            in_dim = in_features

        layers = []
        # first layer allowing for custom initialization and activation
        layers.append(
            BaseLayer(
                in_features=in_dim,
                out_features=hidden_features,
                initializer=initializer,
                activation=activation,
                activation_kwargs=activation_kwargs,
                initializer_kwargs=initializer_kwargs,
        ))
        for _ in range(1, num_layers - 1):
            if residual:
                layers.append(
                    ResidualBaseBlock(
                        in_features = hidden_features,
                        out_features = hidden_features,
                        initializer = initializer,
                        activation = activation,
                        initializer_kwargs = initializer_kwargs,
                        activation_kwargs = activation_kwargs,
                        residual_weight = residual_weight
                    )
                )

            else:
                layers.append(
                    BaseLayer(
                        in_features = hidden_features,
                        out_features = hidden_features,
                        initializer = initializer,
                        activation = activation,
                        initializer_kwargs = initializer_kwargs,
                        activation_kwargs = activation_kwargs
                    ))

        layers.append(
            BaseLayer(
                in_features = hidden_features,
                out_features = 1 +2 * r_half,
                initializer = initializer,
                activation = activation,
                initializer_kwargs = initializer_kwargs,
                activation_kwargs = activation_kwargs,
                is_last = True
            )
        )

        self.net = nn.Sequential(*layers)


    def forward(self):
        if self.use_embedding:
            x = self.embedding(self.input)
        else:
            x = self.input
        out = self.net(x)

        b0 = out[..., :1]
        raw = out[..., 1:].reshape(-1, self.r_half, 2)
        b_half = raw[..., 0] + 1j * raw[..., 1]

        return b0, b_half

class ModalNetwork(nn.Module):
    def __init__(self,
                 r_half,
                 in_features,
                 hidden_features,
                 num_layers,
                 initializer,
                 activation,
                 initializer_kwargs=None,
                 activation_kwargs=None,
                 residual_weight=None,
                 residual=False,
                 use_embedding=False,
                 embedding_type='GAUSSIAN_POSITIONAL',
                 embedding_kwargs=None,
                 ):
        super().__init__()
        self.use_embedding = use_embedding
        embedding_kwargs = embedding_kwargs if embedding_kwargs is not None else {}

        if use_embedding:
            self.embedding = get_embedding(embedding_type, **embedding_kwargs)
            in_dim = self.embedding.out_features
        else:
            in_dim = in_features

        layers = []
        # first layer allowing for custom initialization and activation
        layers.append(
            BaseLayer(
                in_features=in_dim,
                out_features=hidden_features,
                initializer=initializer,
                activation=activation,
                activation_kwargs=activation_kwargs,
                initializer_kwargs=initializer_kwargs,
        ))
        for _ in range(1, num_layers - 1):
            if residual:
                layers.append(
                    ResidualBaseBlock(
                        in_features = hidden_features,
                        out_features = hidden_features,
                        initializer = initializer,
                        activation = activation,
                        initializer_kwargs = initializer_kwargs,
                        activation_kwargs = activation_kwargs,
                        residual_weight = residual_weight
                    )
                )

            else:
                layers.append(
                    BaseLayer(
                        in_features = hidden_features,
                        out_features = hidden_features,
                        initializer = initializer,
                        activation = activation,
                        initializer_kwargs = initializer_kwargs,
                        activation_kwargs = activation_kwargs
                    ))

        # last layer
        layers.append(
            BaseLayer(
                in_features = hidden_features,
                out_features = 1 + r_half,
                initializer = initializer,
                activation = activation,
                initializer_kwargs = initializer_kwargs,
                activation_kwargs = activation_kwargs,
                is_last = True
            )
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_embedding:
            x = self.embedding(x)
        out = self.net(x)
        mode0 = out[..., :1]
        mode_half = out[..., 1:]

        return mode0, mode_half


class NeuralDMD(nn.Module):
    def __init__(self,
                 modal_kwargs,
                 spectral_kwargs,
                 initial_kwargs
                 ):
        super().__init__()
        self.modal = ModalNetwork(**modal_kwargs)
        self.spectral = SpectralNetwork(**spectral_kwargs)
        self.initial = InitialStateNetwork(**initial_kwargs)

    def forward(self, xyt):

        xy, t = xyt[..., :2], xyt[..., 2:]
        mode0, mode_half = self.modal(xy)
        decay, freq = self.spectral()
        b0, b_half = self.initial()

        omega = decay + 1j * freq

        base_term = mode0 * b0
        temporal_term = torch.exp(omega * t)

        interactions_term = mode_half * temporal_term * b_half

        reconstruction = base_term + 2 * interactions_term

        return reconstruction, {
            'mode0': mode0,
            'mode_half': mode_half,
            'b0': b0,
            'b_half': b_half,
            'decay': decay,
            'freq': freq
        }


DMD_REGISTRY = { "neuralDMD": NeuralDMD }