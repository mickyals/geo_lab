import torch
from torch import nn
from geolab.models.components.position_encoders import FourierFeatures



class FCNLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 activation='relu', is_last=False, bias=True, init_type='uniform'):
        super().__init__()


        self.layer = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights(init_type)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Tanh()


        self.is_last = is_last


    def init_weights(self, init_type):
        with torch.no_grad():
            if init_type == 'uniform':
                nn.init.xavier_uniform_(self.layer.weight)
            elif init_type == 'normal':
                nn.init.xavier_normal_(self.layer.weight)
            else:
                raise ValueError(f"Invalid init_type: {init_type}")
            if self.layer.bias is not None:
                nn.init.zeros_(self.layer.bias)


    def forward(self, x):
        wx_b = self.layer(x)
        if not self.is_last:
            out = self.activation(wx_b)
            return out
        return wx_b


class FCN(nn.Module):
    def __init__(self, N_in_features, N_out_features, N_hidden_features, N_hidden_layers,
                 activation='relu', bias=True, init_type='uniform',
                 position_encoder_type=None, mapping_dim=None, scale=1.0,
                 encode_dims=None):
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
                encode_dims=encode_dims,  # Receives pre-computed indices
                trainable=False
            )

            # Update input features for the network
            network_input_features = self.position_encoder.output_dimension

            print(f"Position encoder created:")
            print(f"  Type: {position_encoder_type}")
            print(f"  Encoding dimensions: {encode_dims}")
            print(f"  Passthrough dimensions: {self.position_encoder.passthrough_dims}")
            print(f"  Output dimension: {network_input_features}")
        else:
            self.position_encoder = None
            network_input_features = N_in_features

        self.net = self._build_network(
            network_input_features, N_out_features, N_hidden_features,
            N_hidden_layers, activation, bias, init_type
        )


    def _build_network(self, in_features, out_features, hidden_features, hidden_layers,
                       activation, bias, init_type):

        net = nn.Sequential()
        first_layer = FCNLayer(in_features, hidden_features, activation, is_last=False, bias=bias, init_type=init_type)
        net.add_module('first_layer', first_layer)

        for i in range(hidden_layers):
            hidden_layer = FCNLayer(hidden_features, hidden_features, activation, is_last=False, bias=bias, init_type=init_type)
            net.add_module(f'hidden_layer_{i+1}', hidden_layer)

        last_layer = FCNLayer(hidden_features, out_features, activation, is_last=True, bias=bias, init_type=init_type)
        net.add_module('last_layer', last_layer)

        return net

    def forward(self, x):
        if self.position_encoder is not None:
            x = self.position_encoder(x)
        return self.net(x)