# positional encoders from fourier features to sphere2vec
import torch
import torch.nn as nn


#==============================================================================
# GaussianFourierEmbedding
#==============================================================================
class GaussianFourierEmbedding(nn.Module):
    def __init__(self, input_dim, mapping_dim, scale):
        """
        Initializes the Gaussian Fourier Embedding module. following the paper https://arxiv.org/abs/2006.10739

        Args:
            input_dim (int): The input dimension.
            mapping_dim (int): The mapping dimension.
            scale (float): The scale parameter.
        """
        super().__init__()

        # Input dimension
        self.input_dim = input_dim
        # Mapping dimension
        self.mapping_dim = mapping_dim
        # Scale parameter
        self.scale = scale
        # Beta parameter
        self.beta = torch.randn(input_dim, mapping_dim//2) * scale
        """
        Beta parameter. It is a tensor of shape (input_dim, mapping_dim//2)
        where the first dimension represents the input dimension and the second
        dimension represents the mapping dimension. The beta parameter is
        initialized with random values multiplied by the scale parameter.
        """

    def forward(self, x):
        """
        Apply the Gaussian Fourier Embedding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor with the Gaussian Fourier Embedding applied.

        The Gaussian Fourier Embedding embeds the input tensor in a higher dimensional space using a Fourier
        transform. The input tensor is multiplied by the beta parameter to obtain the transformation matrix.
        The cosine and sine of the transformed matrix are concatenated along the last dimension to obtain
        the final output tensor.
        """
        # Multiply the input tensor by the beta parameter to obtain the transformation matrix.
        x_transform = 2 * torch.pi * torch.matmul(x, self.beta)

        # Apply the cosine and sine functions to the transformed matrix.
        # The cosine and sine functions are concatenated along the last dimension to obtain the final output tensor.
        return torch.cat([torch.cos(x_transform), torch.sin(x_transform)], dim=-1)


#==============================================================================
# PositionalEmbedding
#==============================================================================

class PositionalEmbedding(nn.Module):
    def __init__(self, input_dim, mapping_dim, scale):
        """
        Initializes the Positional Embedding module.

        Args:
            input_dim (int): The input dimension.
            mapping_dim (int): The mapping dimension.
            scale (float): The scale parameter.

        Following the paper https://arxiv.org/abs/2006.10739
        """
        super().__init__()

        # Input dimension
        self.input_dim = input_dim
        # Mapping dimension
        self.mapping_dim = mapping_dim
        # Scale parameter
        self.scale = scale

        # Create a tensor of size mapping_dim, initialized with values from 0 to mapping_dim-1
        j = torch.arange(mapping_dim, dtype=torch.float32)

        # Calculate the beta parameter for each input dimension
        beta_row = scale ** (j/mapping_dim)

        # Register the beta parameter as a buffer tensor, with size (input_dim, mapping_dim)
        self.register_buffer('beta', beta_row.unsqueeze(0).repeat(input_dim, 1))

    def forward(self, x):
        """
        Apply the Positional Embedding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor with the Positional Embedding applied.

        The Positional Embedding embeds the input tensor in a higher dimensional space using a Fourier
        transform. The input tensor is multiplied by the beta parameter to obtain the transformation matrix.
        The cosine and sine of the transformed matrix are concatenated along the last dimension to obtain
        the final output tensor.
        """
        # Multiply the input tensor by the beta parameter to obtain the transformation matrix.
        x_transform = 2 * torch.pi * torch.matmul(x, self.beta)

        # Apply the cosine and sine functions to the transformed matrix.
        # The cosine and sine functions are concatenated along the last dimension to obtain the final output tensor.
        return torch.cat([
            torch.cos(x_transform),  # The cosine of the transformed matrix
            torch.sin(x_transform)   # The sine of the transformed matrix
        ], dim=-1)

#==============================================================================
# SPHERE2VEC
#==============================================================================



#==============================================================================
# SphericalGridEmbedding
#==============================================================================


class SphericalGridEmbedding(nn.Module):

    def __init__(self, scale, r_min, r_max=1.0):
        """
        Initializes the SphericalGridEmbedding module.

        Args:
            scale (int): The scale parameter.
            r_min (float): The minimum radius.
            r_max (float, optional): The maximum radius. Defaults to 1.0.

        Following the paper https://arxiv.org/pdf/2306.17624
        """
        super().__init__()

        # Store the input parameters
        self.scale = scale  # The scale parameter
        self.r_min = r_min  # The minimum radius
        self.r_max = r_max  # The maximum radius (default: 1.0)

        # Calculate the scaling factor for the grid
        self.g = r_max / r_min  # Growth factor for the beta parameter

        # Create a tensor of size scale, initialized with values from 0 to scale-1
        self.s = torch.arange(scale, dtype=torch.float32)  # Tensor of shape (scale,) representing the scale values
        """
        Tensor of shape (scale,) representing the scale values.
        The tensor is initialized with values from 0 to scale-1.
        """

        # Calculate the beta parameter for each coordinate (latitude and longitude)
        scale_minus_one = max(self.scale - 1, 1)  # Avoid division by zero
        beta_row = r_min * self.g ** (self.s / scale_minus_one)  # Tensor of shape (2, scale) representing the beta parameter
        """
        Tensor of shape (2, scale) representing the beta parameter.
        The first dimension corresponds to latitude, and the second dimension corresponds to longitude.
        """

        # Register the beta parameter as a buffer tensor
        self.register_buffer('beta', beta_row.unsqueeze(0).repeat(2, 1))  # Register the beta parameter as a buffer tensor
        """
        Register the beta parameter as a buffer tensor.
        The tensor is of size (2, scale) and is registered as a buffer tensor.
        The first dimension corresponds to latitude, and the second dimension corresponds to longitude.
        """


    def forward(self, x):
        """
        Calculate the embeddings for the given spherical coordinates.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, 2) containing the spherical coordinates.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, 4) containing the embeddings.

        Following the paper https://arxiv.org/pdf/2306.17624
        """
        # Extract latitude and longitude from the input tensor
        lat, lon = x[:, 0], x[:, 1]

        # Calculate the transformed values for latitude and longitude
        lat_transform = lat.unsqueeze(-1) * self.beta[0]
        lon_transform = lon.unsqueeze(-1) * self.beta[1]

        # Concatenate the sine and cosine values of the transformed coordinates
        return torch.cat([
            torch.sin(lat_transform),  # Sine of the transformed latitude
            torch.cos(lat_transform),  # Cosine of the transformed latitude
            torch.sin(lon_transform),  # Sine of the transformed longitude
            torch.cos(lon_transform),  # Cosine of the transformed longitude
        ], dim=-1)


#==============================================================================
# SphericalCartesianEmbedding
#==============================================================================

class SphericalCartesianEmbedding(nn.Module):
    def __init__(self, scale, r_min, r_max=1.0):
        """
        Initialize the SphericalCatesianEmbedding module.
        Following the paper https://arxiv.org/pdf/2306.17624

        Args:
            scale (int): The number of dimensions in the embeddings.
            r_min (float): The minimum radius of the sphere.
            r_max (float, optional): The maximum radius of the sphere. Defaults to 1.0.
        """
        super().__init__()

        # Following the paper https://arxiv.org/pdf/2306.17624

        # Store the input parameters
        self.scale = scale  # The number of dimensions in the embeddings
        self.r_min = r_min  # The minimum radius of the sphere
        self.r_max = r_max  # The maximum radius of the sphere (default: 1.0)

        # Calculate the growth factor for the beta parameter
        self.g = r_max / r_min  # Growth factor for the beta parameter

        # Create a tensor of shape (scale,) representing the scale values
        self.s = torch.arange(scale, dtype=torch.float32)  # Tensor of shape (scale,) representing the scale values

        # Calculate the beta parameter for each coordinate (latitude and longitude)
        scale_minus_one = max(self.scale - 1, 1)  # Avoid division by zero
        beta_row = r_min * self.g ** (self.s / scale_minus_one)  # Tensor of shape (2, scale) representing the beta parameter

        # Register the beta parameter as a buffer tensor, with size (2, scale)
        # The first dimension corresponds to latitude, and the second dimension corresponds to longitude
        self.register_buffer('beta', beta_row.unsqueeze(0).repeat(2, 1))  # Register the beta parameter as a buffer tensor

    def forward(self, x):
        """
        Apply the spherical coordinate embedding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor, with shape (batch_size, 2).

        Returns:
            torch.Tensor: The tensor with the spherical coordinate embeddings, with shape (batch_size, scale).
        """
        # Extract the latitude and longitude from the input tensor
        lat, lon = x[:, 0], x[:, 1]

        # Apply the beta parameter to the latitude and longitude
        lat_transform = lat.unsqueeze(-1) * self.beta[0]
        lon_transform = lon.unsqueeze(-1) * self.beta[1]

        # Calculate the final embeddings by concatenating the sine and cosine of the transformed latitude and longitude
        return torch.cat([
            # Sine of the transformed latitude
            torch.sin(lat_transform),
            # Cosine of the transformed latitude multiplied by the sine of the transformed latitude
            torch.cos(lat_transform) * torch.sin(lat_transform),
            # Cosine of the transformed longitude multiplied by the cosine of the transformed latitude
            torch.cos(lon_transform) * torch.cos(lat_transform)
        ], dim=-1)



#================================================================================
# SphericalMultiScaleEmbedding
#================================================================================
class SphericalMultiScaleEmbedding(nn.Module):
    def __init__(self, scale, r_min, r_max=1.0):
        """
        Initialize the SphericalMultiScaleEmbedding module.
        following the paper https://arxiv.org/pdf/2306.17624

        Args:
            scale (int): The number of dimensions in the embeddings.
            r_min (float): The minimum radius of the sphere.
            r_max (float, optional): The maximum radius of the sphere. Defaults to 1.0.
        """
        super().__init__()

        # Store the input parameters
        self.scale = scale
        self.r_min = r_min
        self.r_max = r_max

        # Calculate the growth factor for the beta parameter
        self.g = r_max / r_min

        # Create a tensor of shape (scale,) representing the scale values
        self.s = torch.arange(scale, dtype=torch.float32)
        """
        Tensor of shape (scale,) representing the scale values.
        The tensor is initialized with values from 0 to scale-1.
        """

        # Calculate the beta parameter for each coordinate (latitude and longitude)
        scale_minus_one = max(self.scale - 1, 1)  # avoids division by zero
        beta_row = r_min * self.g ** (self.s / scale_minus_one)
        """
        Tensor of shape (2, scale) representing the beta parameter.
        The first dimension corresponds to latitude, and the second dimension corresponds to longitude.
        """

        # Register the beta parameter as a buffer tensor
        self.register_buffer(
            'beta',  # buffer name
            beta_row.unsqueeze(0).repeat(2, 1)  # buffer tensor
        )
        """
        Register the beta parameter as a buffer tensor.
        The tensor is of size (2, scale) and is registered as a buffer tensor.
        The first dimension corresponds to latitude, and the second dimension corresponds to longitude.
        """

    def forward(self, x):
        """
        Calculate the embeddings for the given spherical coordinates.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, 2) containing the spherical coordinates.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, 5) containing the embeddings.

        The embeddings are calculated by concatenating the transformed sine and cosine values of the coordinates.
        The following embeddings are calculated for each coordinate:
        - Sine of the transformed latitude
        - Cosine of the transformed latitude multiplied by the cosine of the longitude
        - Cosine of the latitude multiplied by the transformed cosine of the longitude
        - Cosine of the transformed latitude multiplied by the sine of the longitude
        - Sine of the latitude multiplied by the transformed cosine of the longitude
        """
        # Extract latitude and longitude from the input tensor
        lat, lon = x[:, 0], x[:, 1]

        # Calculate the transformed values for latitude and longitude
        lat_transform = lat.unsqueeze(-1) * self.beta[0]
        lon_transform = lon.unsqueeze(-1) * self.beta[1]

        # Concatenate the transformed sine and cosine values of the coordinates
        return torch.cat([
            # Sine of the transformed latitude
            torch.sin(lat_transform),

            # Cosine of the transformed latitude multiplied by the cosine of the longitude
            torch.cos(lat_transform) * torch.cos(lon.unsqueeze(-1)),

            # Cosine of the latitude multiplied by the transformed cosine of the longitude
            torch.cos(lat).unsqueeze(-1) * torch.cos(lon_transform),

            # Cosine of the transformed latitude multiplied by the sine of the longitude
            torch.cos(lat_transform) * torch.sin(lon.unsqueeze(-1)),

            # Sine of the latitude multiplied by the transformed cosine of the longitude
            torch.sin(lat.unsqueeze(-1)) * torch.cos(lon_transform),
        ], dim=-1)


class DoubleFourierSphericalEmbedding(nn.Module):
    def __init__(self, scale, r_lat_min, r_lon_min, r_max=1.0):
        """
        Initialize the DoubleFourierSphericalEmbedding module.

        Args:
            scale (int): The number of dimensions in the embeddings.
            r_lat_min (float): The minimum radius of the latitude sphere.
            r_lon_min (float): The minimum radius of the longitude sphere.
            r_max (float, optional): The maximum radius of the spheres. Defaults to 1.0.
        """
        super().__init__()

        # Store the input parameters
        self.scale = scale
        self.r_lat_min = r_lat_min
        self.r_lon_min = r_lon_min

        # Create a tensor of shape (scale,) representing the scale values
        self.s = torch.arange(scale, dtype=torch.float32)

        # Calculate the growth factors for the latitude and longitude scaling factors
        self.lat_g = r_max / r_lat_min  # growth factor for latitude scaling
        self.lon_g = r_max / r_lon_min  # growth factor for longitude scaling

        # Calculate the scaling factors for latitude and longitude
        scale_minus_one = max(self.scale - 1, 1)  # avoids division by zero
        beta_lat = r_lat_min * (self.lat_g ** (self.s / scale_minus_one))  # scaling factor for latitude
        beta_lon = r_lon_min * (self.lon_g ** (self.s / scale_minus_one))  # scaling factor for longitude

        # Register the scaling factors as buffer tensors, with size (scale,)
        self.register_buffer('beta_lat', beta_lat)  # latitude scaling factors
        self.register_buffer('beta_lon', beta_lon)  # longitude scaling factors

    def forward(self, x):
        """
        Calculate the embeddings for the given spherical coordinates.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, 2) containing the spherical coordinates.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, scale * 2 + scale * (scale - 1) / 2) containing the embeddings.
        """
        lat, lon = x[:, 0], x[:, 1]

        # scale latitude and longitude coordinates
        lat_scaled = lat.unsqueeze(-1) * self.beta_lat
        lon_scaled = lon.unsqueeze(-1) * self.beta_lon

        # calculate base terms
        lat_terms = torch.cat([torch.sin(lat_scaled), torch.cos(lat_scaled)], dim=-1)
        lon_terms = torch.cat([torch.sin(lon_scaled), torch.cos(lon_scaled)], dim=-1)

        # calculate interaction terms
        lat_cos = torch.cos(lat_scaled)
        lat_sin = torch.sin(lat_scaled)
        lon_cos = torch.cos(lon_scaled)
        lon_sin = torch.sin(lon_scaled)

        # each column of interaction_terms corresponds to a combination of latitude and longitude terms
        interaction_terms = torch.cat([
            lat_cos.unsqueeze(2) * lon_cos.unsqueeze(1),  # cos(lat) cos(lon)
            lat_cos.unsqueeze(2) * lon_sin.unsqueeze(1),  # cos(lat) sin(lon)
            lat_sin.unsqueeze(2) * lon_cos.unsqueeze(1),  # sin(lat) cos(lon)
            lat_sin.unsqueeze(2) * lon_sin.unsqueeze(1),  # sin(lat) sin(lon)
        ], dim=-1).flatten(1)  # flatten SxS interactions

        # concatenate base terms, interaction terms, and return
        return torch.cat([lat_terms, lon_terms, interaction_terms], dim=-1)