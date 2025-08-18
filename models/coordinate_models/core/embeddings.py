# positional encoders from fourier features to sphere2vec
import torch
import torch.nn as nn


EMBEDDINGS = {}



def register_embedding(name):
    """
    Registers an embedding with the given name.

    Args:
        name (str): The name of the embedding.

    Returns:
        function: A decorator that registers the embedding with the given name.

    Raises:
        ValueError: If the embedding with the given name already exists.
    """
    def decorator(cls):
        """
        Registers an embedding with the given name.

        Args:
            cls (type): The embedding class to register.

        Raises:
            ValueError: If the embedding with the given name already exists.
        """
        if name in EMBEDDINGS:
            raise ValueError(f"Embedding with name {name} already exists.")
        EMBEDDINGS[name] = cls
        return cls
    return decorator



def get_embedding(name, **kwargs):
    """
    Returns an instance of the embedding with the given name and keyword arguments.

    Args:
        name (str): The name of the embedding.
        **kwargs: Keyword arguments to initialize the embedding.

    Returns:
        nn.Module: An instance of the embedding with the given name and keyword arguments.

    Raises:
        ValueError: If the embedding with the given name does not exist.
    """
    name = name.upper()
    # Check if the embedding with the given name exists
    if name not in EMBEDDINGS:
        raise ValueError(f"Embedding with name {name} does not exist.")

    # Return an instance of the embedding with the given keyword arguments
    return EMBEDDINGS[name](**kwargs)

#==============================================================================
# GaussianFourierEmbedding
#==============================================================================
@register_embedding('GAUSSIAN_POSITIONAL')
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

        # Store the input parameters
        self.input_dim = input_dim
        """
        The input dimension.
        """
        self.mapping_dim = mapping_dim
        """
        The mapping dimension.
        """
        self.scale = scale
        """
        The scale parameter.
        """

        # Initialize the beta parameter
        self.beta = torch.randn(input_dim, mapping_dim // 2) * scale
        """
        Beta parameter. It is a tensor of shape (input_dim, mapping_dim//2)
        where the first dimension represents the input dimension and the second
        dimension represents the mapping dimension. The beta parameter is
        initialized with random values multiplied by the scale parameter.
        """

        self.out_features = mapping_dim
        """
        The output feature dimension. It is equal to the mapping dimension.
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
@register_embedding('GENERAL_POSITIONAL')
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

        # Store the input parameters
        self.input_dim = input_dim
        """
        The input dimension.
        """
        self.mapping_dim = mapping_dim
        """
        The mapping dimension.
        """
        self.scale = scale
        """
        The scale parameter.
        """

        # Create a tensor of size mapping_dim, initialized with values from 0 to mapping_dim-1
        j = torch.arange(mapping_dim, dtype=torch.float32)
        """
        Tensor of size mapping_dim, initialized with values from 0 to mapping_dim-1.
        """

        # Calculate the beta parameter for each input dimension
        beta_row = scale ** (j/mapping_dim)
        """
        Tensor of size mapping_dim, calculated as scale raised to the power of j/mapping_dim.
        """

        # Register the beta parameter as a buffer tensor, with size (input_dim, mapping_dim)
        self.register_buffer('beta', beta_row.unsqueeze(0).repeat(input_dim, 1))
        """
        Buffer tensor of size (input_dim, mapping_dim), initialized with the values of beta_row repeated input_dim times along the first dimension.
        """

        self.out_features = 2 * mapping_dim
        """
        The output feature dimension. It is equal to the mapping dimension.
        """

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

@register_embedding('SPHERE_GRID')
class SphericalGridEmbedding(nn.Module):

    def __init__(self, scale, r_min, r_max=1.0):
        """
        Initializes the SphericalGridEmbedding module.

        Args:
            scale (int): The scale parameter.
            r_min (float): The minimum radius.
            r_max (float, optional): The maximum radius. Defaults to 1.0.

        Following the paper https://arxiv.org/pdf/2306.17624

        Output feature dimension: 4 * scale
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

        self.out_features = 4 * self.scale  # The output feature dimension


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

@register_embedding('SPHERE_C')
class SphericalCartesianEmbedding(nn.Module):
    def __init__(self, scale, r_min, r_max=1.0):
        """
        Initialize the SphericalCartesianEmbedding module.

        Following the paper https://arxiv.org/pdf/2306.17624

        Output feature dimension: 3 * scale

        Args:
            scale (int): The number of dimensions in the embeddings.
            r_min (float): The minimum radius of the sphere.
            r_max (float, optional): The maximum radius of the sphere. Defaults to 1.0.
        """
        super().__init__()

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
        self.register_buffer(
            'beta',  # Name of the buffer tensor
            beta_row.unsqueeze(0).repeat(2, 1)  # Tensor with size (2, scale) representing the beta parameter
        )

        # The output feature dimension is 3 * scale
        self.out_features = 3 * self.scale

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
@register_embedding('SPHERE_M')
class SphericalMultiScaleEmbedding(nn.Module):
    def __init__(self, scale, r_min, r_max=1.0):
        """
        Initialize the SphericalMultiScaleEmbedding module.

        Args:
            scale (int): The number of dimensions in the embeddings.
            r_min (float): The minimum radius of the sphere.
            r_max (float, optional): The maximum radius of the sphere. Defaults to 1.0.
        """
        super().__init__()

        # Store the input parameters
        self.scale = scale  # The number of dimensions in the embeddings
        self.r_min = r_min  # The minimum radius of the sphere
        self.r_max = r_max  # The maximum radius of the sphere (default: 1.0)

        # Calculate the growth factor for the beta parameter
        self.g = r_max / r_min  # Growth factor for the beta parameter
        """
        The growth factor for the beta parameter. It is calculated as the ratio of the maximum radius to the minimum radius.
        """

        # Create a tensor of shape (scale,) representing the scale values
        self.s = torch.arange(scale, dtype=torch.float32)  # Tensor of shape (scale,) representing the scale values
        """
        Tensor of shape (scale,) representing the scale values. The tensor is initialized with values from 0 to scale-1.
        """

        # Calculate the beta parameter for each coordinate (latitude and longitude)
        scale_minus_one = max(self.scale - 1, 1)  # Avoid division by zero
        beta_row = r_min * self.g ** (self.s / scale_minus_one)  # Tensor of shape (2, scale) representing the beta parameter
        """
        Tensor of shape (2, scale) representing the beta parameter.
        The first dimension corresponds to latitude, and the second dimension corresponds to longitude.
        """

        self.out_features = 5 * self.scale  # The output feature dimension is 5 * scale

        # Register the beta parameter as a buffer tensor
        self.register_buffer(
            'beta',  # Name of the buffer tensor
            beta_row.unsqueeze(0).repeat(2, 1)  # Tensor with size (2, scale) representing the beta parameter
        )  # Register the beta parameter as a buffer tensor
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

#================================================================================
# DoubleFourierSphericalEmbedding
#================================================================================
@register_embedding('DFS')
class DoubleFourierSphericalEmbedding(nn.Module):
    def __init__(self, scale, r_lat_min, r_lon_min, r_max=1.0):
        """
        Initialize the DoubleFourierSphericalEmbedding module.

        This module calculates embeddings for spherical coordinates following the paper
        https://arxiv.org/pdf/2306.17624

        Args:
            scale (int): The number of dimensions in the embeddings.
            r_lat_min (float): The minimum radius of the latitude sphere.
            r_lon_min (float): The minimum radius of the longitude sphere.
            r_max (float, optional): The maximum radius of the spheres. Defaults to 1.0.
        """
        super().__init__()

        # Store the input parameters
        self.scale = scale  # number of dimensions in embeddings
        self.r_lat_min = r_lat_min  # minimum radius of latitude sphere
        self.r_lon_min = r_lon_min  # minimum radius of longitude sphere

        # Create a tensor of shape (scale,) representing the scale values
        self.s = torch.arange(scale, dtype=torch.float32)  # scale values tensor

        # Calculate the growth factors for the latitude and longitude scaling factors
        self.lat_g = r_max / r_lat_min  # growth factor for latitude scaling
        self.lon_g = r_max / r_lon_min  # growth factor for longitude scaling

        # Calculate the scaling factors for latitude and longitude
        scale_minus_one = max(self.scale - 1, 1)  # avoids division by zero
        beta_lat = r_lat_min * (self.lat_g ** (self.s / scale_minus_one))  # latitude scaling factors
        beta_lon = r_lon_min * (self.lon_g ** (self.s / scale_minus_one))  # longitude scaling factors

        # Register the scaling factors as buffer tensors, with size (scale,)
        self.register_buffer('beta_lat', beta_lat)  # latitude scaling factors
        self.register_buffer('beta_lon', beta_lon)  # longitude scaling factors

        # Calculate the output features
        self.out_features = 4 * (self.scale) ** 2 + (4 * self.scale)

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

#================================================================================
# SphericalCartesianPlusEmbedding
#================================================================================
@register_embedding('SPHERE_C+')
class SphericalCartesianPlusEmbedding(nn.Module):
    def __init__(self, scale, r_min, r_max=1.0):
        """
        Initialize the SphericalCartesianPlusEmbedding module.

        Following the paper https://arxiv.org/pdf/2306.17624.
        The output_dim is 6 * scale.

        Args:
            scale (int): The number of dimensions in the embeddings.
            r_min (float): The minimum radius of the sphere.
            r_max (float, optional): The maximum radius of the sphere. Defaults to 1.0.
        """
        super().__init__()

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
        self.register_buffer(
            'beta',  # Name of the buffer tensor
            beta_row.unsqueeze(0).repeat(2, 1)  # Tensor with size (2, scale) representing the beta parameter
        )  # Register the beta parameter as a buffer tensor

        # Set the output feature dimension
        self.out_features = 6 * scale

    def forward(self, x):
        """
        Calculate the embeddings for the given spherical coordinates.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, 2) containing the spherical coordinates.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, 7) containing the embeddings.

        The embeddings are calculated by concatenating the transformed sine and cosine values of the coordinates.
        The following embeddings are calculated for each coordinate:
        - Sine of the transformed latitude
        - Cosine of the transformed latitude
        - Sine of the transformed longitude
        - Cosine of the transformed longitude
        - Cosine of the transformed latitude multiplied by the sine of the transformed latitude
        - Cosine of the transformed longitude multiplied by the cosine of the transformed latitude
        """

        # Extract the latitude and longitude from the input tensor
        lat, lon = x[:, 0], x[:, 1]

        # Apply the beta parameter to the latitude and longitude
        lat_transform = lat.unsqueeze(-1) * self.beta[0]
        lon_transform = lon.unsqueeze(-1) * self.beta[1]

        return torch.cat([
            torch.sin(lat_transform),  # Sine of the transformed latitude
            torch.cos(lat_transform),  # Cosine of the transformed latitude
            torch.sin(lon_transform),  # Sine of the transformed longitude
            torch.cos(lon_transform),  # Cosine of the transformed longitude
            # Cosine of the transformed latitude multiplied by the sine of the transformed latitude
            torch.cos(lat_transform) * torch.sin(lat_transform),
            # Cosine of the transformed longitude multiplied by the cosine of the transformed latitude
            torch.cos(lon_transform) * torch.cos(lat_transform)
        ], dim=-1)


#================================================================================
# SphericalMultiscalePlusEmbedding
#================================================================================
@register_embedding('SPHERE_M+')
class SphericalMultiScalePlusEmbedding(nn.Module):
    """
    Module for generating multiscale embeddings for spherical coordinates.
    following the paper https://arxiv.org/pdf/2306.17624
    output_dim = 8 * scale

    Args:
        scale (int): The number of dimensions in the embeddings.
        r_min (float): The minimum radius of the sphere.
        r_max (float, optional): The maximum radius of the sphere. Defaults to 1.0.
    """
    def __init__(self, scale, r_min, r_max=1.0):
        """
        Initialize the SphericalMultiscalePlusEmbedding module.
        following the paper https://arxiv.org/pdf/2306.17624
        output_dim = 8 * scale

        Args:
            scale (int): The number of dimensions in the embeddings.
            r_min (float): The minimum radius of the sphere.
            r_max (float, optional): The maximum radius of the sphere. Defaults to 1.0.
        """
        super().__init__()

        # Store the input parameters
        self.scale = scale  # The number of dimensions in the embeddings
        self.r_min = r_min  # The minimum radius of the sphere
        self.r_max = r_max  # The maximum radius of the sphere (default: 1.0)

        # Calculate the growth factor for the beta parameter
        self.g = r_max / r_min  # Growth factor for the beta parameter

        # Create a tensor of shape (scale,) representing the scale values
        self.s = torch.arange(scale, dtype=torch.float32)
        """
        Tensor of shape (scale,) representing the scale values.
        The tensor is initialized with values from 0 to scale-1.
        """

        # Calculate the beta parameter for each coordinate (latitude and longitude)
        scale_minus_one = max(self.scale - 1, 1)  # Avoid division by zero
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

        self.out_features = 8 * self.scale
        """
        The number of output features of the embedding.
        The output features are determined by the scale parameter.
        """

    def forward(self, x):
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
            torch.cos(lat_transform),  # Cosine of the transformed latitude
            torch.sin(lon_transform),  # Sine of the transformed longitude
            torch.cos(lon_transform),  # Cosine of the transformed longitude

        ], dim=-1)



