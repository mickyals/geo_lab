from typing import Callable, List, Union

import numpy as np
import torch
import pyDOE as lhs

from geolab.data.components.coordinate_data.space import GeoSpatialDomain


"""
The base mesh is a base class for generating mesh data and boundary conditions if needed. It is defined by its domain bounds, direction north,
east, west, lower upper, its initial boundary and conditions and finally regular data points. 


"""
class MeshBase:
    """ This helper class is utilized for Mesh classes to generate mesh data and boundary conditions if needed."""

    def __init__(self):
        """Initialize the MeshBase class."""
        pass

    def domain_bounds(self):
        """stores the min and max values of the geosptial domain, if I store them as an n-dim array i could slice
        it to get the bounds for each dimension. and store it as a property and a small size array.
        """
        pass

    def on_boundary(self):
        """gets the boundary points of the domain, default will be east, west, north, south, upper, lower. The specified
        ones will be returned as a dict of arrays.
        """
        pass

    def collection_points(self):
        """for physical systems these would be sampled from the points not on boundaries and are used to minimise gradients.
        These points should be sampled from no bounary or boundaries not in on_bounardy. So if no upper and lower in on_boundary
        then these points can be sampled from there, if user specifies.
        """

    def data_points(self):
        " not all the time would someone need this but you get every point from the dataset with no special treatment"


    def apply_mask(self):
        """ a numpy mask to be applied to the data points along the spatial dimensions. hiding land points for example """

        pass

    def flatten_mesh(self):
        """ flattening the mesh data to provide to data loader for training. returns solution domain, spatial domain, time domain which are dicts
        of numpy arrays. """
        pass



class Mesh(MeshBase):

    """ for a well defined grid of data"""

    def __init__(self):
        """ initialisation is passsed to this class from the base class, to make a mesh i need the
        read data function, the root_dir of the data, the geospatial_domain, the shape and andy other stuff that the GeoSpatialData Class might need """