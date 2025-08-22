import xarray as xr
import rioxarray # needed for tiff files but not explicitly called
import numpy as np
import dask as da
import dask.array as darray
import math

PI = math.pi


def wrap_longitude(
    ds: xr.Dataset, lon_name: str = "longitude", lat_name: str = "latitude"
) -> xr.Dataset:
    """
    Wraps longitude coordinates to the range [-180, 180] and sorts the dataset by longitude.
    Also renames latitude and longitude to standard names if they have different names from "latitude" and "longitude".

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset.
    lon_name : str, optional
        The name of the longitude coordinate, by default "longitude".
    lat_name : str, optional
        The name of the latitude coordinate, by default "latitude".

    Returns
    -------
    xr.Dataset
        The dataset with wrapped and sorted longitude coordinates.
    """
    # Rename coordinates to standard names if they have different names
    rename_dict = {}
    if lon_name in ds and lon_name != "longitude":
        rename_dict[lon_name] = "longitude"
    if lat_name in ds and lat_name != "latitude":
        rename_dict[lat_name] = "latitude"

    if rename_dict:
        # Rename coordinates to standard names
        ds = ds.rename(rename_dict)

    # Wrap longitude to [-180, 180] and sort
    # This is done by first adding 180 to each longitude value and taking the modulus of 360
    # This will result in values in the range [0, 360) being shifted to [-180, 180)
    # Then subtract 180 from each longitude value to get the range [-180, 180]
    ds = ds.assign_coords(
        longitude=(((ds["longitude"] + 180) % 360) - 180)
    ).sortby("longitude")

    # The sorting is necessary because the longitude wrapping may move some values to the end of the array
    # The sort will put them back in the correct order
    return ds



def extract_lat_lon_coordinates(ds: xr.Dataset, field_of_view: float = PI):
    """
    Extracts and normalizes the latitude and longitude coordinates from a dataset.

    The coordinates are normalized to the range [0, 1] and are then converted to a
    2D grid with the shape (nlat, nlon, 2) where nlat and nlon are the number of
    latitude and longitude points respectively.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to extract coordinates from.
    field_of_view : float
        The view angle of the coordinates in radians, defaults to pi.

    Returns
    -------
    dask.array.Array
        The normalized coordinates, with shape (nlat * nlon, 2) where nlat and nlon are the number of
        latitude and longitude points respectively.
    """

    # --- Unnormalized in degrees ---
    lons_deg = ds["longitude"].data  # dask or numpy
    lats_deg = ds["latitude"].data

    lon_grid_deg, lat_grid_deg = darray.meshgrid(lons_deg, lats_deg, indexing="xy")
    lat_lon_coords = darray.stack([lat_grid_deg.ravel(), lon_grid_deg.ravel()], axis=-1)

    # convert lat lon to radians
    lons = darray.radians(ds["longitude"].data)
    lats = darray.radians(ds["latitude"].data)

    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    lon_norm = (lons - lon_min) / (lon_max - lon_min)
    lat_norm = (lats - lat_min) / (lat_max - lat_min)

    # build flattened lat lon grid
    lon_grid, lat_grid = darray.meshgrid(lon_norm, lat_norm, indexing="xy")

    # lon should be -fov to fov - assuming global domain
    lon_c = (lon_grid - 0.5) * (2 * field_of_view)

    # lat should be -fov/2 to fov/2 - assuming global domain change this to -fov to fov if not global domain
    lat_c = (lat_grid - 0.5) * field_of_view

    lat_lon_coords_norm = darray.stack((lat_c.ravel(), lon_c.ravel()), axis=-1)  # shape (nlat*nlon, 2)


    return lat_lon_coords, lat_lon_coords_norm # this is a dask array to access variables you need compute()


def extract_time_coordinates(
    ds: xr.Dataset,  # The dataset to extract time coordinates from.
    time_name: str = "time"  # The name of the time coordinate, defaults to "time".
):
    """
    Extracts time coordinates from a dataset and normalizes them to [0, 1].

    The time coordinates are first converted to nanoseconds since the epoch
    and then normalized to the range [0, 1].

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to extract time coordinates from.
    time_name : str
        The name of the time coordinate, defaults to "time".

    Returns
    -------
    da.Array
        The normalized time coordinates as a Dask array, shape (T,).
    """
    # convert times to nanoseconds since the epoch
    times = darray.from_array(ds[time_name].data)  # shape (T,)
    if len(times.shape) == 1:
        time_coords = darray.from_array(np.array([0.]))
        return time_coords
    times = times.astype('datetime64[ns]')  # ensure times are nanoseconds

    # normalize the times to the range [0, 1]
    t0 = times.min() # get min time
    t1 = times.max() # get max time
    total_time = (t1 - t0).astype('float64')  # total time
    dt = (times - t0).astype('float64')  # shape (T,)
    time_coords = dt / total_time  # shape (T,)
    time_coords = da.from_array(time_coords)

    return time_coords


def extract_other_coordinates(ds: xr.Dataset, coordinate_names: list):
    """
    Extracts non-spatial coordinates from a dataset and normalizes them.

    This function takes a dataset and a list of coordinate names and returns a
    Dask array of normalized coordinates. The coordinates are normalized to the
    range [0, 1]. The output array has shape (npoints, ncoords) where npoints is
    the number of data points and ncoords is the number of coordinate dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to extract coordinates from.
    coordinate_names : list
        The names of the coordinates to extract.

    Returns
    -------
    da.Array
        The normalized coordinates.
    """
    # empty list to store the normalized coordinates
    coord_list = []
    norm_coords = []

    # loop through the coordinate names
    for coordinate_name in coordinate_names:
        # get the coordinates from the dataset
        coordinates = darray.from_array(ds[coordinate_name].data).astype('float32')
        coord_list.append(coordinates.ravel())

        # normalize the coordinates by subtracting the minimum and
        # then dividing by the range
        coord_min, coord_max = coordinates.min(), coordinates.max()
        coord_norm = (coordinates - coord_min) / (coord_max - coord_min)

        # append the normalized coordinates to the list
        norm_coords.append(coord_norm)

    # stack the normalized coordinates along the last axis
    other_coords = darray.stack(coord_list, axis=-1)
    norm_other_coords = darray.stack(norm_coords, axis=-1)

    return other_coords, norm_other_coords


def extract_variables(
    ds: xr.Dataset,  # The dataset to extract variables from.
    variable_names: list,  # The names of the variables to extract.
):  # The extracted and normalized variables, stacked along the last axis.
    """
    Extract variables from a dataset and normalize them to the range [0, 1].

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to extract variables from.
    variable_names : list
        The names of the variables to extract.

    Returns
    -------
    da.Array
        The extracted and normalized variables, stacked along the last axis.
    """
    # Initialize a list to store the extracted variables
    norm_var_list = []
    var_list = []

    # Iterate over the variable names
    for variable_name in variable_names:
        # Get the variable from the dataset
        var = ds[variable_name].data
        var_list.append(var.astype('float32').ravel())

        # Get the min and max values of the variable
        var_min, var_max = var.min(), var.max()

        # Normalize the variable to the range [0, 1]
        var_norm = (var - var_min) / (var_max - var_min)

        # Append the normalized variable to the list
        norm_var_list.append(var_norm.astype('float32').ravel())

    # Stack the variables along the last axis
    vars = darray.stack(var_list, axis=-1)
    norm_vars = darray.stack(norm_var_list, axis=-1)

    return vars, norm_vars

def keep_coords(
    ds: xr.Dataset,
    coordinate_names: list,
) -> xr.Dataset:
    """
    Drops all coordinates from a dataset except for the ones specified in the
    coordinate_names list.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset.
    coordinate_names : list
        The names of the coordinates to keep.

    Returns
    -------
    xr.Dataset
        The dataset with all coordinates except for the ones in coordinate_names
        dropped.
    """
    # Get the list of coordinates to drop
    coords_to_drop = [coord for coord in ds.coords if coord not in coordinate_names]

    # Drop the coordinates
    return ds.drop_vars(coords_to_drop, errors='ignore')

def spatial_mask(
    ds: xr.Dataset,
    variable_name: str = None,
    mask_value: float = None,
):
    """
    Creates a mask of a dataset or variable to identify fill values.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to create a mask for.
    variable_name : str, optional
        The name of the variable to create a mask for, by default None
    mask_value : float, optional
        The value to use as the fill value, by default None. If not provided, the
        _FillValue attribute from the dataset or variable will be used.

    Returns
    -------
    np.ndarray
        A boolean mask of the same shape as the dataset or variable, where
        True indicates a fill value.
    np.ndarray
        An array of indices of shape (nfillvalues, ndim) where each row is an index
        into the mask array.
    """
    # If variable_name is provided, get the fill value from the variable attributes
    if variable_name is not None:
        # Check that the variable exists in the dataset
        if variable_name not in ds:
            raise ValueError(f"Variable {variable_name} not found in dataset.")
        # Get the fill value from the variable attributes
        fill_value = ds[variable_name].attrs.get("_FillValue", None)
        # If mask_value is provided, use it instead
        if mask_value is not None:
            fill_value = mask_value
        # If mask_value is still None, raise an error
        if fill_value is None:
            raise ValueError(
                "mask_value must be provided if variable_name is provided or no _FillValue attribute found."
            )

        data = ds[variable_name]
    else:
        # Get the fill value from the dataset attributes
        fill_value = ds.attrs.get("_FillValue", None)
        # If mask_value is provided, use it instead
        if mask_value is not None:
            fill_value = mask_value
        # If mask_value is still None, raise an error
        if fill_value is None:
            raise ValueError(
                "mask_value must be provided if variable_name is provided or no _FillValue attribute found."
            )
        # Convert the dataset data to a dask array
        data = ds.data

    # Create a mask of the dataset or variable where fill values are True
    mask = data == fill_value
    # Get the indices of the fill values in the mask
    indices = darray.argwhere(mask.data.ravel())

    # Return the mask and indices
    return mask, indices


