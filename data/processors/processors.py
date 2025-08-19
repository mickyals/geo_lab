import xarray as xr
import rioxarray
import numpy as np

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

    return ds


def extract_lat_lon_coordinates(ds: xr.Dataset, field_of_view: float = np.pi):
    """
    Extracts and normalizes the latitude and longitude coordinates from a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to extract coordinates from.
    field_of_view : float
        The view angle of the coordinates in radians, defaults to pi.

    Returns
    -------
    np.ndarray
        The normalized coordinates, shape (nlat*nlon, 2).
    """

     #mask is true where there is a fill value

    # convert lat lon to radians
    lons = np.deg2rad(ds["longitude"].values)
    lats = np.deg2rad(ds["latitude"].values)

    # get dimensions of lats and lons
    nlat, nlon = lats.size, lons.size

    # build flattened lat lon grid
    lon_grid, lat_grid = np.meshgrid(lons, lats, indexing="xy")
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()

    # normalise lat and lon
    lon_min, lon_max = lon_flat.min(), lon_flat.max()
    lat_min, lat_max = lat_flat.min(), lat_flat.max()
    lon_norm = (lon_flat - lon_min) / (lon_max - lon_min)
    lat_norm = (lat_flat - lat_min) / (lat_max - lat_min)

    # lon should be -fov to fov - assuming global domain
    lon_c = (lon_norm - 0.5) * (2 * field_of_view)

    # lat should be -fov/2 to fov/2 - assuming global domain change this to -fov to fov if not global domain
    lat_c = (lat_norm - 0.5) * field_of_view

    lat_lon_coords = np.stack((lat_c, lon_c), axis=-1)  # shape (nlat*nlon, 2)

    return lat_lon_coords

def extract_time_coordinates(ds: xr.Dataset, time_name: str = "time") -> np.ndarray:
    """
    Extracts time coordinates from a dataset and normalizes them to [0, 1].

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to extract time coordinates from.
    time_name : str
        The name of the time coordinate, defaults to "time".

    Returns
    -------
    np.ndarray
        The normalized time coordinates, shape (T,).
    """

    times = ds[time_name].values.astype('datetime64[ns]')  # shape (T,)
    t0, t1 = times.min(), times.max()
    dt = (times - t0).astype('timedelta64[s]').astype('float64')  # shape (T,)
    total_time = (t1 - t0).astype('timedelta64[s]').astype('float64')
    time_coords = dt / total_time  # shape (T,)

    return time_coords

def extract_other_coordinates(ds: xr.Dataset, coordinate_names: list) -> np.ndarray:
    """
    Extracts non-spatial coordinates from a dataset and normalizes them.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to extract coordinates from.
    coordinate_names : list
        The names of the coordinates to extract.

    Returns
    -------
    np.ndarray
        The normalized coordinates, shape (ncoord, nvalues, nlevels).
    """
    coord_list = []
    for coordinate_name in coordinate_names:
        coordinates = ds[coordinate_name].values
        # normalize
        coord_min, coord_max = coordinates.min(), coordinates.max()
        coord_norm = (coordinates - coord_min) / (coord_max - coord_min)
        coord_list.append(coord_norm)

    return np.stack(coord_list, axis=-1)


def extract_variables(ds:xr.Dataset, variable_names: list):
    """
    Extract variables from a dataset and normalize them.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to extract variables from.
    variable_names : list
        The names of the variables to extract.

    Returns
    -------
    np.ndarray
        The extracted and normalized variables, stacked along the last axis.
    """
    var_list = []
    for variable_name in variable_names:
        var = ds[variable_name].values
        var_min, var_max = var.min(), var.max()
        var_norm = (var - var_min) / (var_max - var_min)
        var_list.append(var_norm)

    return np.stack(var_list, axis=-1)

def spatial_mask(
    ds: xr.Dataset,
    variable_name: str = None,
    mask_value: float = None,
) -> np.ndarray:
    """
    Creates a mask of a dataset or variable to identify fill values.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to create a mask for.
    variable_name : str, optional
        The name of the variable to create a mask for, by default None
    mask_value : float, optional
        The value to use as the fill value, by default None.

    Returns
    -------
    np.ndarray
        A boolean mask of the same shape as the dataset or variable, where
        True indicates a fill value.
    """
    # If variable_name is provided, get the fill value from the variable attributes
    if variable_name is not None:
        # Check if the variable exists in the dataset
        if variable_name not in ds:
            raise ValueError(f"Variable {variable_name} not found in dataset.")
        # Get the fill value from the variable attributes if not provided
        fill_value = ds[variable_name].attrs.get("_FillValue", None)
        # Use the provided mask value if not None, otherwise use the fill value
        mask_value = fill_value if mask_value is None else mask_value
        # Raise an error if the mask value is still None
        if mask_value is None:
            raise ValueError(
                "mask_value must be provided if variable_name is provided or no _FillValue attribute found."
            )
        # Create the mask using the mask value
        mask = ds[variable_name].values == mask_value
    # If variable_name is not provided, get the fill value from the dataset attributes
    else:
        # Get the fill value from the dataset attributes if not provided
        fill_value = ds.attrs.get("_FillValue", None)
        # Use the provided mask value if not None, otherwise use the fill value
        mask_value = fill_value if mask_value is None else mask_value
        # Raise an error if the mask value is still None
        if mask_value is None:
            raise ValueError(
                "mask_value must be provided if variable_name is provided or no _FillValue attribute found."
            )
        # Create the mask using the mask value
        mask = ds.values == mask_value

    # Return the mask as a numpy array
    return np.argwhere(mask)


def sampler(seed: int, total_samples: int, num_sensors: int) -> np.ndarray:
    """
    Creates an array of random sample indices from a given range of total samples.

    Parameters
    ----------
    seed : int
        The seed for the random number generator.
    total_samples : int
        The total number of samples to draw from.
    num_ssensors : int
        The number of samples to draw.

    Returns
    -------
    np.ndarray
        An array of num_samples random sample indices from the range [0, total_samples).
    """
    # Create a random number generator with the given seed
    rng = np.random.default_rng(seed)

    # Use the random number generator to draw num_samples random samples from
    # the range [0, total_samples)
    # if total_samples equals num_samples, this will return a shuffled list of indices
    return rng.choice(total_samples, size=num_sensors, replace=False)







