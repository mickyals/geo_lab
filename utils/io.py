# loading rasterio and xarray data from tiff, nc and zarr files
import rioxarray
import xarray as xr


def load_data(filepath, variables=None, chunks='auto', **kwargs):
    """
    Load geospatial data from GeoTIFF, NetCDF, or Zarr files.

    The function uses xarray to load the data and returns an xarray Dataset or
    DataArray. If the file is a GeoTIFF, rasterio is used to load the data and
    the function returns a rasterio Dataset if the `wrap_in_xarray` parameter is
    set to False (default). Otherwise, the function returns an xarray DataArray
    for consistent API.

    Parameters
    ----------
    filepath : str
        Path to the data file.
    variables : list, optional
        List of variables to load (for NetCDF/Zarr).
    chunks : str or int, optional
        Chunk size (default is 'auto'). See xarray documentation for more
        information.
    kwargs : dict, optional
        Additional keyword arguments passed to xarray open functions.

    Returns
    -------
    data : xarray Dataset or DataArray, or rasterio Dataset
        The loaded data.
    """
    if filepath.endswith((".tif", ".tiff")):
        data = xr.open_dataset(filepath, engine="rasterio", chunks=chunks, **kwargs)
        # Optionally, wrap in xarray for consistent API:
        # data = xr.open_rasterio(filepath, **kwargs)
    elif filepath.endswith(".nc"):
        data = xr.open_dataset(filepath, chunks=chunks, **kwargs)
        if variables:
            data = data[variables]
    elif filepath.endswith(".zarr"):
        data = xr.open_zarr(filepath, chunks=chunks, **kwargs)
        if variables:
            data = data[variables]
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    return data


def get_dimensions(data):
    """
    Returns a dictionary with the dimensions as keys and their size as values.

    The function works with both DataArray and Dataset objects. For DataArray,
    the dimensions are obtained from the dims attribute and the sizes are
    obtained from the shape attribute. For Dataset, the dimensions are obtained
    from the dims attribute and the sizes are obtained from the length of the
    coordinates of each dimension.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        The dataset or data array to extract dimensions from.

    Returns
    -------
    dict
        A dictionary with the dimensions as keys and their size as values.

    Raises
    ------
    ValueError
        If the data type is not supported.
    """
    if isinstance(data, xr.DataArray):
        # For DataArray, the dimensions are stored in the dims attribute
        # and the sizes are stored in the shape attribute
        return dict(zip(data.dims, data.shape))
    elif isinstance(data, xr.Dataset):
        # For Dataset, the dimensions are stored in the dims attribute
        # and the sizes are the length of the coordinates of each dimension
        return {dim: len(data[dim]) for dim in data.dims}
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def get_global_attrs(data):
    """
    Returns a dictionary with the global attributes of the dataset.

    Parameters
    ----------
    data : xr.Dataset
        The dataset to extract global attributes from.

    Returns
    -------
    dict
        A dictionary with the global attributes of the dataset.
    """
    return data.attrs


def get_variable_and_attrs(data, variables=None):
    """
    Returns a dictionary with the variables as keys and their attributes as values.

    Parameters
    ----------
    data : xr.Dataset
        The dataset to extract attributes from.
    variables : list of str
        The names of the variables for which to print attributes.

    Returns
    -------
    dict
        A dictionary with the variables as keys and their attributes as values.
    """
    if variables is None:
        variables = list(data.data_vars)
    variables_and_attrs = {}
    for variable in variables:
        if variable not in data:
            print(f"Variable '{variable}' not found in dataset.")
            continue
        # Get the attributes of the variable
        attributes = data[variable].attrs
        # Add the variable and its attributes to the dictionary
        variables_and_attrs[variable] = attributes
    return variables_and_attrs

