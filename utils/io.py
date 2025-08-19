# loading rasterio and xarray data from tiff, nc and zarr files
import rioxarray
import xarray as xr


def load_data(filepath, variables=None, **kwargs):
    """
    Load geospatial data from GeoTIFF, NetCDF, or Zarr files.

    :param filepath: Path to the data file.
    :param variables: Optional list of variables to load (for NetCDF/Zarr).
    :param kwargs: Additional keyword arguments passed to xarray open functions.
    :return: xarray Dataset or DataArray, or rasterio Dataset for GeoTIFF.
    """

    if filepath.endswith((".tif", ".tiff")):
        data = xr.open_dataset(filepath, engine="rasterio", **kwargs)
        # Optionally, wrap in xarray for consistent API:
        # data = xr.open_rasterio(filepath, **kwargs)
    elif filepath.endswith(".nc"):
        data = xr.open_dataset(filepath, **kwargs)
        if variables:
            data = data[variables]
    elif filepath.endswith(".zarr"):
        data = xr.open_zarr(filepath, **kwargs)
        if variables:
            data = data[variables]
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    return data
