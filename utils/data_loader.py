"""
utils/data_loader.py
Handles loading and preprocessing of NetCDF climate datasets using xarray.
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path


# Common variable name aliases across different NetCDF datasets
VARIABLE_ALIASES = {
    "Temperature": ["t2m", "tas", "temperature", "temp", "T", "air", "2m_temperature", "SAT"],
    "Precipitation": ["tp", "pr", "precipitation", "precip", "rain", "PRECT", "pcp"],
    "Wind Speed": ["si10", "sfcWind", "wind_speed", "wspd", "u10", "v10", "UV"],
    "Humidity": ["hurs", "rh", "humidity", "relative_humidity", "q"],
    "Sea Level Pressure": ["msl", "psl", "slp", "pressure"],
}

COORD_ALIASES = {
    "lat": ["lat", "latitude", "LAT", "Latitude", "nav_lat", "y"],
    "lon": ["lon", "longitude", "LON", "Longitude", "nav_lon", "x"],
    "time": ["time", "TIME", "Time", "t", "date"],
}


def resolve_coord(ds: xr.Dataset, coord_type: str) -> str | None:
    """Return the actual coordinate name in the dataset for a given type."""
    for alias in COORD_ALIASES[coord_type]:
        if alias in ds.coords or alias in ds.dims:
            return alias
    return None


def resolve_variable(ds: xr.Dataset, friendly_name: str) -> str | None:
    """Return actual variable name in dataset for a friendly label."""
    aliases = VARIABLE_ALIASES.get(friendly_name, [friendly_name])
    for alias in aliases:
        if alias in ds.data_vars:
            return alias
    return None


def load_dataset(filepath: str) -> xr.Dataset:
    """Load a NetCDF file and return an xarray Dataset."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    ds = xr.open_dataset(filepath, engine="netcdf4")
    return ds


def get_available_variables(ds: xr.Dataset) -> list[str]:
    """
    Return friendly variable names available in the dataset.
    Falls back to raw variable names if no alias matches.
    """
    friendly = []
    raw_vars = list(ds.data_vars)

    for friendly_name, aliases in VARIABLE_ALIASES.items():
        for alias in aliases:
            if alias in raw_vars:
                friendly.append(friendly_name)
                break

    # Add any unrecognised variables as-is
    recognised_raw = []
    for aliases in VARIABLE_ALIASES.values():
        recognised_raw.extend(aliases)
    for v in raw_vars:
        if v not in recognised_raw:
            friendly.append(v)

    return friendly if friendly else raw_vars


def get_time_range(ds: xr.Dataset) -> tuple:
    """Return (min_time, max_time) as pandas Timestamps."""
    time_coord = resolve_coord(ds, "time")
    if time_coord is None:
        return None, None
    times = pd.to_datetime(ds[time_coord].values)
    return times.min(), times.max()


def slice_dataset(
    ds: xr.Dataset,
    variable: str,
    time_start=None,
    time_end=None,
) -> xr.DataArray:
    """
    Slice a dataset to a single variable within a time range.
    Returns an xr.DataArray.
    """
    var_name = resolve_variable(ds, variable) or variable
    if var_name not in ds.data_vars:
        raise ValueError(f"Variable '{variable}' not found in dataset.")

    da = ds[var_name]

    time_coord = resolve_coord(ds, "time")
    if time_coord and time_start and time_end:
        da = da.sel({time_coord: slice(str(time_start), str(time_end))})

    return da


def normalize_coords(da: xr.DataArray) -> xr.DataArray:
    """
    Rename lat/lon/time dimensions to standard names for downstream use.
    """
    rename_map = {}
    for std, aliases in COORD_ALIASES.items():
        for alias in aliases:
            if alias in da.dims and alias != std:
                rename_map[alias] = std
                break
    if rename_map:
        da = da.rename(rename_map)
    return da


def get_spatial_mean(da: xr.DataArray) -> xr.DataArray:
    """Return the spatial mean over lat/lon for each time step."""
    da = normalize_coords(da)
    dims_to_avg = [d for d in ["lat", "lon"] if d in da.dims]
    return da.mean(dim=dims_to_avg)


def get_time_slice_mean(da: xr.DataArray) -> xr.DataArray:
    """Return mean over time dimension."""
    da = normalize_coords(da)
    if "time" in da.dims:
        return da.mean(dim="time")
    return da


def extract_timeseries(da: xr.DataArray, lat: float, lon: float) -> pd.Series:
    """
    Extract a time-series at the nearest grid point to (lat, lon).
    Returns a pandas Series indexed by time.
    """
    da = normalize_coords(da)
    lat_coord = da["lat"].values
    lon_coord = da["lon"].values

    # Normalise lon to 0-360 if needed
    if lon < 0 and lon_coord.min() >= 0:
        lon = lon + 360

    lat_idx = int(np.argmin(np.abs(lat_coord - lat)))
    lon_idx = int(np.argmin(np.abs(lon_coord - lon)))

    ts = da.isel(lat=lat_idx, lon=lon_idx)
    if "time" in ts.dims:
        times = pd.to_datetime(ts["time"].values)
        return pd.Series(ts.values, index=times)
    return pd.Series(ts.values)


def kelvin_to_celsius(da: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Kelvin to Celsius if values look like Kelvin."""
    if da.values.mean() > 200:
        return da - 273.15
    return da


def get_units(ds: xr.Dataset, variable: str) -> str:
    """Return units string for a variable if available."""
    var_name = resolve_variable(ds, variable) or variable
    if var_name in ds.data_vars:
        attrs = ds[var_name].attrs
        return attrs.get("units", attrs.get("unit", ""))
    return ""


def create_sample_dataset(output_path: str = "data/sample.nc"):
    """
    Generate a synthetic sample NetCDF dataset for demo/testing purposes.
    Covers global grid, 1990-2020 monthly, with temperature + precipitation.
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.random.seed(42)
    times = pd.date_range("1990-01", periods=372, freq="ME")  # 31 years monthly
    lats = np.linspace(-90, 90, 73)
    lons = np.linspace(-180, 180, 144)

    n_t, n_lat, n_lon = len(times), len(lats), len(lons)

    # --- Temperature: lat-dependent base + warming trend + seasonal cycle ---
    lat_grid = lats[:, None]
    base_temp = 30 - 0.6 * np.abs(lat_grid)  # warmest at equator
    trend = np.linspace(0, 1.5, n_t)         # ~1.5°C warming over 31 years

    temp = np.zeros((n_t, n_lat, n_lon))
    for i, t in enumerate(times):
        seasonal = 8 * np.cos(2 * np.pi * (t.month - 1) / 12) * (lat_grid / 90)
        noise = np.random.randn(n_lat, 1) * 1.5
        temp[i] = base_temp + trend[i] + seasonal + noise

    # Broadcast to full lon grid + small lon noise
    temp = np.broadcast_to(temp, (n_t, n_lat, n_lon)).copy()
    temp += np.random.randn(n_t, n_lat, n_lon) * 0.8

    # --- Precipitation: tropical peak, seasonal variation ---
    precip = np.zeros((n_t, n_lat, n_lon))
    for i, t in enumerate(times):
        trop_peak = 5 * np.exp(-((lat_grid) ** 2) / 200)
        seasonal_p = 2 * np.cos(2 * np.pi * (t.month - 7) / 12) * np.exp(-((lat_grid - 15) ** 2) / 300)
        noise_p = np.abs(np.random.randn(n_lat, 1)) * 0.5
        precip[i] = np.clip(trop_peak + seasonal_p + noise_p, 0, None)

    precip = np.broadcast_to(precip, (n_t, n_lat, n_lon)).copy()
    precip += np.abs(np.random.randn(n_t, n_lat, n_lon)) * 0.3

    ds = xr.Dataset(
        {
            "t2m": xr.DataArray(
                temp.astype("float32"),
                dims=["time", "lat", "lon"],
                attrs={"units": "°C", "long_name": "2-metre air temperature"},
            ),
            "tp": xr.DataArray(
                precip.astype("float32"),
                dims=["time", "lat", "lon"],
                attrs={"units": "mm/day", "long_name": "Total precipitation"},
            ),
        },
        coords={
            "time": times,
            "lat": xr.DataArray(lats, dims=["lat"], attrs={"units": "degrees_north"}),
            "lon": xr.DataArray(lons, dims=["lon"], attrs={"units": "degrees_east"}),
        },
        attrs={"title": "PyClimaExplorer Sample Dataset", "source": "Synthetic data for demo"},
    )

    ds.to_netcdf(output_path)
    return output_path
