"""
visualizations/heatmap.py
Global climate heatmap using Plotly.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import xarray as xr


# Colour scales per variable
COLORSCALES = {
    "Temperature": "RdBu_r",
    "Precipitation": "Blues",
    "Wind Speed": "Viridis",
    "Humidity": "Teal",
    "Sea Level Pressure": "Plasma",
    "default": "Turbo",
}

UNIT_LABELS = {
    "Temperature": "°C",
    "Precipitation": "mm/day",
    "Wind Speed": "m/s",
    "Humidity": "%",
    "Sea Level Pressure": "hPa",
}


def _get_colorscale(variable: str) -> str:
    for key in COLORSCALES:
        if key.lower() in variable.lower():
            return COLORSCALES[key]
    return COLORSCALES["default"]


def _get_unit_label(variable: str, units: str = "") -> str:
    if units:
        return units
    for key, u in UNIT_LABELS.items():
        if key.lower() in variable.lower():
            return u
    return ""


def plot_global_heatmap(
    da: xr.DataArray,
    variable: str,
    time_label: str = "",
    units: str = "",
    projection: str = "natural earth",
) -> go.Figure:
    """
    Render a global filled-contour heatmap on a geographic map.

    Parameters
    ----------
    da      : 2-D DataArray with lat/lon coords (time already sliced)
    variable: friendly variable name for titles / colouring
    time_label: string shown in the title
    units   : unit string for colour-bar label
    projection: Plotly geo projection name
    """
    # Ensure 2-D (lat × lon)
    if "time" in da.dims:
        da = da.isel(time=0)

    # Resolve coord names
    lat_name = next((c for c in da.dims if "lat" in c.lower()), da.dims[0])
    lon_name = next((c for c in da.dims if "lon" in c.lower()), da.dims[1])

    lats = da[lat_name].values
    lons = da[lon_name].values
    vals = da.values

    # Flatten for scatter_geo
    lat_flat, lon_flat = np.meshgrid(lats, lons, indexing="ij")
    lat_flat = lat_flat.ravel()
    lon_flat = lon_flat.ravel()
    val_flat = vals.ravel()

    valid = ~np.isnan(val_flat)
    lat_flat = lat_flat[valid]
    lon_flat = lon_flat[valid]
    val_flat = val_flat[valid]

    colorscale = _get_colorscale(variable)
    unit_label = _get_unit_label(variable, units)
    title = f"{variable} Heatmap"
    if time_label:
        title += f" — {time_label}"

    fig = go.Figure()

    fig.add_trace(
        go.Densitymap(
            lat=lat_flat,
            lon=lon_flat,
            z=val_flat,
            radius=8,
            colorscale=colorscale,
            colorbar=dict(title=unit_label, thickness=15, len=0.7),
            hovertemplate=(
                f"<b>{variable}</b><br>"
                "Lat: %{lat:.2f}°<br>"
                "Lon: %{lon:.2f}°<br>"
                f"Value: %{{z:.2f}} {unit_label}<extra></extra>"
            ),
            showscale=True,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#E2E8F0"), x=0.5),
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=0, lon=0),
            zoom=0.5,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"),
        height=480,
    )

    return fig


def plot_heatmap_choropleth(
    da: xr.DataArray,
    variable: str,
    time_label: str = "",
    units: str = "",
) -> go.Figure:
    """
    Alternative: render a lat/lon heatmap as a filled contour using go.Heatmap
    (faster, works offline without mapbox token).
    """
    if "time" in da.dims:
        da = da.isel(time=0)

    lat_name = next((c for c in da.dims if "lat" in c.lower()), da.dims[0])
    lon_name = next((c for c in da.dims if "lon" in c.lower()), da.dims[1])

    lats = da[lat_name].values
    lons = da[lon_name].values
    vals = da.values

    colorscale = _get_colorscale(variable)
    unit_label = _get_unit_label(variable, units)
    title = f"{variable} Heatmap"
    if time_label:
        title += f" — {time_label}"

    fig = go.Figure(
        go.Heatmap(
            z=vals,
            x=lons,
            y=lats,
            colorscale=colorscale,
            colorbar=dict(
                title=dict(text=unit_label, side="right"),
                thickness=15,
                len=0.8,
            ),
            hovertemplate=(
                "Lat: %{y:.1f}°<br>"
                "Lon: %{x:.1f}°<br>"
                f"{variable}: %{{z:.2f}} {unit_label}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#E2E8F0"), x=0.5),
        xaxis=dict(
            title="Longitude",
            tickfont=dict(color="#94A3B8"),
            gridcolor="#1E293B",
            zerolinecolor="#334155",
        ),
        yaxis=dict(
            title="Latitude",
            tickfont=dict(color="#94A3B8"),
            gridcolor="#1E293B",
            zerolinecolor="#334155",
            scaleanchor="x",
            scaleratio=1,
        ),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"),
        margin=dict(l=60, r=40, t=60, b=60),
        height=480,
    )

    return fig


def plot_comparison_heatmap(
    da1: xr.DataArray,
    da2: xr.DataArray,
    variable: str,
    label1: str = "Period 1",
    label2: str = "Period 2",
    units: str = "",
) -> go.Figure:
    """
    Side-by-side comparison heatmap (bonus: Comparison Mode).
    Shows two time slices as subplots.
    """
    from plotly.subplots import make_subplots

    def _prep(da):
        if "time" in da.dims:
            da = da.mean(dim="time")
        lat_name = next((c for c in da.dims if "lat" in c.lower()), da.dims[0])
        lon_name = next((c for c in da.dims if "lon" in c.lower()), da.dims[1])
        return da[lat_name].values, da[lon_name].values, da.values

    lats, lons, vals1 = _prep(da1)
    _, _, vals2 = _prep(da2)

    colorscale = _get_colorscale(variable)
    unit_label = _get_unit_label(variable, units)
    zmin = min(np.nanmin(vals1), np.nanmin(vals2))
    zmax = max(np.nanmax(vals1), np.nanmax(vals2))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[label1, label2],
        horizontal_spacing=0.08,
    )

    common = dict(
        x=lons, y=lats,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        colorbar=dict(title=unit_label, thickness=12, len=0.7),
        showscale=False,
    )

    fig.add_trace(go.Heatmap(z=vals1, **common, showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=vals2, **common, showscale=True,
                              colorbar=dict(title=unit_label, thickness=12, len=0.7, x=1.02)),
                  row=1, col=2)

    fig.update_layout(
        title=dict(
            text=f"{variable} Comparison: {label1} vs {label2}",
            font=dict(size=15, color="#E2E8F0"), x=0.5,
        ),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"),
        height=420,
        margin=dict(l=50, r=80, t=70, b=50),
    )

    for axis in ["xaxis", "xaxis2"]:
        fig.update_layout(**{axis: dict(tickfont=dict(color="#94A3B8"), gridcolor="#1E293B")})
    for axis in ["yaxis", "yaxis2"]:
        fig.update_layout(**{axis: dict(tickfont=dict(color="#94A3B8"), gridcolor="#1E293B")})

    return fig
