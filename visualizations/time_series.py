"""
visualizations/time_series.py
Time-series charts with anomaly highlighting and trend overlay.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─── Dark-theme palette ───────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#38BDF8",   # sky blue — main line
    "trend":     "#FB923C",   # orange — trend line
    "rolling":   "#A78BFA",   # violet — rolling mean
    "anomaly":   "#F87171",   # red — anomaly markers
    "fill":      "#1E3A5F",   # dark fill for confidence bands
    "grid":      "#1E293B",
    "bg":        "#0F172A",
    "text":      "#E2E8F0",
    "subtext":   "#94A3B8",
    "bar":       "#22D3EE",
}


def _base_layout(title: str, yaxis_title: str, height: int = 420) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color=PALETTE["text"]), x=0.5),
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
        font=dict(color=PALETTE["text"]),
        xaxis=dict(
            gridcolor=PALETTE["grid"],
            zerolinecolor=PALETTE["grid"],
            tickfont=dict(color=PALETTE["subtext"]),
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor=PALETTE["grid"],
            zerolinecolor=PALETTE["grid"],
            tickfont=dict(color=PALETTE["subtext"]),
        ),
        legend=dict(
            bgcolor="rgba(15,23,42,0.8)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color=PALETTE["text"]),
        ),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=60, b=60),
        height=height,
    )


def plot_time_series(
    series: pd.Series,
    variable: str,
    units: str = "",
    anomaly_mask: pd.Series = None,
    show_rolling: bool = True,
    show_trend: bool = True,
    rolling_window: int = 12,
    lat: float = None,
    lon: float = None,
) -> go.Figure:
    """
    Full-featured time-series chart:
    - Raw line
    - Rolling mean
    - Linear trend
    - Anomaly scatter overlay
    """
    unit_label = f" ({units})" if units else ""
    loc = ""
    if lat is not None and lon is not None:
        lat_str = f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}"
        lon_str = f"{abs(lon):.1f}°{'E' if lon >= 0 else 'W'}"
        loc = f" @ {lat_str}, {lon_str}"

    title = f"{variable} Time Series{loc}"
    fig = go.Figure()

    # ── Raw line ──────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        name=f"{variable}",
        line=dict(color=PALETTE["primary"], width=1.2),
        opacity=0.8,
        hovertemplate=f"%{{x}}<br>{variable}: %{{y:.2f}}{units}<extra></extra>",
    ))

    # ── Rolling mean ──────────────────────────────────────────────────────────
    if show_rolling and len(series) > rolling_window:
        roll = series.rolling(window=rolling_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=roll.index,
            y=roll.values,
            mode="lines",
            name=f"{rolling_window}-step rolling mean",
            line=dict(color=PALETTE["rolling"], width=2, dash="dot"),
            hovertemplate=f"%{{x}}<br>Rolling mean: %{{y:.2f}}{units}<extra></extra>",
        ))

    # ── Linear trend ──────────────────────────────────────────────────────────
    if show_trend and len(series) > 2:
        from utils.analysis import compute_linear_trend
        trend = compute_linear_trend(series)
        if trend.get("x") is not None:
            fitted = pd.Series(trend["fitted"], index=series.dropna().index)
            slope = trend["slope_per_year"]
            sign = "+" if slope >= 0 else ""
            fig.add_trace(go.Scatter(
                x=fitted.index,
                y=fitted.values,
                mode="lines",
                name=f"Trend ({sign}{slope:.3f}{units}/yr)",
                line=dict(color=PALETTE["trend"], width=2, dash="dash"),
                hovertemplate=f"Trend: %{{y:.2f}}{units}<extra></extra>",
            ))

    # ── Anomalies ─────────────────────────────────────────────────────────────
    if anomaly_mask is not None and anomaly_mask.any():
        anom_vals = series[anomaly_mask]
        fig.add_trace(go.Scatter(
            x=anom_vals.index,
            y=anom_vals.values,
            mode="markers",
            name="Anomaly",
            marker=dict(
                color=PALETTE["anomaly"],
                size=8,
                symbol="circle-open",
                line=dict(width=2),
            ),
            hovertemplate=f"%{{x}}<br>⚠️ Anomaly: %{{y:.2f}}{units}<extra></extra>",
        ))

    fig.update_layout(**_base_layout(title, f"{variable}{unit_label}"))
    return fig


def plot_annual_bar(
    series: pd.Series,
    variable: str,
    units: str = "",
) -> go.Figure:
    """Annual mean bar chart coloured by value."""
    if not isinstance(series.index, pd.DatetimeIndex):
        return go.Figure()

    annual = series.groupby(series.index.year).mean()
    unit_label = f" ({units})" if units else ""

    # Colour gradient: cool→warm for temperature, else single colour
    if "temp" in variable.lower():
        norm = (annual.values - annual.values.min()) / (annual.values.ptp() + 1e-9)
        colors = [f"rgb({int(60+195*v)},{int(130-100*v)},{int(200-180*v)})" for v in norm]
    else:
        colors = PALETTE["bar"]

    fig = go.Figure(go.Bar(
        x=annual.index,
        y=annual.values,
        marker_color=colors,
        hovertemplate=f"%{{x}}<br>{variable}: %{{y:.2f}}{units}<extra></extra>",
        name=variable,
    ))

    fig.update_layout(
        **_base_layout(f"Annual Mean {variable}", f"{variable}{unit_label}", height=360)
    )
    return fig


def plot_anomaly_heatmap_calendar(
    series: pd.Series,
    variable: str,
    units: str = "",
) -> go.Figure:
    """
    Month × Year heatmap (calendar view) of monthly anomalies from the
    long-term monthly mean — inspired by climate stripe diagrams.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        return go.Figure()

    df = series.to_frame("val")
    df["year"] = df.index.year
    df["month"] = df.index.month

    monthly_clim = df.groupby("month")["val"].mean()
    df["anomaly"] = df.apply(lambda r: r["val"] - monthly_clim[r["month"]], axis=1)

    pivot = df.pivot_table(index="year", columns="month", values="anomaly")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    colorscale = "RdBu_r" if "temp" in variable.lower() else "BrBG"

    absmax = np.nanmax(np.abs(pivot.values))
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[month_names[m - 1] for m in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=colorscale,
        zmid=0,
        zmin=-absmax,
        zmax=absmax,
        colorbar=dict(title=f"Δ{units}", thickness=14, len=0.7),
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Anomaly: %{z:.2f}" + units + "<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(f"{variable} Monthly Anomalies (vs. Climatology)", "Year", height=420)
    )
    fig.update_xaxes(tickfont=dict(color=PALETTE["subtext"]))
    return fig


def plot_multi_variable(
    series_dict: dict,
    title: str = "Multi-Variable Comparison",
) -> go.Figure:
    """
    Plot multiple time series on the same axes.
    series_dict: {label: pd.Series}
    """
    colors = [PALETTE["primary"], PALETTE["trend"], PALETTE["rolling"],
              "#34D399", "#FBBF24", "#F472B6"]
    fig = go.Figure()

    for i, (label, s) in enumerate(series_dict.items()):
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            mode="lines",
            name=label,
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(**_base_layout(title, "Value"))
    return fig
