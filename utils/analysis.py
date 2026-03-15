"""
utils/analysis.py
Statistical analysis, anomaly detection, and climate story generation.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────
# Anomaly Detection
# ─────────────────────────────────────────────

def detect_anomalies_zscore(series: pd.Series, threshold: float = 2.0) -> pd.Series:
    """
    Flag anomalies using z-score method.
    Returns a boolean Series (True = anomaly).
    """
    z = np.abs(stats.zscore(series.dropna()))
    mask = pd.Series(False, index=series.index)
    mask.loc[series.dropna().index] = z > threshold
    return mask


def detect_anomalies_moving_avg(
    series: pd.Series, window: int = 12, threshold: float = 2.0
) -> pd.Series:
    """
    Flag anomalies as deviations from a rolling mean beyond threshold * rolling_std.
    Returns a boolean Series (True = anomaly).
    """
    rolling_mean = series.rolling(window=window, center=True, min_periods=1).mean()
    rolling_std = series.rolling(window=window, center=True, min_periods=1).std()
    deviation = np.abs(series - rolling_mean)
    mask = deviation > (threshold * rolling_std)
    return mask.fillna(False)


def detect_anomalies(
    series: pd.Series,
    method: str = "z-score",
    threshold: float = 2.0,
    window: int = 12,
) -> pd.Series:
    """Unified anomaly detection dispatcher."""
    if method == "z-score":
        return detect_anomalies_zscore(series, threshold)
    elif method == "Moving Average Deviation":
        return detect_anomalies_moving_avg(series, window, threshold)
    return pd.Series(False, index=series.index)


# ─────────────────────────────────────────────
# Trend Analysis
# ─────────────────────────────────────────────

def compute_linear_trend(series: pd.Series) -> dict:
    """
    Fit a linear trend to a time series.
    Returns slope (per year), intercept, r², p-value.
    """
    clean = series.dropna()
    if len(clean) < 3:
        return {"slope_per_year": 0, "r2": 0, "p_value": 1, "trend_total": 0}

    # Convert index to numeric years
    if isinstance(clean.index, pd.DatetimeIndex):
        x = (clean.index - clean.index[0]).days / 365.25
    else:
        x = np.arange(len(clean), dtype=float)

    slope, intercept, r, p, _ = stats.linregress(x, clean.values)
    years_span = x[-1] - x[0]

    return {
        "slope_per_year": float(slope),
        "r2": float(r**2),
        "p_value": float(p),
        "trend_total": float(slope * years_span),
        "intercept": float(intercept),
        "x": x,
        "fitted": slope * x + intercept,
    }


def compute_decadal_means(series: pd.Series) -> pd.DataFrame:
    """Group a time series by decade and compute mean."""
    if not isinstance(series.index, pd.DatetimeIndex):
        return pd.DataFrame()
    df = series.to_frame("value")
    df["decade"] = (df.index.year // 10) * 10
    return df.groupby("decade")["value"].mean().reset_index()


def compute_rolling_mean(series: pd.Series, window: int = 12) -> pd.Series:
    """Return a rolling mean series."""
    return series.rolling(window=window, min_periods=1).mean()


# ─────────────────────────────────────────────
# Climate Insights
# ─────────────────────────────────────────────

def compute_insights(series: pd.Series, variable: str, units: str = "") -> dict:
    """
    Compute a set of summary statistics / insights for the time series.
    """
    clean = series.dropna()
    if len(clean) == 0:
        return {}

    trend = compute_linear_trend(clean)
    anomaly_mask_z = detect_anomalies_zscore(clean)
    anomaly_mask_ma = detect_anomalies_moving_avg(clean)

    years = None
    if isinstance(clean.index, pd.DatetimeIndex):
        years = clean.index.year
        year_mean = clean.groupby(years).mean()
        max_year = int(year_mean.idxmax())
        min_year = int(year_mean.idxmin())
    else:
        max_year = min_year = None

    u = f" {units}" if units else ""

    return {
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "max_year": max_year,
        "min_year": min_year,
        "trend_total": trend["trend_total"],
        "slope_per_year": trend["slope_per_year"],
        "r2": trend["r2"],
        "p_value": trend["p_value"],
        "n_anomalies_zscore": int(anomaly_mask_z.sum()),
        "n_anomalies_ma": int(anomaly_mask_ma.sum()),
        "units": units,
        "variable": variable,
    }


# ─────────────────────────────────────────────
# Climate Story Generator
# ─────────────────────────────────────────────

def generate_climate_story(
    series: pd.Series,
    variable: str,
    units: str = "",
    lat: float = None,
    lon: float = None,
) -> str:
    """
    Auto-generate a natural-language climate story from a time series.
    """
    clean = series.dropna()
    if len(clean) == 0:
        return "No data available to generate a climate story."

    insights = compute_insights(clean, variable, units)
    trend = compute_linear_trend(clean)
    u = units if units else ""

    # Time span
    if isinstance(clean.index, pd.DatetimeIndex):
        year_start = clean.index.year.min()
        year_end = clean.index.year.max()
        span_years = year_end - year_start
    else:
        year_start = year_end = span_years = None

    # Location description
    if lat is not None and lon is not None:
        lat_str = f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}"
        lon_str = f"{abs(lon):.1f}°{'E' if lon >= 0 else 'W'}"
        loc = f" at {lat_str}, {lon_str}"
    else:
        loc = " in the selected region"

    # Build story paragraphs
    story_parts = []

    # --- Overview ---
    if year_start:
        story_parts.append(
            f"📅 **Overview ({year_start}–{year_end})**\n"
            f"This analysis covers {span_years} years of {variable.lower()} data{loc}. "
            f"The average {variable.lower()} over this period was **{insights['mean']:.2f}{u}**, "
            f"ranging from {insights['min']:.2f}{u} to {insights['max']:.2f}{u}."
        )
    else:
        story_parts.append(
            f"📊 **Overview**\n"
            f"The average {variable.lower()}{loc} was **{insights['mean']:.2f}{u}**, "
            f"ranging from {insights['min']:.2f}{u} to {insights['max']:.2f}{u}."
        )

    # --- Trend ---
    total = insights["trend_total"]
    slope = insights["slope_per_year"]
    r2 = insights["r2"]
    p = insights["p_value"]
    confidence = "statistically significant" if p < 0.05 else "not statistically significant at the 95% level"

    if abs(total) < 0.05:
        trend_desc = f"no meaningful trend ({confidence})"
    elif total > 0:
        trend_desc = f"an **increasing trend** of +{total:.2f}{u} over the period (+{slope:.3f}{u}/year)"
    else:
        trend_desc = f"a **decreasing trend** of {total:.2f}{u} over the period ({slope:.3f}{u}/year)"

    story_parts.append(
        f"📈 **Trend Analysis**\n"
        f"The data shows {trend_desc}. "
        f"The linear fit explains {r2*100:.1f}% of the variance (R² = {r2:.3f}), "
        f"which is {confidence}."
    )

    # --- Extremes ---
    if insights.get("max_year") and insights.get("min_year"):
        story_parts.append(
            f"🌡️ **Extreme Years**\n"
            f"The highest annual mean {variable.lower()} was recorded in **{insights['max_year']}** "
            f"and the lowest in **{insights['min_year']}**."
        )

    # --- Anomalies ---
    n_z = insights["n_anomalies_zscore"]
    n_ma = insights["n_anomalies_ma"]
    if n_z > 0:
        pct = n_z / len(clean) * 100
        story_parts.append(
            f"⚠️ **Anomaly Detection**\n"
            f"Z-score analysis identified **{n_z} anomalous observations** ({pct:.1f}% of the record). "
            f"Moving-average deviation flagged **{n_ma} events**. "
            f"These anomalies may indicate extreme climate events such as heatwaves, cold snaps, "
            f"or unusual precipitation episodes."
        )
    else:
        story_parts.append(
            f"✅ **Anomaly Detection**\n"
            f"No significant anomalies were detected in this record."
        )

    # --- Interpretation ---
    if variable == "Temperature" and total > 0.5:
        story_parts.append(
            f"🌍 **Climate Context**\n"
            f"A warming of {total:.2f}°C over {span_years} years is consistent with broader "
            f"global warming trends observed in climate science. Continued warming at this rate "
            f"could have significant implications for ecosystems, agriculture, and water resources "
            f"in this region."
        )
    elif variable == "Precipitation" and abs(total) > 0.2:
        direction = "increase" if total > 0 else "decrease"
        story_parts.append(
            f"💧 **Climate Context**\n"
            f"The {direction} in precipitation of {abs(total):.2f}{u} over {span_years} years "
            f"may reflect shifting atmospheric circulation patterns or changes in monsoon systems. "
            f"This trend could affect regional water availability and flood risk."
        )

    return "\n\n".join(story_parts)
