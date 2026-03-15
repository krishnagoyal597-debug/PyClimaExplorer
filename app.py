"""
app.py  ─  PyClimaExplorer
Interactive climate data exploration dashboard.
Run with:  streamlit run app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="PyClimaExplorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: dark theme polish ─────────────────────────────────────────────
st.markdown("""
<style>
/* Global background */
.stApp { background-color: #0A1628; color: #E2E8F0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F1F3D 0%, #0A1628 100%);
    border-right: 1px solid #1E3A5F;
}
[data-testid="stSidebar"] .stMarkdown { color: #94A3B8; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #0F2044;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 12px;
}
[data-testid="stMetricValue"] { color: #38BDF8; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: #94A3B8; }
[data-testid="stMetricDelta"] { font-size: 0.85rem; }

/* Headers */
h1, h2, h3 { color: #E2E8F0 !important; }

/* Insight box */
.insight-box {
    background: #0F2044;
    border-left: 4px solid #38BDF8;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 8px 0;
    color: #CBD5E1;
    line-height: 1.7;
}

/* Story box */
.story-box {
    background: linear-gradient(135deg, #0F2044 0%, #0F172A 100%);
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 20px 24px;
    color: #CBD5E1;
    line-height: 1.8;
}

/* Warning badge */
.anomaly-badge {
    background: rgba(248,113,113,0.15);
    border: 1px solid #F87171;
    border-radius: 20px;
    padding: 4px 12px;
    color: #F87171;
    font-size: 0.85rem;
    font-weight: 600;
}

/* Section divider */
.section-header {
    border-bottom: 1px solid #1E3A5F;
    padding-bottom: 6px;
    margin: 20px 0 14px 0;
    color: #38BDF8 !important;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #94A3B8;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #38BDF8 !important;
    border-bottom-color: #38BDF8 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0369A1, #0EA5E9);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0EA5E9, #38BDF8);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0F2044;
    border: 1px dashed #1E3A5F;
    border-radius: 10px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0A1628; }
::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Project imports ───────────────────────────────────────────────────────────
from utils.data_loader import (
    load_dataset, get_available_variables, get_time_range,
    slice_dataset, normalize_coords, get_spatial_mean,
    get_time_slice_mean, extract_timeseries, kelvin_to_celsius,
    get_units, create_sample_dataset, resolve_variable,
)
from utils.analysis import (
    detect_anomalies, compute_insights, generate_climate_story,
    compute_rolling_mean, compute_linear_trend, compute_decadal_means,
)
from visualizations.heatmap import (
    plot_heatmap_choropleth, plot_comparison_heatmap,
)
from visualizations.time_series import (
    plot_time_series, plot_annual_bar,
    plot_anomaly_heatmap_calendar,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state defaults
# ─────────────────────────────────────────────────────────────────────────────
if "ds" not in st.session_state:
    st.session_state.ds = None
if "filepath" not in st.session_state:
    st.session_state.filepath = None


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load & cache dataset
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading climate dataset…")
def cached_load(path: str):
    return load_dataset(path)


@st.cache_data(show_spinner=False)
def cached_timeseries(path, variable, ts, te, lat, lon):
    ds = cached_load(path)
    da = slice_dataset(ds, variable, ts, te)
    da = normalize_coords(da)
    if variable == "Temperature":
        da = kelvin_to_celsius(da)
    return extract_timeseries(da, lat, lon)


@st.cache_data(show_spinner=False)
def cached_spatial_mean(path, variable, ts, te):
    ds = cached_load(path)
    da = slice_dataset(ds, variable, ts, te)
    da = normalize_coords(da)
    if variable == "Temperature":
        da = kelvin_to_celsius(da)
    return get_spatial_mean(da)


@st.cache_data(show_spinner=False)
def cached_time_slice(path, variable, time_idx):
    ds = cached_load(path)
    var_name = resolve_variable(ds, variable) or variable
    da = ds[var_name].isel(time=time_idx)
    da = normalize_coords(da)
    if variable == "Temperature":
        da = kelvin_to_celsius(da)
    return da


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 PyClimaExplorer")
    st.markdown("<p style='color:#64748B;font-size:0.8rem;'>Technex '26 · Quantum Bits</p>",
                unsafe_allow_html=True)
    st.divider()

    # ── Dataset source ────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📂 Dataset</p>', unsafe_allow_html=True)
    data_source = st.radio(
        "Choose data source",
        ["Use Sample Dataset", "Upload NetCDF File", "Enter File Path"],
        label_visibility="collapsed",
    )

    filepath = None

    if data_source == "Use Sample Dataset":
        sample_path = "data/sample.nc"
        if not os.path.exists(sample_path):
            with st.spinner("Generating sample dataset…"):
                create_sample_dataset(sample_path)
        filepath = sample_path
        st.success("✅ Sample dataset ready (1990–2020 monthly)")

    elif data_source == "Upload NetCDF File":
        uploaded = st.file_uploader("Upload .nc file", type=["nc", "nc4"])
        if uploaded:
            save_path = f"data/uploaded_{uploaded.name}"
            os.makedirs("data", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded.read())
            filepath = save_path
            st.success(f"✅ {uploaded.name} uploaded")

    else:
        filepath = st.text_input("NetCDF file path", value="data/sample.nc")

    # ── Load dataset ──────────────────────────────────────────────────────────
    if filepath and os.path.exists(filepath):
        if st.session_state.filepath != filepath:
            st.session_state.ds = cached_load(filepath)
            st.session_state.filepath = filepath

        ds = st.session_state.ds

        # ── Variable selector ─────────────────────────────────────────────────
        st.markdown('<p class="section-header">🌡️ Variable</p>', unsafe_allow_html=True)
        available_vars = get_available_variables(ds)
        variable = st.selectbox("Select variable", available_vars)
        units = get_units(ds, variable)

        # ── Time range ────────────────────────────────────────────────────────
        st.markdown('<p class="section-header">📅 Time Range</p>', unsafe_allow_html=True)
        t_min, t_max = get_time_range(ds)
        if t_min and t_max:
            year_min, year_max = int(t_min.year), int(t_max.year)
            sel_years = st.slider(
                "Year range",
                min_value=year_min, max_value=year_max,
                value=(year_min, year_max),
            )
            time_start = pd.Timestamp(f"{sel_years[0]}-01-01")
            time_end   = pd.Timestamp(f"{sel_years[1]}-12-31")
        else:
            time_start = time_end = None
            st.info("No time dimension detected.")

        # ── Location picker ───────────────────────────────────────────────────
        st.markdown('<p class="section-header">📍 Location (Time Series)</p>',
                    unsafe_allow_html=True)
        sel_lat = st.slider("Latitude",  -90.0, 90.0, 20.0, 0.5)
        sel_lon = st.slider("Longitude", -180.0, 180.0, 78.0, 0.5)

        # ── Analysis options ──────────────────────────────────────────────────
        st.markdown('<p class="section-header">🔬 Analysis</p>', unsafe_allow_html=True)
        anom_method = st.selectbox(
            "Anomaly detection method",
            ["z-score", "Moving Average Deviation"],
        )
        anom_threshold = st.slider("Anomaly threshold (σ)", 1.0, 4.0, 2.0, 0.1)
        rolling_window = st.slider("Rolling mean window", 3, 36, 12)

        # ── Bonus features ────────────────────────────────────────────────────
        st.markdown('<p class="section-header">✨ Bonus Features</p>',
                    unsafe_allow_html=True)
        show_comparison  = st.checkbox("Comparison Mode", value=False)
        show_story       = st.checkbox("Climate Story Mode", value=True)
        show_calendar    = st.checkbox("Monthly Anomaly Calendar", value=False)

        if show_comparison:
            comp_year1 = st.slider("Compare: Period 1 end year",
                                   year_min, year_max, min(year_min + 9, year_max))
            comp_year2 = st.slider("Compare: Period 2 start year",
                                   year_min, year_max, max(year_max - 9, year_min))

    else:
        st.warning("⚠️ No dataset loaded yet.")
        ds = None
        variable = units = None
        time_start = time_end = None
        sel_lat = sel_lon = None
        anom_method = "z-score"
        anom_threshold = 2.0
        rolling_window = 12
        show_comparison = show_story = show_calendar = False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🌍 PyClimaExplorer")
st.markdown(
    "<p style='color:#64748B;margin-top:-10px;'>Interactive Climate Data Exploration Dashboard &nbsp;·&nbsp; Technex '26</p>",
    unsafe_allow_html=True,
)

if ds is None:
    st.info("👈 Load a dataset from the sidebar to get started.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# QUICK STATS ROW
# ─────────────────────────────────────────────────────────────────────────────
try:
    spatial_ts = cached_spatial_mean(filepath, variable, time_start, time_end)
    spatial_series = pd.Series(spatial_ts.values, index=pd.to_datetime(spatial_ts.time.values)) \
        if hasattr(spatial_ts, "time") else pd.Series(spatial_ts.values)

    global_mean = float(spatial_series.mean())
    global_std  = float(spatial_series.std())
    n_times     = len(spatial_series)

    if isinstance(spatial_series.index, pd.DatetimeIndex):
        annual = spatial_series.groupby(spatial_series.index.year).mean()
        max_yr = int(annual.idxmax())
        min_yr = int(annual.idxmin())
        trend  = compute_linear_trend(spatial_series)
        delta  = trend["trend_total"]
        delta_str = f"{'+' if delta>=0 else ''}{delta:.2f} {units}"
    else:
        max_yr = min_yr = None
        delta_str = "N/A"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌡️ Global Mean",   f"{global_mean:.2f} {units}")
    c2.metric("📊 Std Dev",        f"±{global_std:.2f} {units}")
    c3.metric("📅 Data Points",    f"{n_times:,}")
    c4.metric("📈 Total Trend",    delta_str)
    c5.metric("🔥 Warmest Year" if "temp" in variable.lower() else "📌 Peak Year",
              str(max_yr) if max_yr else "–")

except Exception as e:
    st.warning(f"Could not compute quick stats: {e}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Heatmap",
    "📈 Time Series",
    "⚠️ Anomalies",
    "📖 Climate Story",
    "🔬 Deep Analysis",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: GLOBAL HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🗺️ Global Climate Heatmap")

    col_map, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown("**Map Controls**")

        # Time slice slider
        da_full = slice_dataset(ds, variable, time_start, time_end)
        da_full = normalize_coords(da_full)
        if "time" in da_full.dims:
            n_t = len(da_full.time)
            time_idx = st.slider("Time slice", 0, max(n_t - 1, 0), 0,
                                 help="Move through time steps")
            try:
                time_label_str = str(pd.to_datetime(da_full.time.values[time_idx]).strftime("%b %Y"))
            except Exception:
                time_label_str = f"Step {time_idx}"
        else:
            time_idx = 0
            time_label_str = ""

        show_mean_map = st.checkbox("Show period mean instead", value=False)

    with col_map:
        with st.spinner("Rendering heatmap…"):
            try:
                if show_mean_map:
                    da_map = get_time_slice_mean(da_full)
                    tl = f"Mean {sel_years[0]}–{sel_years[1]}" if time_start else "Period Mean"
                else:
                    da_map = cached_time_slice(filepath, variable, time_idx)
                    tl = time_label_str

                if variable == "Temperature":
                    da_map = kelvin_to_celsius(da_map)

                fig_map = plot_heatmap_choropleth(da_map, variable, tl, units)
                st.plotly_chart(fig_map, use_container_width=True)
            except Exception as e:
                st.error(f"Heatmap error: {e}")

    # ── Comparison Mode ───────────────────────────────────────────────────────
    if show_comparison:
        st.markdown("---")
        st.markdown("### ↔️ Comparison Mode")
        try:
            da1 = slice_dataset(ds, variable,
                                pd.Timestamp(f"{sel_years[0]}-01-01"),
                                pd.Timestamp(f"{comp_year1}-12-31"))
            da2 = slice_dataset(ds, variable,
                                pd.Timestamp(f"{comp_year2}-01-01"),
                                pd.Timestamp(f"{sel_years[1]}-12-31"))
            da1 = normalize_coords(da1)
            da2 = normalize_coords(da2)
            if variable == "Temperature":
                da1 = kelvin_to_celsius(da1)
                da2 = kelvin_to_celsius(da2)

            fig_comp = plot_comparison_heatmap(
                da1, da2, variable,
                label1=f"{sel_years[0]}–{comp_year1}",
                label2=f"{comp_year2}–{sel_years[1]}",
                units=units,
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        except Exception as e:
            st.error(f"Comparison error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📈 Time-Series Trend Analysis")
    st.markdown(
        f"<p style='color:#64748B'>Showing <b>{variable}</b> at "
        f"<b>{abs(sel_lat):.1f}°{'N' if sel_lat>=0 else 'S'}, "
        f"{abs(sel_lon):.1f}°{'E' if sel_lon>=0 else 'W'}</b> "
        f"({sel_years[0] if time_start else ''}–{sel_years[1] if time_end else ''})</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Extracting time series…"):
        try:
            ts = cached_timeseries(filepath, variable, time_start, time_end, sel_lat, sel_lon)
            anomaly_mask = detect_anomalies(ts, anom_method, anom_threshold, rolling_window)

            fig_ts = plot_time_series(
                ts, variable, units,
                anomaly_mask=anomaly_mask,
                show_rolling=True,
                show_trend=True,
                rolling_window=rolling_window,
                lat=sel_lat, lon=sel_lon,
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            # Annual bar
            st.markdown("#### 📊 Annual Means")
            fig_bar = plot_annual_bar(ts, variable, units)
            st.plotly_chart(fig_bar, use_container_width=True)

        except Exception as e:
            st.error(f"Time series error: {e}")

    # Spatial mean time series
    st.markdown("#### 🌐 Global Spatial Mean Over Time")
    try:
        smts = pd.Series(
            spatial_ts.values,
            index=pd.to_datetime(spatial_ts.time.values)
            if hasattr(spatial_ts, "time") else range(len(spatial_ts)),
        )
        fig_sm = plot_time_series(
            smts, f"{variable} (Global Mean)", units,
            show_rolling=True, show_trend=True,
            rolling_window=rolling_window,
        )
        st.plotly_chart(fig_sm, use_container_width=True)
    except Exception as e:
        st.warning(f"Global mean plot: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### ⚠️ Climate Anomaly Detection")

    try:
        ts_anom = cached_timeseries(filepath, variable, time_start, time_end, sel_lat, sel_lon)
        mask_z  = detect_anomalies(ts_anom, "z-score", anom_threshold)
        mask_ma = detect_anomalies(ts_anom, "Moving Average Deviation", anom_threshold, rolling_window)

        col_a, col_b = st.columns(2)
        col_a.metric("⚠️ Z-Score Anomalies",         f"{mask_z.sum()}",
                     delta=f"{mask_z.sum()/len(ts_anom)*100:.1f}% of record")
        col_b.metric("⚠️ Moving-Avg Anomalies",       f"{mask_ma.sum()}",
                     delta=f"{mask_ma.sum()/len(ts_anom)*100:.1f}% of record")

        st.markdown("#### Z-Score Method")
        fig_z = plot_time_series(
            ts_anom, variable, units,
            anomaly_mask=mask_z,
            show_rolling=True, show_trend=False,
            rolling_window=rolling_window, lat=sel_lat, lon=sel_lon,
        )
        st.plotly_chart(fig_z, use_container_width=True)

        st.markdown("#### Moving Average Deviation Method")
        fig_ma = plot_time_series(
            ts_anom, variable, units,
            anomaly_mask=mask_ma,
            show_rolling=True, show_trend=False,
            rolling_window=rolling_window, lat=sel_lat, lon=sel_lon,
        )
        st.plotly_chart(fig_ma, use_container_width=True)

        # Monthly anomaly calendar
        if show_calendar and isinstance(ts_anom.index, pd.DatetimeIndex):
            st.markdown("#### 📅 Monthly Anomaly Calendar")
            fig_cal = plot_anomaly_heatmap_calendar(ts_anom, variable, units)
            st.plotly_chart(fig_cal, use_container_width=True)

        # Table of anomalous dates
        if mask_z.any():
            st.markdown("#### 📋 Anomalous Dates (Z-Score)")
            anom_df = ts_anom[mask_z].reset_index()
            anom_df.columns = ["Date", f"{variable} ({units})"]
            anom_df["Date"] = anom_df["Date"].dt.strftime("%Y-%m")
            st.dataframe(anom_df.style.highlight_max(color="#1E3A5F"), use_container_width=True)

    except Exception as e:
        st.error(f"Anomaly detection error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: CLIMATE STORY
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📖 Climate Story Mode")
    st.markdown(
        "<p style='color:#64748B'>Auto-generated natural-language narrative explaining climate trends.</p>",
        unsafe_allow_html=True,
    )

    if show_story:
        try:
            ts_story = cached_timeseries(
                filepath, variable, time_start, time_end, sel_lat, sel_lon
            )
            story = generate_climate_story(ts_story, variable, units, sel_lat, sel_lon)

            st.markdown(f'<div class="story-box">{story}</div>', unsafe_allow_html=True)

            # Also show for global mean
            st.markdown("---")
            st.markdown("#### 🌐 Global Mean Story")
            smts_story = pd.Series(
                spatial_ts.values,
                index=pd.to_datetime(spatial_ts.time.values)
                if hasattr(spatial_ts, "time") else range(len(spatial_ts)),
            )
            global_story = generate_climate_story(smts_story, variable, units)
            st.markdown(f'<div class="story-box">{global_story}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Story generation error: {e}")
    else:
        st.info("Enable **Climate Story Mode** in the sidebar to see auto-generated narratives.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: DEEP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🔬 Deep Analysis & Insights")

    try:
        ts_deep = cached_timeseries(
            filepath, variable, time_start, time_end, sel_lat, sel_lon
        )
        insights = compute_insights(ts_deep, variable, units)
        trend    = compute_linear_trend(ts_deep)
        decadal  = compute_decadal_means(ts_deep)

        # ── Insight cards ──────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean", f"{insights['mean']:.2f} {units}")
        c2.metric("Std Dev", f"±{insights['std']:.2f} {units}")
        c3.metric("Range", f"{insights['min']:.2f} to {insights['max']:.2f} {units}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Total Trend", f"{insights['trend_total']:+.3f} {units}")
        c5.metric("Rate/Year",   f"{insights['slope_per_year']:+.4f} {units}/yr")
        c6.metric("R²",          f"{insights['r2']:.3f}")

        # ── Decadal table ──────────────────────────────────────────────────────
        if not decadal.empty:
            st.markdown("#### 📊 Decadal Mean Table")
            decadal["value"] = decadal["value"].round(3)
            decadal.columns = ["Decade", f"Mean {variable} ({units})"]
            st.dataframe(decadal, use_container_width=True, hide_index=True)

        # ── Insight bullets ────────────────────────────────────────────────────
        st.markdown("#### 💡 Auto-Generated Insights")

        insight_lines = []
        u = units

        if insights.get("max_year"):
            insight_lines.append(
                f"🔥 Highest annual mean {variable.lower()} in **{insights['max_year']}** "
                f"({insights['max']:.2f} {u})"
            )
        if insights.get("min_year"):
            insight_lines.append(
                f"❄️ Lowest annual mean {variable.lower()} in **{insights['min_year']}** "
                f"({insights['min']:.2f} {u})"
            )

        sign = "+" if insights["trend_total"] >= 0 else ""
        insight_lines.append(
            f"📈 Long-term trend: **{sign}{insights['trend_total']:.3f} {u}** "
            f"over the full period ({sign}{insights['slope_per_year']:.4f} {u}/year)"
        )

        if insights["p_value"] < 0.05:
            insight_lines.append(
                f"✅ Trend is **statistically significant** (p = {insights['p_value']:.4f})"
            )
        else:
            insight_lines.append(
                f"ℹ️ Trend is **not significant** at 95% level (p = {insights['p_value']:.4f})"
            )

        if insights["n_anomalies_zscore"] > 0:
            pct = insights["n_anomalies_zscore"] / len(ts_deep) * 100
            insight_lines.append(
                f"⚠️ **{insights['n_anomalies_zscore']} anomalous observations** detected "
                f"({pct:.1f}% of record)"
            )

        for line in insight_lines:
            st.markdown(f'<div class="insight-box">{line}</div>', unsafe_allow_html=True)

        # ── Raw data preview ────────────────────────────────────────────────────
        with st.expander("🗃️ Raw Data Preview"):
            df_raw = ts_deep.to_frame(f"{variable} ({units})")
            if isinstance(df_raw.index, pd.DatetimeIndex):
                df_raw.index = df_raw.index.strftime("%Y-%m")
            st.dataframe(df_raw.round(4), use_container_width=True)

    except Exception as e:
        st.error(f"Analysis error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#334155;font-size:0.8rem;'>"
    "PyClimaExplorer · Quantum Bits · Technex '26 · IIT(BHU)"
    "</p>",
    unsafe_allow_html=True,
)
