# -*- coding: utf-8 -*-
"""
Dash app: CSFlux (TGA310 methane EC) dashboard.

- Reads Cattle_Experiment_Eagle_TGA310_CSFlux.dat from GitHub (Prajaya2017/cattle_methane_flux, main)
- Plots ONLY main flux and meteorological variables
  (no QC flags, no SIGMA/statistics, no sample counts, no diagnostics)
- Tabs "Site and Setup" (site photos + descriptions), "Fluxes and Turbulence" and "Meteorology", grid of subplots, calendar date-range picker
- If start_date == end_date, shows the FULL single day (00:00:00 to 23:59:59.999999)
- Duplicate / out-of-order records are removed (sorted by TIMESTAMP)
- Render-ready: start with  gunicorn app:server
- Re-downloads the file from GitHub every REFRESH_MINUTES so new pushes appear
  without restarting the service

@author: pprajapati
"""

import os
os.environ["DASH_JUPYTER_MODE"] = "_none"

import math
import threading
import time
from io import StringIO
import requests
import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================
# USER SETTINGS
# =========================
TIME_COL = "TIMESTAMP"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

GITHUB_REPO = "Prajaya2017/cattle_methane_flux"
BRANCH = "main"
FILENAME = "Cattle_Experiment_Eagle_TGA310_CSFlux.dat"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{FILENAME}"

# How often to re-download data from GitHub (minutes)
REFRESH_MINUTES = int(os.environ.get("REFRESH_MINUTES", "10"))

GRID_COLS = 5
ROW_HEIGHT_PX = 230

# Main variables only, grouped into tabs (edit to add/remove)
TABS = {
    "Fluxes and Turbulence": [
        "FCH4_mass",     # CH4 flux
        "FC_mass",       # CO2 flux
        "LE",            # latent heat flux
        "H",             # sensible heat flux
        "ET",            # evapotranspiration
        "TAU",           # momentum flux
        "USTAR",         # friction velocity
        "TKE",           # turbulent kinetic energy
        "Bowen_ratio",
    ],
    "Meteorology": [
        ("Air & soil temperature (deg C)",
         [("TA_1_1_1", "Air temp"), ("TS_1_1_1", "Soil temp")]),   # one plot, two lines
        "RH_1_1_1",      # relative humidity
        "T_DP_1_1_1",    # dew point
        "e_amb",         # vapor pressure
        "VPD",           # vapor pressure deficit
        "PA",            # air pressure
        "WS",            # wind speed
        "WS_MAX",        # max wind speed
        "WD",            # wind direction
        "SWC_1_1_1",     # soil water content
    ],
}

# Wind direction correction: added to WD (compass wind direction) when the data is read
WD_COL = "WD"
WD_OFFSET_DEG = 180
WS_COL = "WS"

FLUX_TAB = "Fluxes and Turbulence"
MET_TAB = "Meteorology"
FCH4_COL, FCH4_QC_COL = "FCH4_mass", "FCH4_QC"     # ugCH4 m-2 s-1
FC_COL, FC_QC_COL = "FC_mass", "FC_QC"             # mgCO2 m-2 s-1
M_CH4, M_CO2 = 16.04, 44.01                        # g mol-1

# Wind direction filter (degrees, after offset). N wraps around 0.
WD_SECTORS = {
    "N (337.5-22.5)": (337.5, 22.5), "NE (22.5-67.5)": (22.5, 67.5),
    "E (67.5-112.5)": (67.5, 112.5), "SE (112.5-157.5)": (112.5, 157.5),
    "S (157.5-202.5)": (157.5, 202.5), "SW (202.5-247.5)": (202.5, 247.5),
    "W (247.5-292.5)": (247.5, 292.5), "NW (292.5-337.5)": (292.5, 337.5),
}

# Optional fixed Y ranges, e.g. {"FCH4_mass": [-1, 5]}
Y_RANGES = {}


# =========================
# Helpers (GitHub + TOA5)
# =========================
def _looks_like_toa5(text: str) -> bool:
    if not text:
        return False
    t = text.lstrip()
    return t.startswith('"TOA5"') or t.startswith("TOA5")


def fetch_toa5_text_from_github() -> str:
    # Public repo: no token / login used
    headers = {"Accept": "application/vnd.github.raw"}
    r = requests.get(GITHUB_API_URL, headers=headers, params={"ref": BRANCH}, timeout=60)

    if r.status_code == 404:
        raise RuntimeError(
            f"GitHub returned 404 for {GITHUB_REPO}/{FILENAME} (branch '{BRANCH}'). "
            "Check the repo name, branch and file name, and that the repo is public."
        )

    if r.status_code != 200:
        try:
            raise RuntimeError(f"GitHub fetch failed ({r.status_code}): {r.json()}")
        except ValueError:
            raise RuntimeError(f"GitHub fetch failed ({r.status_code}): {r.text[:300]}")

    text = r.text
    if not _looks_like_toa5(text):
        snippet = text[:300].replace("\n", "\\n")
        raise RuntimeError("Downloaded content does not look like TOA5. " f"First 300 chars: {snippet}")
    return text


def read_units_map_from_toa5_text(toa5_text: str) -> dict[str, str]:
    lines = toa5_text.splitlines()
    if len(lines) < 4:
        raise ValueError(f"TOA5 content too short: {len(lines)} lines")

    cols = pd.read_csv(StringIO(lines[1].strip()), header=None).iloc[0].tolist()
    units = pd.read_csv(StringIO(lines[2].strip()), header=None).iloc[0].tolist()

    cols = [str(c).strip().strip('"').lstrip("﻿") for c in cols]
    units = [str(u).strip().strip('"') for u in units]
    return dict(zip(cols, units))


def read_toa5_df_from_text(toa5_text: str) -> pd.DataFrame:
    df = pd.read_csv(
        StringIO(toa5_text),
        skiprows=[0, 2, 3],
        header=0,
        na_values=["NAN", "NaN", "nan", ""],
        keep_default_na=True,
        low_memory=False,
    )
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]

    if TIME_COL not in df.columns:
        raise ValueError(f"'{TIME_COL}' not found in TOA5 data")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], format=TIMESTAMP_FORMAT, errors="coerce")
    df = df.dropna(subset=[TIME_COL])

    for c in df.columns:
        if c != TIME_COL:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Wind direction offset (e.g. sonic mounted pointing the opposite way)
    if WD_COL in df.columns and WD_OFFSET_DEG:
        df[WD_COL] = (df[WD_COL] + WD_OFFSET_DEG) % 360

    # Remove duplicate records (logger re-collection) and sort by time
    df = df.drop_duplicates(subset=[TIME_COL], keep="last").sort_values(TIME_COL).reset_index(drop=True)
    return df


# =========================
# Helpers (Dashboard)
# =========================
def format_title(var: str, units_map: dict[str, str]) -> str:
    u = units_map.get(var)
    if u is None or pd.isna(u) or str(u).strip() == "":
        return var
    return f"{var} ({str(u).strip()})"


def filter_df_by_datepicker_range(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """Inclusive filtering with FULL-DAY support when start_date == end_date."""
    if start_date is None and end_date is None:
        return df
    if start_date is not None and end_date is None:
        end_date = start_date
    if end_date is not None and start_date is None:
        start_date = end_date

    start_dt = pd.to_datetime(start_date).normalize()
    end_dt = pd.to_datetime(end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return df[(df[TIME_COL] >= start_dt) & (df[TIME_COL] <= end_dt)]


def aligned_tick0(xmin: pd.Timestamp, dtick):
    if isinstance(dtick, (int, float)):
        return xmin.normalize()
    s = str(dtick)
    if s.startswith("M"):
        return pd.Timestamp(xmin.year, xmin.month, 1)
    return xmin.normalize()


def choose_axis_settings(dff: pd.DataFrame):
    if dff.empty:
        return "M1", "%y/%m/%d"

    span_days = (dff[TIME_COL].max() - dff[TIME_COL].min()).total_seconds() / 86400
    span_months = span_days / 30.4375
    span_years = span_days / 365.25

    if span_days < 2:
        return 6 * 3600 * 1000, "%H:%M"
    if span_days <= 8:
        return 24 * 3600 * 1000, "%y/%m/%d"
    if span_days <= 15:
        return 3 * 24 * 3600 * 1000, "%y/%m/%d"
    if span_months <= 2:
        return 7 * 24 * 3600 * 1000, "%y/%m/%d"
    if span_months <= 4:
        return 14 * 24 * 3600 * 1000, "%y/%m/%d"
    if span_months <= 6:
        return "M1", "%y/%m/%d"
    if span_years < 2:
        return "M3", "%y/%m/%d"
    return "M6", "%y/%m/%d"


COMBO_COLORS = ["#1f77b4", "#8c564b", "#2ca02c", "#d62728"]


def panel_vars(panel):
    """Variables used by a panel: a column name, or (title, [(col, label), ...])."""
    return [panel] if isinstance(panel, str) else [c for c, _ in panel[1]]


def make_grid_figure(df, vars_list, units_map, title_text, dtick, tickformat) -> go.Figure:
    n_rows = max(1, math.ceil(len(vars_list) / GRID_COLS))
    n_cells = n_rows * GRID_COLS

    subplot_titles = [format_title(v, units_map) if isinstance(v, str) else v[0] for v in vars_list]
    subplot_titles += [""] * (n_cells - len(vars_list))

    fig = make_subplots(
        rows=n_rows,
        cols=GRID_COLS,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
        vertical_spacing=min(0.12, 0.35 / n_rows),
    )
    fig.update_annotations(font=dict(size=14, color="#333"))
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))

    legend_panel = None
    for i, panel in enumerate(vars_list):
        r = i // GRID_COLS + 1
        c = i % GRID_COLS + 1
        if isinstance(panel, str):
            series = [(panel, panel, None, False)]
        else:
            series = [(col, lab, COMBO_COLORS[k % len(COMBO_COLORS)], True)
                      for k, (col, lab) in enumerate(panel[1]) if col in df.columns]
            if legend_panel is None:
                legend_panel = i
        for col, lab, color, show in series:
            fig.add_trace(
                go.Scatter(
                    x=df[TIME_COL],
                    y=df[col],
                    mode="lines+markers",
                    marker=dict(size=3, color=color) if color else dict(size=3),
                    name=lab,
                    showlegend=show,
                    line=dict(color=color) if color else None,
                    hovertemplate=("%{x|%y/%m/%d %H:%M}<br>" + f"{lab}: " + "%{y}<extra></extra>"),
                ),
                row=r, col=c,
            )
            if col in Y_RANGES:
                fig.update_yaxes(range=Y_RANGES[col], row=r, col=c)

    if legend_panel is not None:
        # put the legend inside the combined panel (top-left corner)
        n = legend_panel + 1
        xd = fig.layout["xaxis" if n == 1 else f"xaxis{n}"].domain
        yd = fig.layout["yaxis" if n == 1 else f"yaxis{n}"].domain
        fig.update_layout(legend=dict(x=xd[0] + 0.005, y=yd[1] - 0.005, xanchor="left", yanchor="top",
                                      bgcolor="rgba(255,255,255,0.7)", font=dict(size=11)))

    tick0 = aligned_tick0(df[TIME_COL].min(), dtick)
    fig.update_xaxes(
        type="date", tickmode="linear", tick0=tick0, dtick=dtick,
        tickformat=tickformat, tickangle=30, showticklabels=True,
    )

    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center"),
        height=n_rows * ROW_HEIGHT_PX + 140,
        margin=dict(l=30, r=20, t=70, b=80),
    )
    return fig


# =========================
# App objects
# =========================
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server          # Render / gunicorn entry point:  gunicorn app:server
app.title = "Cattle Methane Emission Measurement"


# =========================
# Data cache (reloaded from GitHub every REFRESH_MINUTES)
# =========================
_DATA = {"df": pd.DataFrame(), "units": {}, "loaded_at": 0.0, "error": ""}
_LOCK = threading.Lock()
RETRY_SECONDS_WHEN_EMPTY = 60   # while there is no data yet, re-try GitHub every minute


def load_data(force: bool = False):
    """
    Return (df, units_map). Re-downloads from GitHub if the cache is older than REFRESH_MINUTES.
    Never raises: if GitHub can't be read, the app keeps running (with the last good data,
    or with an empty table + message) and tries again later.
    """
    with _LOCK:
        age = time.time() - _DATA["loaded_at"]
        max_age = RETRY_SECONDS_WHEN_EMPTY if _DATA["df"].empty else REFRESH_MINUTES * 60
        if force or age > max_age:
            try:
                text = fetch_toa5_text_from_github()
                _DATA["units"] = read_units_map_from_toa5_text(text)
                _DATA["df"] = read_toa5_df_from_text(text)
                _DATA["error"] = ""
                df = _DATA["df"]
                print(f"[DATA] Loaded {len(df)} records "
                      f"({df[TIME_COL].min()} -> {df[TIME_COL].max()})", flush=True)
            except Exception as e:
                _DATA["error"] = str(e)
                print(f"[DATA] Could not load data from GitHub: {e}", flush=True)
            _DATA["loaded_at"] = time.time()
        return _DATA["df"], _DATA["units"]


def no_data_message():
    return html.Div(
        style={"textAlign": "center", "marginTop": "40px", "color": "#555"},
        children=[
            html.H4("Waiting for data from GitHub"),
            html.P(f"{GITHUB_REPO} / {FILENAME} (branch {BRANCH}) could not be read yet."),
            html.P(_DATA["error"], style={"fontSize": "12px", "color": "#999"}),
            html.P(f"The app re-checks every {RETRY_SECONDS_WHEN_EMPTY} s - this page updates automatically."),
        ],
    )


# Initial load at startup (app still starts if GitHub/file isn't available yet)
load_data(force=True)

# Tabs are defined by TABS; variables missing from the file are skipped when plotting
pages = dict(TABS)
tab_names = list(pages.keys())


# =========================
# Setup tab (photos live in assets/setup/, served automatically by Dash)
# =========================
SETUP_TAB = "Site and Setup"

# Page content lives in assets/setup/setup.html (edit that file, not this code)
SETUP_PAGE = "setup/setup.html"


def setup_layout():
    return html.Iframe(
        src=app.get_asset_url(SETUP_PAGE),
        style={"width": "100%", "height": "calc(100vh - 140px)", "minHeight": "600px",
               "border": "none"},
        title="Site setup",
    )


# =========================
# Layout
# =========================
TAB_STYLE = {
    "padding": "6px 14px",
    "fontSize": "13px",
    "height": "32px",
    "lineHeight": "32px",
    "border": "1px solid #ddd",
    "borderBottom": "none",
    "borderRadius": "6px 6px 0 0",
    "backgroundColor": "#f7f7f7",
    "marginRight": "6px",
}
TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "fontWeight": "bold",
    "backgroundColor": "white",
    "borderTop": "2px solid #1f77b4",
}


QC_OPTIONS = [{"label": "All", "value": "all"}] + [
    {"label": f"<= {g}", "value": g} for g in range(1, 10)]
FLUX_CONTROLS_STYLE = {"display": "inline-flex", "alignItems": "center", "gap": "6px",
                       "marginLeft": "14px", "flexWrap": "wrap"}
WD_CONTROLS_STYLE = {"display": "inline-flex", "alignItems": "center", "gap": "6px",
                     "marginLeft": "14px"}
WD_SUMMARY_STYLE = {"cursor": "pointer", "listStyle": "none", "border": "1px solid #ccc",
                    "borderRadius": "4px", "padding": "6px 10px", "minWidth": "150px",
                    "fontSize": "14px", "backgroundColor": "white", "userSelect": "none"}
WD_PANEL_STYLE = {"position": "absolute", "top": "38px", "left": "0", "zIndex": 1000,
                  "backgroundColor": "white", "border": "1px solid #ccc", "borderRadius": "4px",
                  "boxShadow": "0 4px 12px rgba(0,0,0,0.15)", "padding": "8px 12px",
                  "minWidth": "190px"}
WD_BTN_STYLE = {"fontSize": "12px", "padding": "2px 8px", "cursor": "pointer"}


def in_sector(wd: pd.Series, sector: str) -> pd.Series:
    lo, hi = WD_SECTORS[sector]
    return (wd >= lo) | (wd < hi) if lo > hi else (wd >= lo) & (wd < hi)


def all_sectors(sectors) -> bool:
    """No selection or every box ticked = no wind-direction filter."""
    return not sectors or sectors == "all" or set(sectors) >= set(WD_SECTORS)


def filter_wd(df: pd.DataFrame, sectors) -> pd.DataFrame:
    """Blank (NaN) every value outside the selected directions, keeping timestamps so
    time-series plots show gaps instead of joining across removed periods."""
    if all_sectors(sectors) or WD_COL not in df:
        return df
    keep = pd.Series(False, index=df.index)
    for sec in sectors:
        keep |= in_sector(df[WD_COL], sec)
    d = df.copy()
    cols = [c for c in d.columns if c != TIME_COL]
    d.loc[~keep, cols] = float("nan")
    return d


def n_records(df: pd.DataFrame) -> int:
    """Records that survived the wind-direction filter."""
    return int(df.drop(columns=[TIME_COL]).notna().any(axis=1).sum())


def sectors_label(sectors) -> str:
    if all_sectors(sectors):
        return "All"
    order = [k for k in WD_SECTORS if k in sectors]
    return ", ".join(k.split()[0] for k in order)


def filter_flux_df(df: pd.DataFrame, qc_ch4, qc_c, sector) -> pd.DataFrame:
    """QC grade <= limit keeps the flux (worse grades set to NaN); sector keeps rows in that WD range."""
    d = df.copy()
    if qc_ch4 != "all" and FCH4_QC_COL in d and FCH4_COL in d:
        d.loc[~(d[FCH4_QC_COL] <= qc_ch4), FCH4_COL] = float("nan")
    if qc_c != "all" and FC_QC_COL in d and FC_COL in d:
        d.loc[~(d[FC_QC_COL] <= qc_c), FC_COL] = float("nan")
    return filter_wd(d, sector)


def _empty_fig(title, msg="Not enough data"):
    f = go.Figure()
    f.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                     showarrow=False, font=dict(color="#888"))
    f.update_layout(title=dict(text=title, x=0.5), height=460,
                    xaxis=dict(visible=False), yaxis=dict(visible=False))
    return f


def make_wind_rose(d: pd.DataFrame) -> go.Figure:
    title = f"Wind rose (WD + {WD_OFFSET_DEG}°)"
    if WD_COL not in d or WS_COL not in d:
        return _empty_fig(title)
    x = d[[WD_COL, WS_COL]].dropna()
    if x.empty:
        return _empty_fig(title)
    width = 22.5
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    sec = (((x[WD_COL] + width / 2) % 360) // width).astype(int)
    bins = [0, 1, 2, 3, 5, float("inf")]
    labels = ["0-1", "1-2", "2-3", "3-5", ">5"]
    colors = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    spd = pd.cut(x[WS_COL], bins=bins, labels=labels, right=False)
    n = len(x)
    f = go.Figure()
    for lab, col in zip(labels, colors):
        freq = [100 * ((sec == i) & (spd == lab)).sum() / n for i in range(16)]
        f.add_trace(go.Barpolar(r=freq, theta=dirs, name=lab, marker_color=col,
                                hovertemplate="%{theta}: %{r:.1f}%<extra>" + lab + " m/s</extra>"))
    f.update_layout(
        title=dict(text=title, x=0.5), height=460, margin=dict(l=60, r=60, t=60, b=70),
        legend=dict(title="Wind speed (m/s)", orientation="h", x=0.5, xanchor="center",
                    y=-0.08, yanchor="top", font=dict(size=11)),
        polar=dict(angularaxis=dict(direction="clockwise", rotation=90),
                   radialaxis=dict(ticksuffix="%", angle=45, tickfont=dict(size=9))),
    )
    return f


def make_fch4_vs_wd(d: pd.DataFrame) -> go.Figure:
    title = "FCH4 vs wind direction"
    if WD_COL not in d or FCH4_COL not in d:
        return _empty_fig(title)
    x = d[[WD_COL, FCH4_COL]].dropna()
    if x.empty:
        return _empty_fig(title)
    y = x[FCH4_COL] / M_CH4 * 1000            # ugCH4 -> nmol
    f = go.Figure()
    f.add_trace(go.Scatter(x=x[WD_COL], y=y, mode="markers", name="30-min",
                           marker=dict(size=6, color="#1f77b4", opacity=0.6),
                           hovertemplate="WD %{x:.0f}°<br>FCH4 %{y:.1f}<extra></extra>"))
    # sector means (22.5 deg)
    sec = ((x[WD_COL] // 22.5) * 22.5 + 11.25)
    m = y.groupby(sec).mean()
    f.add_trace(go.Scatter(x=m.index, y=m.values, mode="lines+markers", name="Sector mean",
                           line=dict(color="#d62728", width=2)))
    f.update_layout(
        title=dict(text=title, x=0.5), height=460, margin=dict(l=60, r=20, t=60, b=50),
        xaxis=dict(title="Wind direction (°)", range=[0, 360], tickvals=[0, 90, 180, 270, 360],
                   ticktext=["0 N", "90 E", "180 S", "270 W", "360 N"]),
        yaxis=dict(title="FCH4 (nmol m-2 s-1)"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
    )
    return f


def make_fch4_fc_regression(d: pd.DataFrame) -> go.Figure:
    title = "FCH4 vs CO2 flux (linear regression)"
    if FCH4_COL not in d or FC_COL not in d:
        return _empty_fig(title)
    x = d[[FC_COL, FCH4_COL]].dropna()
    if len(x) < 3:
        return _empty_fig(title)
    xc = x[FC_COL] / M_CO2 * 1000             # mgCO2 -> umol
    yc = x[FCH4_COL] / M_CH4 * 1000           # ugCH4 -> nmol
    slope, intercept = np.polyfit(xc, yc, 1)
    r2 = float(np.corrcoef(xc, yc)[0, 1] ** 2)
    xs = np.linspace(xc.min(), xc.max(), 50)
    f = go.Figure()
    f.add_trace(go.Scatter(x=xc, y=yc, mode="markers", name="30-min",
                           marker=dict(size=6, color="#2ca02c", opacity=0.6),
                           hovertemplate="FC %{x:.2f}<br>FCH4 %{y:.1f}<extra></extra>"))
    f.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Linear fit",
                           line=dict(color="#d62728", width=2)))
    f.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper", xanchor="left", yanchor="top",
                     showarrow=False, align="left", bgcolor="rgba(255,255,255,0.8)",
                     text=(f"y = {slope:.3f} x {'+' if intercept >= 0 else '-'} {abs(intercept):.3f}"
                           f"<br>R² = {r2:.3f}, n = {len(x)}"))
    f.update_layout(
        title=dict(text=title, x=0.5), height=460, margin=dict(l=60, r=20, t=60, b=50),
        xaxis=dict(title="FC (µmol m-2 s-1)"), yaxis=dict(title="FCH4 (nmol m-2 s-1)"),
        legend=dict(x=0.99, xanchor="right", y=0.01, yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.7)"),
    )
    return f


def serve_layout():
    """Built on every page load, so the date picker always reflects the latest data."""
    df, _ = load_data()
    if df.empty:
        min_d = max_d = None
    else:
        min_d = df[TIME_COL].min().date()
        max_d = df[TIME_COL].max().date()

    return html.Div(
        style={"fontFamily": "Arial", "padding": "10px"},
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "gap": "8px",
                },
                children=[
                    html.H3("Cattle Methane Emission Measurement using TGA310", style={"margin": "0", "textAlign": "center"}),
                    html.Div(
                        id="range-row",
                        style={
                            "display": "inline-flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "gap": "10px",
                            "flexWrap": "wrap",
                        },
                        children=[
                            html.Span("Range:", style={"fontSize": "16px"}),
                            dcc.DatePickerRange(
                                id="dp-range",
                                min_date_allowed=min_d,
                                max_date_allowed=max_d,
                                start_date=min_d,
                                end_date=max_d,
                                display_format="YYYY-MM-DD",
                                clearable=True,
                            ),
                            html.Div(id="flux-controls", style=FLUX_CONTROLS_STYLE, children=[
                                html.Span("FCH4_QC:", style={"fontSize": "14px"}),
                                dcc.Dropdown(id="qc-fch4", options=QC_OPTIONS, value="all",
                                             clearable=False, style={"width": "120px"}),
                                html.Span("FC_QC:", style={"fontSize": "14px"}),
                                dcc.Dropdown(id="qc-fc", options=QC_OPTIONS, value="all",
                                             clearable=False, style={"width": "120px"}),
                            ]),
                            html.Div(id="wd-controls", style=WD_CONTROLS_STYLE, children=[
                                html.Span("Wind direction:", style={"fontSize": "14px"}),
                                html.Details(style={"position": "relative"}, children=[
                                    html.Summary(id="wd-summary", children="All ▾",
                                                 style=WD_SUMMARY_STYLE),
                                    html.Div(style=WD_PANEL_STYLE, children=[
                                        html.Div(style={"display": "flex", "gap": "6px",
                                                        "marginBottom": "6px"}, children=[
                                            html.Button("Select all", id="wd-all", n_clicks=0,
                                                        style=WD_BTN_STYLE),
                                            html.Button("Clear", id="wd-none", n_clicks=0,
                                                        style=WD_BTN_STYLE),
                                        ]),
                                        dcc.Checklist(
                                            id="wd-sector",
                                            options=[{"label": " " + k, "value": k} for k in WD_SECTORS],
                                            value=list(WD_SECTORS),      # default: all directions
                                            labelStyle={"display": "block", "fontSize": "13px",
                                                        "padding": "2px 0", "cursor": "pointer"},
                                        ),
                                    ]),
                                ]),
                            ]),
                        ],
                    ),
                    html.Div(id="last-updated", style={"fontSize": "12px", "color": "#666"}),
                ],
            ),
            dcc.Tabs(id="tabs", value=SETUP_TAB, children=[
                dcc.Tab(label=n, value=n, style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE)
                for n in [SETUP_TAB] + tab_names
            ]),
            html.Div(id="tab-content", style={"marginTop": "8px"}),
            # Re-check GitHub while the page is open
            dcc.Interval(id="refresh", n_intervals=0,
                         interval=(RETRY_SECONDS_WHEN_EMPTY * 1000 if df.empty
                                   else REFRESH_MINUTES * 60 * 1000)),
        ],
    )


app.layout = serve_layout


# =========================
# Callbacks
# =========================
RANGE_ROW_STYLE = {
    "display": "inline-flex",
    "alignItems": "center",
    "justifyContent": "center",
    "gap": "10px",
    "flexWrap": "wrap",
}


@app.callback(
    Output("range-row", "style"),
    Output("last-updated", "style"),
    Output("flux-controls", "style"),
    Output("wd-controls", "style"),
    Input("tabs", "value"),
)
def toggle_range(tab_value):
    """Date picker hidden on Setup; QC filters on the flux tab; wind direction on flux + met tabs."""
    lu = {"fontSize": "12px", "color": "#666"}
    hide = {"display": "none"}
    fc = FLUX_CONTROLS_STYLE if tab_value == FLUX_TAB else {**FLUX_CONTROLS_STYLE, **hide}
    wd = WD_CONTROLS_STYLE if tab_value in (FLUX_TAB, MET_TAB) else {**WD_CONTROLS_STYLE, **hide}
    if tab_value == SETUP_TAB:
        return {**RANGE_ROW_STYLE, **hide}, {**lu, **hide}, fc, wd
    return RANGE_ROW_STYLE, lu, fc, wd


@app.callback(
    Output("wd-sector", "value"),
    Input("wd-all", "n_clicks"),
    Input("wd-none", "n_clicks"),
    prevent_initial_call=True,
)
def wd_select_all_none(_a, _b):
    from dash import ctx
    return list(WD_SECTORS) if ctx.triggered_id == "wd-all" else []


@app.callback(Output("wd-summary", "children"), Input("wd-sector", "value"))
def wd_summary(sectors):
    return f"{sectors_label(sectors)} ▾"


@app.callback(
    Output("dp-range", "min_date_allowed"),
    Output("dp-range", "max_date_allowed"),
    Output("dp-range", "start_date"),
    Output("dp-range", "end_date"),
    Output("last-updated", "children"),
    Output("refresh", "interval"),
    Input("refresh", "n_intervals"),
    State("dp-range", "start_date"),
    State("dp-range", "end_date"),
    State("dp-range", "max_date_allowed"),
)
def refresh_dates(_n, start_date, end_date, old_max):
    """Extend the picker when new data arrives; follow the latest day if user was viewing it."""
    df, _ = load_data()
    if df.empty:
        return (None, None, None, None, "No data yet - waiting for GitHub file",
                RETRY_SECONDS_WHEN_EMPTY * 1000)

    new_min = df[TIME_COL].min().date()
    new_max = df[TIME_COL].max().date()
    last_rec = df[TIME_COL].max().strftime("%Y-%m-%d %H:%M")

    if start_date is None:                       # data just arrived
        start_date = new_min
    if end_date is None or (old_max is not None and str(end_date)[:10] == str(old_max)[:10]):
        end_date = new_max
    return (new_min, new_max, start_date, end_date,
            f"Last record: {last_rec}  ·  refreshes every {REFRESH_MINUTES} min",
            REFRESH_MINUTES * 60 * 1000)


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("dp-range", "start_date"),
    Input("dp-range", "end_date"),
    Input("refresh", "n_intervals"),
    Input("qc-fch4", "value"),
    Input("qc-fc", "value"),
    Input("wd-sector", "value"),
)
def render_tab(tab_value, start_date, end_date, _n, qc_ch4="all", qc_c="all", sector=None):
    if tab_value == SETUP_TAB:
        return setup_layout()

    df, units_map = load_data()
    if df.empty:
        return no_data_message()

    dff = filter_df_by_datepicker_range(df, start_date, end_date)
    if dff.empty:
        return html.Div("No data for selected date range.")

    vars_list = [p for p in pages.get(tab_value, [])
                 if any(v in df.columns for v in panel_vars(p))]
    if not vars_list:
        return html.Div("No variables to display in this tab.")

    dtick, tickformat = choose_axis_settings(dff)
    s_txt = "" if start_date is None else str(start_date)
    e_txt = "" if end_date is None else str(end_date)
    title_range = f"{s_txt} → {e_txt}".strip(" →")

    if tab_value == MET_TAB:
        dfm = filter_wd(dff, sector)
        note = f"Wind direction: {sectors_label(sector)} · {n_records(dfm)} of {len(dff)} records"
        if n_records(dfm) == 0:
            return html.Div("No data for the selected wind directions. (" + note + ")",
                            style={"textAlign": "center", "marginTop": "30px"})
        fig = make_grid_figure(dfm, vars_list, units_map, title_range, dtick, tickformat)
        return html.Div([
            html.Div(note, style={"textAlign": "center", "fontSize": "12px", "color": "#666"}),
            dcc.Graph(figure=fig),
        ])

    if tab_value != FLUX_TAB:
        fig = make_grid_figure(dff, vars_list, units_map, title_range, dtick, tickformat)
        return dcc.Graph(figure=fig)

    # Flux tab: apply QC + wind-direction filters, then time series + analysis plots
    dfq = filter_flux_df(dff, qc_ch4, qc_c, sector)
    notes = [f"FCH4_QC: {'all' if qc_ch4 == 'all' else '<= ' + str(qc_ch4)}",
             f"FC_QC: {'all' if qc_c == 'all' else '<= ' + str(qc_c)}",
             f"Wind direction: {sectors_label(sector)}",
             f"{n_records(dfq)} of {len(dff)} records",
             f"valid FCH4: {int(dfq[FCH4_COL].notna().sum()) if FCH4_COL in dfq else 0}",
             f"valid FC: {int(dfq[FC_COL].notna().sum()) if FC_COL in dfq else 0}"]
    if n_records(dfq) == 0:
        return html.Div("No data for the selected filters. (" + " · ".join(notes) + ")",
                        style={"textAlign": "center", "marginTop": "30px"})

    fig = make_grid_figure(dfq, vars_list, units_map, title_range, dtick, tickformat)
    analysis_style = {"flex": "1 1 380px", "minWidth": "340px"}
    return html.Div([
        html.Div(" · ".join(notes), style={"textAlign": "center", "fontSize": "12px", "color": "#666"}),
        dcc.Graph(figure=fig),
        html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "10px"}, children=[
            html.Div(dcc.Graph(figure=make_wind_rose(dfq)), style=analysis_style),
            html.Div(dcc.Graph(figure=make_fch4_vs_wd(dfq)), style=analysis_style),
            html.Div(dcc.Graph(figure=make_fch4_fc_regression(dfq)), style=analysis_style),
        ]),
    ])


# =========================
# Local run (Render uses gunicorn instead)
# =========================
if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8050"))
    print(f"Dash app running at: http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}")
    app.server.run(host=host, port=port, debug=False, use_reloader=False)
