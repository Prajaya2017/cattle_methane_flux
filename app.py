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

import re
import threading
import time
from io import StringIO
import requests
import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go


# =========================
# USER SETTINGS
# =========================
TIME_COL = "TIMESTAMP"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

GITHUB_REPO = "Prajaya2017/cattle_methane_flux"
BRANCH = "main"
FILENAME = "Cattle_Experiment_Eagle_TGA310_CSFlux.dat"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{FILENAME}"
# Direct file download: not subject to the GitHub API rate limit (60 requests/hour per IP)
GITHUB_RAW_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{BRANCH}/{FILENAME}"

# How often to re-download data from GitHub (minutes)
REFRESH_MINUTES = int(os.environ.get("REFRESH_MINUTES", "10"))

PANEL_HEIGHT_PX = 270       # height of each time-series plot (columns fill the page width)

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
LE_COL, LE_QC_COL = "LE", "LE_QC"                   # W m-2
H_COL, H_QC_COL = "H", "H_QC"                       # W m-2

USTAR_COL = "USTAR"
USTAR_OPTIONS = [{"label": "None", "value": "none"},
                 {"label": ">= 0.1 m/s", "value": 0.1},
                 {"label": ">= 0.2 m/s", "value": 0.2}]
ET_COL = "ET"

# QC dropdowns on the flux tab: (dropdown id, label, flux column, QC column)
QC_FILTERS = [
    ("qc-fch4", "FCH4_QC", FCH4_COL, FCH4_QC_COL),
    ("qc-fc", "FC_QC", FC_COL, FC_QC_COL),
    ("qc-le", "LE_QC", LE_COL, LE_QC_COL),
    ("qc-h", "H_QC", H_COL, H_QC_COL),
]
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
    """
    Download the data file from the public repo via raw.githubusercontent.com
    (no API rate limit, no login). If a GITHUB_TOKEN environment variable is set on
    Render, the authenticated API (5,000 requests/hour) is used as a fallback.
    """
    r = requests.get(GITHUB_RAW_URL, params={"t": int(time.time() // 60)},  # bust CDN cache ~1 min
                     headers={"Cache-Control": "no-cache"}, timeout=60)

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if r.status_code != 200 and token:
        r = requests.get(GITHUB_API_URL, params={"ref": BRANCH}, timeout=60,
                         headers={"Accept": "application/vnd.github.raw",
                                  "Authorization": f"Bearer {token}"})

    if r.status_code == 404:
        raise RuntimeError(
            f"GitHub returned 404 for {GITHUB_REPO}/{FILENAME} (branch '{BRANCH}'). "
            "Check the repo name, branch and file name, and that the repo is public."
        )

    if r.status_code != 200:
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


def pretty_label(text: str) -> str:
    """Plotly HTML for chemical formulas and units, e.g.
    'FCH4_mass (ugCH4 m-2 s-1)' -> 'FCH<sub>4</sub>_mass (µg CH<sub>4</sub> m<sup>−2</sup> s<sup>−1</sup>)'."""
    if not text or "<sub>" in text or "<sup>" in text:
        return text
    t = text.replace("deg C", "°C").replace("decimal degrees", "°")
    t = re.sub(r"(?<![A-Za-z])u(?=g|mol)", "µ", t)                                   # ug, umol
    t = re.sub(r"(?<![A-Za-z])(µ?g|mg|n?mol|µmol|mmol)(CH4|CO2|H2O)", r"\1 \2", t)  # ugCH4 -> ug CH4
    t = re.sub(r"CH4", "CH<sub>4</sub>", t)
    t = re.sub(r"CO2", "CO<sub>2</sub>", t)
    t = re.sub(r"H2O", "H<sub>2</sub>O", t)
    t = re.sub(r"(?<![A-Za-z0-9_])(m|s|hour|mol)(-?)(\d)(?![\dA-Za-z_])",
               lambda m: f"{m[1]}<sup>{'−' if m[2] else ''}{m[3]}</sup>", t)          # m-2 -> m<sup>−2</sup>
    return t


def fix_labels(fig: go.Figure) -> go.Figure:
    """Apply pretty_label to the title, subplot titles, axis titles and colorbar titles."""
    lay = fig.layout
    if lay.title and lay.title.text:
        lay.title.text = pretty_label(lay.title.text)
    for a in lay.annotations or []:
        if a.text:
            a.text = pretty_label(a.text)
    for name in list(lay):
        if name.startswith(("xaxis", "yaxis")) and lay[name].title and lay[name].title.text:
            lay[name].title.text = pretty_label(lay[name].title.text)
    for tr in fig.data:
        cb = getattr(tr, "colorbar", None)
        if cb is not None and cb.title and cb.title.text:
            cb.title.text = pretty_label(cb.title.text)
    return fig


def plot_box(fig: go.Figure, style=None) -> html.Div:
    """Graph with a maximize button (behaviour in assets/maximize.js)."""
    return html.Div(className="plot-box", style=style, children=[
        html.Button("⛶", className="max-btn", title="Maximize", n_clicks=0),
        dcc.Graph(figure=fix_labels(fig), config={"responsive": True}),
    ])


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


def make_panel_figure(df, panel, units_map, dtick, tickformat) -> go.Figure:
    """One time-series panel: a column name, or (title, [(col, label), ...]) for several lines."""
    if isinstance(panel, str):
        title = format_title(panel, units_map)
        series = [(panel, panel, None)]
    else:
        title = panel[0]
        series = [(col, lab, COMBO_COLORS[k % len(COMBO_COLORS)])
                  for k, (col, lab) in enumerate(panel[1]) if col in df.columns]

    fig = go.Figure()
    for col, lab, color in series:
        fig.add_trace(go.Scatter(
            x=df[TIME_COL], y=df[col], mode="lines+markers", name=lab,
            marker=dict(size=3, color=color) if color else dict(size=3),
            line=dict(color=color) if color else None,
            hovertemplate=("%{x|%y/%m/%d %H:%M}<br>" + f"{lab}: " + "%{y}<extra></extra>"),
        ))
        if col in Y_RANGES:
            fig.update_yaxes(range=Y_RANGES[col])

    fig.update_xaxes(type="date", tickmode="linear", tick0=aligned_tick0(df[TIME_COL].min(), dtick),
                     dtick=dtick, tickformat=tickformat, tickangle=30, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14, color="#333")),
        height=PANEL_HEIGHT_PX, margin=dict(l=45, r=10, t=40, b=55),
        showlegend=len(series) > 1,
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.7)", font=dict(size=11)),
    )
    return fig


def plot_grid(figs, title_text="") -> html.Div:
    """Grid of individually maximizable plots under a common heading."""
    return html.Div([
        html.H4(title_text, className="grid-title") if title_text else None,
        html.Div(className="plot-grid", children=[plot_box(f) for f in figs]),
    ])


def time_series_grid(df, vars_list, units_map, title_text, dtick, tickformat) -> html.Div:
    return plot_grid([make_panel_figure(df, p, units_map, dtick, tickformat) for p in vars_list],
                     title_text)


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


def filter_flux_df(df: pd.DataFrame, qc_limits, sector, ustar="none") -> pd.DataFrame:
    """For each flux, QC grade <= limit keeps the value (worse grades set to NaN);
    u* filter blanks all fluxes when USTAR < threshold (weak turbulence);
    sector keeps only the selected wind directions."""
    d = df.copy()
    for (_id, _lab, col, qc_col), lim in zip(QC_FILTERS, qc_limits):
        if lim not in (None, "all") and col in d and qc_col in d:
            d.loc[~(d[qc_col] <= lim), col] = float("nan")
    if ustar not in (None, "none") and USTAR_COL in d:
        flux_cols = [c for c in [q[2] for q in QC_FILTERS] + [ET_COL, "TAU", "Bowen_ratio"] if c in d]
        d.loc[~(d[USTAR_COL] >= float(ustar)), flux_cols] = float("nan")
    return filter_wd(d, sector)


def _empty_fig(title, msg="Not enough data"):
    f = go.Figure()
    f.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                     showarrow=False, font=dict(color="#888"))
    f.update_layout(title=dict(text=title, x=0.5), height=460,
                    xaxis=dict(visible=False), yaxis=dict(visible=False))
    return f


def make_wind_rose(d: pd.DataFrame) -> go.Figure:
    title = "Wind rose"
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


SECT8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _sector_index(wd: pd.Series, n: int) -> pd.Series:
    w = 360 / n
    return (((wd + w / 2) % 360) // w).astype(int)


def make_dir_hour_heatmap(d: pd.DataFrame) -> go.Figure:
    title = "FCH4 by wind direction × time of day"
    if WD_COL not in d or FCH4_COL not in d:
        return _empty_fig(title)
    x = d[[TIME_COL, WD_COL, FCH4_COL]].dropna()
    if len(x) < 3:
        return _empty_fig(title)
    x = x.assign(sec=_sector_index(x[WD_COL], 8), hr=x[TIME_COL].dt.hour,
                 f=x[FCH4_COL] / M_CH4 * 1000)
    piv = x.pivot_table(index="sec", columns="hr", values="f", aggfunc="mean").reindex(
        index=range(8), columns=range(24))
    cnt = x.pivot_table(index="sec", columns="hr", values="f", aggfunc="count").reindex(
        index=range(8), columns=range(24)).fillna(0).astype(int)
    lim = float(np.nanpercentile(np.abs(piv.values), 95)) if np.isfinite(piv.values).any() else 1
    f = go.Figure(go.Heatmap(
        z=piv.values, x=list(range(24)), y=SECT8, customdata=cnt.values,
        colorscale="RdBu_r", zmid=0, zmin=-lim, zmax=lim,
        colorbar=dict(title="nmol m-2 s-1", thickness=12),
        hovertemplate="%{y}, %{x}:00<br>FCH4 %{z:.1f}<br>n = %{customdata}<extra></extra>",
    ))
    f.update_layout(
        title=dict(text=title, x=0.5), height=460, margin=dict(l=50, r=20, t=60, b=50),
        xaxis=dict(title="Hour of day", dtick=3), yaxis=dict(title="Wind direction"),
    )
    return f


DIURNAL_VARS = [  # (column, label, factor, unit)
    (FCH4_COL, "FCH4", 1000 / M_CH4, "nmol m-2 s-1"),
    (FC_COL, "FC", 1000 / M_CO2, "µmol m-2 s-1"),
    (LE_COL, "LE", 1, "W m-2"),
    (H_COL, "H", 1, "W m-2"),
]


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def make_diurnal(d: pd.DataFrame) -> list[go.Figure]:
    """One figure per flux: mean by time of day (30-min bins) with a shaded +/- 1 SD band."""
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    hod = d[TIME_COL].dt.hour + d[TIME_COL].dt.minute / 60
    figs = []
    for i, (col, lab, k, u) in enumerate(DIURNAL_VARS):
        title = f"{lab} diurnal cycle ({u})"
        if col not in d:
            figs.append(_empty_fig(title))
            continue
        g = (d[col] * k).groupby(hod).agg(["mean", "std", "count"])
        g = g[g["count"] > 0].sort_index()
        if g.empty:
            figs.append(_empty_fig(title))
            continue
        sd = g["std"].fillna(0)
        up, dn = g["mean"] + sd, g["mean"] - sd
        c = colors[i % len(colors)]
        f = go.Figure()
        # shaded SD band: upper edge, then lower edge filled up to it
        f.add_trace(go.Scatter(x=g.index, y=up, mode="lines", line=dict(width=0),
                               hoverinfo="skip", showlegend=False))
        f.add_trace(go.Scatter(x=g.index, y=dn, mode="lines", line=dict(width=0),
                               fill="tonexty", fillcolor=_rgba(c, 0.25), name="± 1 SD",
                               hoverinfo="skip"))
        f.add_trace(go.Scatter(x=g.index, y=g["mean"], mode="lines+markers", name="Mean",
                               line=dict(color=c, width=2), marker=dict(size=4),
                               customdata=np.stack([sd, g["count"]], axis=-1),
                               hovertemplate="%{x:.1f} h<br>mean %{y:.2f}<br>SD %{customdata[0]:.2f}"
                                             "<br>n = %{customdata[1]}<extra>" + lab + "</extra>"))
        f.update_xaxes(range=[0, 24], dtick=6, title_text="Hour of day")
        f.update_layout(title=dict(text=title, x=0.5, font=dict(size=14)), height=360,
                        margin=dict(l=45, r=10, t=45, b=50),
                        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)", font=dict(size=11)))
        figs.append(f)
    return figs


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
                                *[el for qid, lab, _c, _q in QC_FILTERS for el in (
                                    html.Span(f"{lab}:", style={"fontSize": "14px"}),
                                    dcc.Dropdown(id=qid, options=QC_OPTIONS, value="all",
                                                 clearable=False, style={"width": "100px"}),
                                )],
                                html.Span("u*:", style={"fontSize": "14px", "marginLeft": "6px"}),
                                dcc.Dropdown(id="ustar-filter", options=USTAR_OPTIONS, value="none",
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
            f"Last record: {last_rec}",
            REFRESH_MINUTES * 60 * 1000)


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("dp-range", "start_date"),
    Input("dp-range", "end_date"),
    Input("refresh", "n_intervals"),
    *[Input(qid, "value") for qid, _l, _c, _q in QC_FILTERS],
    Input("wd-sector", "value"),
    Input("ustar-filter", "value"),
)
def render_tab(tab_value, start_date, end_date, _n, *args):
    nq = len(QC_FILTERS)
    qc_limits = list(args[:nq])
    sector = args[nq] if len(args) > nq else None
    ustar = args[nq + 1] if len(args) > nq + 1 else "none"
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
        return html.Div([
            html.Div(note, style={"textAlign": "center", "fontSize": "12px", "color": "#666"}),
            time_series_grid(dfm, vars_list, units_map, title_range, dtick, tickformat),
        ])

    if tab_value != FLUX_TAB:
        return time_series_grid(dff, vars_list, units_map, title_range, dtick, tickformat)

    # Flux tab: apply QC + wind-direction filters, then time series + analysis plots
    dfq = filter_flux_df(dff, qc_limits, sector, ustar)
    notes = [f"{lab}: {'all' if lim in (None, 'all') else '<= ' + str(lim)}"
             for (_i, lab, _c, _q), lim in zip(QC_FILTERS, qc_limits)] + [
             f"u*: {'none' if ustar in (None, 'none') else '>= ' + str(ustar) + ' m/s'}",
             f"Wind direction: {sectors_label(sector)}",
             f"{n_records(dfq)} of {len(dff)} records",
             "valid " + ", ".join(f"{lab.replace('_QC', '')}: {int(dfq[c].notna().sum()) if c in dfq else 0}"
                                  for _i, lab, c, _q in QC_FILTERS)]
    if n_records(dfq) == 0:
        return html.Div("No data for the selected filters. (" + " · ".join(notes) + ")",
                        style={"textAlign": "center", "marginTop": "30px"})

    analysis_style = {"flex": "1 1 380px", "minWidth": "340px"}
    return html.Div([
        html.Div(" · ".join(notes), style={"textAlign": "center", "fontSize": "12px", "color": "#666"}),
        time_series_grid(dfq, vars_list, units_map, title_range, dtick, tickformat),
        html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "10px"}, children=[
            plot_box(make_wind_rose(dfq), style=analysis_style),
            plot_box(make_fch4_vs_wd(dfq), style=analysis_style),
            plot_box(make_fch4_fc_regression(dfq), style=analysis_style),
            plot_box(make_dir_hour_heatmap(dfq), style=analysis_style),
        ]),
        plot_grid(make_diurnal(dfq), "Diurnal cycle (mean ± 1 SD)"),
    ])


# =========================
# Local run (Render uses gunicorn instead)
# =========================
if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8050"))
    print(f"Dash app running at: http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}")
    app.server.run(host=host, port=port, debug=False, use_reloader=False)
