# -*- coding: utf-8 -*-
"""
Dash app: CSFlux (TGA310 methane EC) dashboard.

- Reads Eage_TGA310_methane_CSFlux.dat from GitHub (Prajaya2017/cattle_methane_flux, main)
- Plots ONLY main flux and meteorological variables
  (no QC flags, no SIGMA/statistics, no sample counts, no diagnostics)
- Tab "Setup" (site photos + descriptions), "Fluxes" and "Meteorology", grid of subplots, calendar date-range picker
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
FILENAME = "Eage_TGA310_methane_CSFlux.dat"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{FILENAME}"

# How often to re-download data from GitHub (minutes)
REFRESH_MINUTES = int(os.environ.get("REFRESH_MINUTES", "10"))

GRID_COLS = 5
ROW_HEIGHT_PX = 230

# Main variables only, grouped into tabs (edit to add/remove)
TABS = {
    "Fluxes": [
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
        "TA_1_1_1",      # air temperature
        "RH_1_1_1",      # relative humidity
        "T_DP_1_1_1",    # dew point
        "e_amb",         # vapor pressure
        "VPD",           # vapor pressure deficit
        "PA",            # air pressure
        "WS",            # wind speed
        "WS_MAX",        # max wind speed
        "WD",            # wind direction
        "TS_1_1_1",      # soil temperature
        "SWC_1_1_1",     # soil water content
    ],
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


def make_grid_figure(df, vars_list, units_map, title_text, dtick, tickformat) -> go.Figure:
    n_rows = max(1, math.ceil(len(vars_list) / GRID_COLS))
    n_cells = n_rows * GRID_COLS

    subplot_titles = [format_title(v, units_map) for v in vars_list]
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

    for i, v in enumerate(vars_list):
        r = i // GRID_COLS + 1
        c = i % GRID_COLS + 1
        fig.add_trace(
            go.Scatter(
                x=df[TIME_COL],
                y=df[v],
                mode="lines",
                showlegend=False,
                hovertemplate=("%{x|%y/%m/%d %H:%M}<br>" + f"{v}: " + "%{y}<extra></extra>"),
            ),
            row=r, col=c,
        )
        if v in Y_RANGES:
            fig.update_yaxes(range=Y_RANGES[v], row=r, col=c)

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
app.title = "CSFlux Dashboard"


# =========================
# Data cache (reloaded from GitHub every REFRESH_MINUTES)
# =========================
_DATA = {"df": pd.DataFrame(), "units": {}, "loaded_at": 0.0}
_LOCK = threading.Lock()


def load_data(force: bool = False):
    """Return (df, units_map). Re-downloads from GitHub if cache is older than REFRESH_MINUTES."""
    with _LOCK:
        age = time.time() - _DATA["loaded_at"]
        if force or _DATA["df"].empty or age > REFRESH_MINUTES * 60:
            try:
                text = fetch_toa5_text_from_github()
                _DATA["units"] = read_units_map_from_toa5_text(text)
                _DATA["df"] = read_toa5_df_from_text(text)
                _DATA["loaded_at"] = time.time()
                df = _DATA["df"]
                print(f"[DATA] Loaded {len(df)} records "
                      f"({df[TIME_COL].min()} -> {df[TIME_COL].max()})", flush=True)
            except Exception as e:
                # Keep serving the last good data if GitHub is temporarily unreachable
                if _DATA["df"].empty:
                    raise
                print(f"[DATA] Refresh failed, keeping previous data: {e}", flush=True)
                _DATA["loaded_at"] = time.time()   # don't retry on every request
        return _DATA["df"], _DATA["units"]


# Initial load at import time (so gunicorn fails loudly if GitHub can't be read)
_df0, _ = load_data(force=True)

# Tabs: keep only variables present in the file
pages = {}
for tab_name, vars_list in TABS.items():
    present = [v for v in vars_list if v in _df0.columns]
    missing = [v for v in vars_list if v not in _df0.columns]
    if missing:
        print(f"[WARN] {tab_name}: not in file, skipped -> {missing}")
    if present:
        pages[tab_name] = present
tab_names = list(pages.keys())


# =========================
# Setup tab (photos live in assets/setup/, served automatically by Dash)
# =========================
SETUP_TAB = "Setup"

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


def serve_layout():
    """Built on every page load, so the date picker always reflects the latest data."""
    df, _ = load_data()
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
                    html.H3("TGA310 Methane Flux - CSFlux", style={"margin": "0", "textAlign": "center"}),
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
            dcc.Interval(id="refresh", interval=REFRESH_MINUTES * 60 * 1000, n_intervals=0),
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
    Input("tabs", "value"),
)
def toggle_range(tab_value):
    """Date picker isn't used on the Setup tab."""
    hidden = tab_value == SETUP_TAB
    lu = {"fontSize": "12px", "color": "#666"}
    if hidden:
        return {**RANGE_ROW_STYLE, "display": "none"}, {**lu, "display": "none"}
    return RANGE_ROW_STYLE, lu


@app.callback(
    Output("dp-range", "max_date_allowed"),
    Output("dp-range", "end_date"),
    Output("last-updated", "children"),
    Input("refresh", "n_intervals"),
    State("dp-range", "end_date"),
    State("dp-range", "max_date_allowed"),
)
def refresh_dates(_n, end_date, old_max):
    """Extend the picker when new data arrives; follow the latest day if user was viewing it."""
    df, _ = load_data()
    new_max = df[TIME_COL].max().date()
    last_rec = df[TIME_COL].max().strftime("%Y-%m-%d %H:%M")

    if end_date is not None and old_max is not None and str(end_date)[:10] == str(old_max)[:10]:
        end_date = new_max
    return new_max, end_date, f"Last record: {last_rec}  ·  refreshes every {REFRESH_MINUTES} min"


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("dp-range", "start_date"),
    Input("dp-range", "end_date"),
    Input("refresh", "n_intervals"),
)
def render_tab(tab_value, start_date, end_date, _n):
    if tab_value == SETUP_TAB:
        return setup_layout()

    df, units_map = load_data()
    dff = filter_df_by_datepicker_range(df, start_date, end_date)
    if dff.empty:
        return html.Div("No data for selected date range.")

    vars_list = pages.get(tab_value, [])
    if not vars_list:
        return html.Div("No variables to display in this tab.")

    dtick, tickformat = choose_axis_settings(dff)
    s_txt = "" if start_date is None else str(start_date)
    e_txt = "" if end_date is None else str(end_date)
    title_range = f"{s_txt} → {e_txt}".strip(" →")

    fig = make_grid_figure(dff, vars_list, units_map, title_range, dtick, tickformat)
    return dcc.Graph(figure=fig)


# =========================
# Local run (Render uses gunicorn instead)
# =========================
if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8050"))
    print(f"Dash app running at: http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}")
    app.server.run(host=host, port=port, debug=False, use_reloader=False)
