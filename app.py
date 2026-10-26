# -*- coding: utf-8 -*-
r"""
Dash app: AmeriFlux (TGA310 methane EC) dashboard.

- Reads Cattle_Experiment_Eagle_TGA310_AmrFlux.dat (AmeriFlux-format table) from GitHub (Prajaya2017/cattle_methane_flux, main)
- Plots ONLY main flux and meteorological variables
  (no QC flags, no SIGMA/statistics, no sample counts, no diagnostics)
- Tabs "Site and Setup", "Flux and Meteorology" (fluxes, turbulence, mixing ratios and meteorology)
  (a "TGAMon" tab for the TGA310 health table is built in; set ENABLE_TGAMON = True to show it),
  grid of subplots, calendar date-range picker
- Data source: the LoggerNet files in C:\Campbellsci\LoggerNet\Data when they exist (VS Code on the
  LoggerNet PC), otherwise GitHub (Render)
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
import re as _re
import threading
import time
from io import StringIO
import requests
import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State, ALL
from urllib.parse import quote
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


# =========================
# Look and feel (one Plotly template for every figure; page styling is in assets/theme.css)
# =========================
FONT_FAMILY = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
INK, INK_MUTED, GRID = "#1f2a33", "#5f6b76", "#e9edf1"
# Categorical palette in fixed order (colour-blind checked); slot 1 is used for single-series plots
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

pio.templates["usu_flux"] = go.layout.Template(layout=go.Layout(
    font=dict(family=FONT_FAMILY, size=11, color=INK),
    title=dict(font=dict(family=FONT_FAMILY, size=13, color=INK), x=0.5, xanchor="center"),
    colorway=SERIES_COLORS,
    paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(gridcolor=GRID, linecolor="#c9d1d9", zerolinecolor="#c9d1d9", ticks="outside",
               tickcolor="#c9d1d9", ticklen=3, tickfont=dict(color=INK_MUTED), showline=True),
    yaxis=dict(gridcolor=GRID, linecolor="#c9d1d9", zerolinecolor="#c9d1d9", ticks="outside",
               tickcolor="#c9d1d9", ticklen=3, tickfont=dict(color=INK_MUTED), showline=True),
    polar=dict(bgcolor="white", angularaxis=dict(gridcolor=GRID, linecolor="#c9d1d9"),
               radialaxis=dict(gridcolor=GRID, linecolor="#c9d1d9")),
    hoverlabel=dict(bgcolor="white", bordercolor="#c9d1d9", font=dict(family=FONT_FAMILY, size=12, color=INK)),
    legend=dict(font=dict(size=11, color=INK_MUTED)),
    # vertical tool bar on the right edge, so it does not cover the plot title
    modebar=dict(orientation="v", bgcolor="rgba(255,255,255,0.7)", color="#8a95a0", activecolor="#2a78d6"),
))
pio.templates.default = "plotly_white+usu_flux"


# =========================
# USER SETTINGS
# =========================
TIME_COL = "TIMESTAMP"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

GITHUB_REPO = "Prajaya2017/cattle_methane_flux"
BRANCH = "main"
FILENAME = "Cattle_Experiment_Eagle_TGA310_AmrFlux.dat"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{FILENAME}"
# Direct file download: not subject to the GitHub API rate limit (60 requests/hour per IP)
GITHUB_RAW_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{BRANCH}/{FILENAME}"

# Local LoggerNet files: used automatically when they exist (e.g. running app.py in VS Code
# on the LoggerNet PC). On Render these paths don't exist, so the GitHub copies are used.
LOGGERNET_DATA_DIR = r"C:\Campbellsci\LoggerNet\Data"
LOCAL_FLUX_FILE = os.path.join(LOGGERNET_DATA_DIR, FILENAME)
TGAMON_FILENAME = "Cattle_Experiment_Eagle_TGA310_TGAMonitor.dat"
LOCAL_TGAMON_FILE = os.path.join(LOGGERNET_DATA_DIR, TGAMON_FILENAME)
TGAMON_GITHUB_RAW_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{BRANCH}/{TGAMON_FILENAME}"

# How often to re-download data from GitHub (minutes)
REFRESH_MINUTES = int(os.environ.get("REFRESH_MINUTES", "10"))

GRID_COLS = 5
ROW_HEIGHT_PX = 230

# Main variables only, grouped into tabs (edit to add/remove)
TABS = {
    "Flux and Meteorology": [
        "FCH4",          # CH4 flux (nmol m-2 s-1)
        "FC",            # CO2 flux (umol m-2 s-1)
        "LE",            # latent heat flux
        "H",             # sensible heat flux
        "ET",            # evapotranspiration (calculated from FH2O)
        "FH2O",          # water vapour flux
        "TAU",           # momentum flux
        "USTAR",         # friction velocity
        "TKE",           # turbulent kinetic energy (calculated from U/V/W_SIGMA)
        "Bowen_ratio",   # H / LE (calculated)
        "CH4",           # CH4 mole fraction (TGA310)
        "CO2",           # CO2 mole fraction
        "H2O",           # H2O mole fraction
        # --- meteorology (formerly its own tab) ---
        ("Air & soil temperature (deg C)",
         [("TA_1_1_1", "Air temp"), ("TS_1_1_1", "Soil temp")]),   # one plot, two lines
        "RH_1_1_1",      # relative humidity
        "T_DP_1_1_1",    # dew point (calculated from TA and RH)
        "e_amb",         # vapor pressure (calculated from TA and RH)
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
# The offset is applied only to records with TIMESTAMP at or before this time (sonic orientation was
# corrected on 2026-10-06 at 10:30; records after it are already correct). Set to None to offset all records.
WD_OFFSET_UNTIL = pd.Timestamp("2026-10-06 10:30:00")
WS_COL = "WS"

FLUX_TAB = "Flux and Meteorology"
OLD_FLUX_TABS = ("Fluxes and Turbulence",)   # earlier names; plots saved under them map here
MET_TAB = "Meteorology"            # old tab (merged into FLUX_TAB); kept so saved plots still map
TGAMON_TAB = "TGAMon"
ENABLE_TGAMON = False              # True = show the TGAMon tab (TGA310 health table)

# ---- TGAMon tab (TGA310 health / diagnostics table) ----
TGAMON_DISPLAY_OPTIONS = [{"label": "30 min", "value": "30min"},
                          {"label": "Original (10 s)", "value": "raw"}]
TGAMON_RAW_MAX_DAYS = 3
TGAMON_PLOTS_PER_ROW = 6           # plots per row on the TGAMon tab            # longer ranges switch to 30 min automatically
# Default TGAMon plots (from the TGAMonitr dashboard). Plots whose variables are not in the
# file are skipped. axis_mode: single | secondary | triple
TGAMON_PLOT_SPECS = [
    {"vars": ["CH4"], "title": "CH4", "axis_mode": "single"},
    {"vars": ["N2O"], "title": "N2O", "axis_mode": "single"},
    {"vars": ["currnt_lasr"], "title": "currnt_lasr", "axis_mode": "single"},
    {"vars": ["tmpr_lasr"], "title": "tmpr_lasr", "axis_mode": "single"},
    {"vars": ["duty_lasr_tec"], "title": "duty_lasr_tec", "axis_mode": "single"},
    {"vars": ["volt_lasr"], "title": "volt_lasr", "axis_mode": "single"},
    {"vars": ["diag_tga"], "title": "diag_tga", "axis_mode": "single"},
    {"vars": ["warng_maint_tga"], "title": "warng_maint_tga", "axis_mode": "single"},
    {"vars": ["sig_ref_det", "sig_smp_det"], "title": "Detector signals",
     "axis_mode": "secondary", "secondary_y": ["sig_smp_det"]},
    {"vars": ["tmpr_ref_det", "tmpr_smp_det"], "title": "Detector temperatures", "axis_mode": "single"},
    {"vars": ["duty_ref_det_tec", "duty_smp_det_tec"], "title": "Detector duty cycles",
     "axis_mode": "secondary", "secondary_y": ["duty_smp_det_tec"]},
    {"vars": ["trasmtnc_ref_det", "trasmtnc_smp_det"], "title": "Detector transmittance",
     "axis_mode": "secondary", "secondary_y": ["trasmtnc_smp_det"]},
    {"vars": ["tmpr_shroud_1", "tmpr_shroud_2", "tmpr_shroud_3", "tmpr_shroud_4",
              "tmpr_shroud_5", "tmpr_shroud_6", "tmpr_laser", "tmpr_det"],
     "title": "Optical bench temperatures", "axis_mode": "single"},
    {"vars": ["duty_shroud_tec_1", "duty_shroud_tec_2", "duty_shroud_tec_3", "duty_shroud_tec_4",
              "duty_shroud_tec_5", "duty_shroud_tec_6", "duty_laser_tec", "duty_det_tec"],
     "title": "TEC duty cycles", "axis_mode": "single"},
    {"vars": ["tmpr_shroud_ht_snk_1", "tmpr_shroud_ht_snk_2", "tmpr_shroud_ht_snk_3",
              "tmpr_shroud_ht_snk_4", "tmpr_shroud_ht_snk_5", "tmpr_shroud_ht_snk_6",
              "tmpr_laser_ht_snk", "tmpr_dect_ht_snk"],
     "title": "Heat sink temperatures", "axis_mode": "single"},
    {"vars": ["press_smp_flow", "press_ref_flow", "press_vrtx_bypss", "press_vrtx_thp"],
     "title": "Flow path pressures", "axis_mode": "single"},
    {"vars": ["tmpr_smp_flow", "tmpr_ref_flow", "tmpr_vrtx_bypss", "tmpr_vrtx_thp"],
     "title": "Flow path temperatures", "axis_mode": "single"},
    {"vars": ["duty_ref_valv", "duty_smp_valv"], "title": "Valve duty cycles",
     "axis_mode": "secondary", "secondary_y": ["duty_smp_valv"]},
    {"vars": ["volt_tga_pwr", "current_tga", "pwr_tga"], "title": "System voltage, current, power",
     "axis_mode": "triple", "secondary_y": ["current_tga"], "third_y": ["pwr_tga"]},
    {"vars": ["press_pmp_inlt", "press_pmp_stg_2"], "title": "Pump pressures", "axis_mode": "single"},
    {"vars": ["tmpr_pmp_1A", "tmpr_pmp_2", "tmpr_pmp_3", "tmpr_pmp_manfld", "tmpr_pmp_1B", "tmpr_pmp_inlt"],
     "title": "Pump temperatures", "axis_mode": "single"},
    {"vars": ["duty_pmp_1", "duty_pmp_2", "duty_pmp_3"], "title": "Pump duty cycles", "axis_mode": "single"},
    {"vars": ["spd_pmp_1", "spd_pmp_2", "spd_pmp_3"], "title": "Pump tachometers", "axis_mode": "single"},
    {"vars": ["press_ref_inlt"], "title": "press_ref_inlt", "axis_mode": "single"},
    {"vars": ["tmpr_ref_inlt"], "title": "tmpr_ref_inlt", "axis_mode": "single"},
    {"vars": ["duty_pump_htr", "duty_valv_htr"], "title": "Pump and valve heater duty", "axis_mode": "single"},
    {"vars": ["press_smp_cell", "press_smp_cell_thp"], "title": "Sample cell pressures",
     "axis_mode": "secondary", "secondary_y": ["press_smp_cell_thp"]},
    {"vars": ["tmpr_smp_cell", "tmpr_smp_cell_thp"], "title": "Sample cell temperatures", "axis_mode": "single"},
    {"vars": ["flow_ref", "flow_smp"], "title": "Reference and sample flow",
     "axis_mode": "secondary", "secondary_y": ["flow_smp"]},
    {"vars": ["flow_vrtx"], "title": "flow_vrtx", "axis_mode": "single"},
    {"vars": ["press_inlt_fltr", "press_outlt_fltr"], "title": "Filter pressures", "axis_mode": "single"},
    {"vars": ["tmpr_inlt_fltr", "tmpr_outlt_fltr"], "title": "Filter temperatures", "axis_mode": "single"},
    {"vars": ["diff_press_fltr"], "title": "diff_press_fltr", "axis_mode": "single"},
    {"vars": ["press_vrtx_bypss"], "title": "press_vrtx_bypss", "axis_mode": "single"},
    {"vars": ["press_tga_housng"], "title": "press_tga_housng", "axis_mode": "single"},
    {"vars": ["RH_housg"], "title": "RH_housg", "axis_mode": "single"},
    {"vars": ["RH_smp_cell_thp"], "title": "RH_smp_cell_thp", "axis_mode": "single"},
    {"vars": ["RH_vrtx_thp"], "title": "RH_vrtx_thp", "axis_mode": "single"},
]
# AmeriFlux table: fluxes are already molar; QC = SSITC test flag (0 = best, 1 = ok, 2 = bad)
FCH4_COL, FCH4_QC_COL = "FCH4", "FCH4_SSITC_TEST"  # nmolCH4 m-2 s-1
FC_COL, FC_QC_COL = "FC", "FC_SSITC_TEST"          # umolCO2 m-2 s-1
LE_COL, LE_QC_COL = "LE", "LE_SSITC_TEST"           # W m-2
H_COL, H_QC_COL = "H", "H_SSITC_TEST"               # W m-2
FH2O_QC_COL = "FH2O_SSITC_TEST"

USTAR_COL = "USTAR"
# u* filter: removes weak-turbulence half-hours (u* <= threshold)
USTAR_OPTIONS = [{"label": "None", "value": "none"},
                 {"label": "\u2264 0.1", "value": 0.1},
                 {"label": "\u2264 0.15", "value": 0.15},
                 {"label": "\u2264 0.2", "value": 0.2},
                 {"label": "\u2264 0.25", "value": 0.25}]
# Variables blanked by the u* filter (removed points are drawn in grey on their plots)
USTAR_EXTRA_COLS = ["USTAR", "TKE", "CH4", "CO2", "H2O"]
ET_COL = "ET"
CH4_CONC_COL = "CH4"                   # nmol mol-1 (ppb)
TA_COL, PA_COL = "TA_1_1_1", "PA"      # deg C, kPa

# QC dropdowns on the flux tab: (dropdown id, label, flux column, QC column)
QC_FILTERS = [
    ("qc-fch4", "FCH4_QC", FCH4_COL, FCH4_QC_COL),
    ("qc-fc", "FC_QC", FC_COL, FC_QC_COL),
    ("qc-le", "LE_QC", LE_COL, LE_QC_COL),
    ("qc-h", "H_QC", H_COL, H_QC_COL),
]
M_CH4, M_CO2 = 16.04, 44.01                        # g mol-1
# Factors from the table's flux units to the plotted units (AmeriFlux fluxes are already
# nmol CH4 / umol CO2 m-2 s-1, so no conversion is needed)
FCH4_TO_NMOL = 1.0
FC_TO_UMOL = 1.0

# Wind direction filter (degrees, after offset). N wraps around 0.
WD_SECTORS = {
    "N (337.5-22.5)": (337.5, 22.5), "NE (22.5-67.5)": (22.5, 67.5),
    "E (67.5-112.5)": (67.5, 112.5), "SE (112.5-157.5)": (112.5, 157.5),
    "S (157.5-202.5)": (157.5, 202.5), "SW (202.5-247.5)": (202.5, 247.5),
    "W (247.5-292.5)": (247.5, 292.5), "NW (292.5-337.5)": (292.5, 337.5),
}

# Optional fixed Y ranges, e.g. {"FCH4": [-50, 300]}
Y_RANGES = {}

# Full variable names, shown when the mouse hovers over a plot title
LONG_NAMES = {
    "FCH4": "Methane (CH₄) flux",
    "FC": "Carbon dioxide (CO₂) flux",
    "LE": "Latent heat flux",
    "H": "Sensible heat flux",
    "ET": "Evapotranspiration (from water vapor flux)",
    "FH2O": "Water vapor (H₂O) flux",
    "TAU": "Momentum flux (Reynolds stress)",
    "USTAR": "Friction velocity (u*)",
    "TKE": "Turbulent kinetic energy",
    "Bowen_ratio": "Bowen ratio (sensible / latent heat flux, H/LE)",
    "CH4": "Methane (CH₄) mole fraction",
    "CO2": "Carbon dioxide (CO₂) mole fraction",
    "H2O": "Water vapor (H₂O) mole fraction",
    "CH4_SIGMA": "Standard deviation of CH₄ mole fraction",
    "CO2_SIGMA": "Standard deviation of CO₂ mole fraction",
    "H2O_SIGMA": "Standard deviation of H₂O mole fraction",
    "TA_1_1_1": "Air temperature",
    "TS_1_1_1": "Soil temperature",
    "RH_1_1_1": "Relative humidity",
    "T_DP_1_1_1": "Dew point temperature",
    "e_amb": "Ambient vapor pressure",
    "VPD": "Vapor pressure deficit",
    "PA": "Atmospheric (air) pressure",
    "WS": "Wind speed (mean horizontal)",
    "WS_MAX": "Maximum wind speed",
    "WD": "Wind direction",
    "SWC_1_1_1": "Soil water content",
    "U_SIGMA": "Standard deviation of streamwise wind (u)",
    "V_SIGMA": "Standard deviation of crosswind (v)",
    "W_SIGMA": "Standard deviation of vertical wind (w)",
    "T_SONIC": "Sonic temperature",
    "T_SONIC_SIGMA": "Standard deviation of sonic temperature",
    "MO_LENGTH": "Monin–Obukhov length",
    "zL": "Stability parameter (z/L)",
    "PBLH_F": "Planetary boundary layer height",
    "FETCH_MAX": "Distance of maximum footprint contribution",
    "FETCH_70": "Fetch containing 70% of the flux footprint",
    "FETCH_80": "Fetch containing 80% of the flux footprint",
    "FETCH_90": "Fetch containing 90% of the flux footprint",
}


def long_name(var: str) -> str:
    """Full name of a variable for tooltips (the column name itself if unknown)."""
    if var in LONG_NAMES:
        return LONG_NAMES[var]
    base = _re.sub(r"(_\d+){3}$", "", var) if isinstance(var, str) else var
    for k, v in LONG_NAMES.items():
        if _re.sub(r"(_\d+){3}$", "", k) == base:
            return v
    return var


def title_tip(*vars_) -> str:
    """Tooltip text: 'FCH4 = Methane (CH4) flux' for each variable that has a full name."""
    lines = [f"{v} — {long_name(v)}" for v in vars_ if v and long_name(v) != v]
    return "\n".join(lines)


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
    # Local file first: FLUX_LOCAL_FILE (or the older CSFLUX_LOCAL_FILE) if set, else the LoggerNet file if it exists (VS Code on the
    # LoggerNet PC). On Render neither exists, so the GitHub copy is downloaded.
    local = (os.environ.get("FLUX_LOCAL_FILE", "") or os.environ.get("CSFLUX_LOCAL_FILE", "")).strip().strip('"')
    if not local and os.path.isfile(LOCAL_FLUX_FILE):
        local = LOCAL_FLUX_FILE
    if local:
        with open(local, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

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
    units = ["" if pd.isna(u) else str(u).strip().strip('"') for u in units]
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

    # Concentrations: 0 means the analyzer had no valid reading -> treat as missing
    for c in ("CH4", "CO2", "H2O"):
        if c in df.columns:
            df.loc[df[c] <= 0, c] = np.nan

    ta = df["TA_1_1_1"] if "TA_1_1_1" in df.columns else 20.0

    # Water vapour flux (only if the table doesn't have it): FH2O = LE / lambda(T) / M_H2O
    if "FH2O" not in df.columns and "LE" in df.columns:
        lam = (2.501 - 0.00237 * ta) * 1e6                    # J kg-1
        df["FH2O"] = df["LE"] / lam / 18.015 * 1e6           # mmol m-2 s-1

    # Evapotranspiration from FH2O: mmol m-2 s-1 -> mm h-1 (1 kg m-2 = 1 mm)
    if "ET" not in df.columns and "FH2O" in df.columns:
        df["ET"] = df["FH2O"] * 18.015e-6 * 3600

    # Bowen ratio H / LE (undefined when LE is ~0)
    if "Bowen_ratio" not in df.columns and {"H", "LE"} <= set(df.columns):
        le = df["LE"].where(df["LE"].abs() > 1e-6)
        df["Bowen_ratio"] = df["H"] / le

    # Turbulent kinetic energy from the wind-component standard deviations
    if "TKE" not in df.columns and {"U_SIGMA", "V_SIGMA", "W_SIGMA"} <= set(df.columns):
        df["TKE"] = 0.5 * (df["U_SIGMA"] ** 2 + df["V_SIGMA"] ** 2 + df["W_SIGMA"] ** 2)

    # Vapour pressure (kPa) and dew point (deg C) from TA and RH (Buck 1981)
    if {"TA_1_1_1", "RH_1_1_1"} <= set(df.columns):
        es = 0.61121 * np.exp(17.502 * df["TA_1_1_1"] / (240.97 + df["TA_1_1_1"]))
        rh = df["RH_1_1_1"].clip(lower=0.1, upper=100)
        if "e_amb" not in df.columns:
            df["e_amb"] = es * rh / 100
        if "T_DP_1_1_1" not in df.columns:
            x = np.log(df["e_amb"] / 0.61121)
            df["T_DP_1_1_1"] = 240.97 * x / (17.502 - x)

    # Wind direction offset (e.g. sonic mounted pointing the opposite way)
    # Only records up to WD_OFFSET_UNTIL are corrected; data collected after it are left as logged.
    if WD_COL in df.columns and WD_OFFSET_DEG:
        old = df[TIME_COL] <= WD_OFFSET_UNTIL if WD_OFFSET_UNTIL is not None else slice(None)
        df.loc[old, WD_COL] = (df.loc[old, WD_COL] + WD_OFFSET_DEG) % 360

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


COMBO_COLORS = SERIES_COLORS[:4]


def panel_vars(panel):
    """Variables used by a panel: a column name, or (title, [(col, label), ...])."""
    return [panel] if isinstance(panel, str) else [c for c, _ in panel[1]]


PANEL_HEIGHT_PX = 260

# Maximize icon
_MAX_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" fill="none">
<rect x="3" y="3" width="10" height="10" rx="0.5" stroke="#2a78d6" stroke-width="1.5"/></svg>"""   # Windows-style maximize box
MAXIMIZE_ICON_URI = "data:image/svg+xml;utf8," + quote(_MAX_SVG)

CARD_STYLE = {"minWidth": 0}     # look of the cards is in assets/theme.css (.plot-card)
MODAL_HIDDEN = {"display": "none", "position": "fixed", "top": 0, "left": 0, "width": "100%",
                "height": "100%", "backgroundColor": "rgba(15,23,32,0.55)", "zIndex": 9999,
                "justifyContent": "center", "alignItems": "center"}


import re as _re

_CHEM = [("CH4", "CH<sub>4</sub>"), ("CO2", "CO<sub>2</sub>"), ("H2O", "H<sub>2</sub>O"),
         ("N2O", "N<sub>2</sub>O")]


def fmt_units(text):
    """Scientific formatting for Plotly text: ugCH4 -> ug CH4 with subscripts,
    m-2 s-1 -> m^-2 s^-1 superscripts, deg C -> degC."""
    if not isinstance(text, str) or not text or "<sup>" in text or "<sub>" in text:
        return text
    t = text
    t = _re.sub(r"\bdecimal degrees\b", "\u00b0", t)
    t = _re.sub(r"\bdeg C\b", "\u00b0C", t)
    t = _re.sub(r"\bu(g|mol)(?=CH4|CO2|H2O|N2O)", "\u00b5\\1 ", t)      # ugCH4 -> \u00b5g CH4
    t = _re.sub(r"\b(m|n)?(g|mol)(?=CH4|CO2|H2O|N2O)", r"\1\2 ", t)      # mgCO2 -> mg CO2
    for a, b in _CHEM:
        t = _re.sub(r"(?<![A-Za-z0-9_])" + a + r"(?![A-Za-z0-9_])", b, t)
    # unit exponents: m-2, s-1, m2, hour-1, kg m-1 s-2 ...
    t = _re.sub(r"(?<![A-Za-z0-9_\-])(m|s|kg|g|hour|h|d|mol|W|J)(-?\d)(?![\d_])",
                lambda m: m.group(1) + "<sup>" + m.group(2).replace("-", "\u2212") + "</sup>", t)
    return t


def prettify_fig(fig: go.Figure) -> go.Figure:
    """Apply fmt_units to titles, axis titles, subplot titles and colorbar titles."""
    lay = fig.layout
    if lay.title and lay.title.text:
        lay.title.text = fmt_units(lay.title.text)
    for k in lay:
        if (k.startswith("xaxis") or k.startswith("yaxis")) and lay[k].title and lay[k].title.text:
            lay[k].title.text = fmt_units(lay[k].title.text)
    for ann in lay.annotations or []:
        if ann.text:
            ann.text = fmt_units(ann.text)
    if lay.legend and lay.legend.title and lay.legend.title.text:
        lay.legend.title.text = fmt_units(lay.legend.title.text)
    for tr in fig.data:
        cb = getattr(getattr(tr, "marker", None), "colorbar", None) if hasattr(tr, "marker") else None
        for c in (getattr(tr, "colorbar", None), cb):
            if c is not None and c.title and c.title.text:
                c.title.text = fmt_units(c.title.text)
        if getattr(tr, "name", None):
            tr.name = fmt_units(tr.name)
    return fig


def plot_card(fig: go.Figure, idx: int, height: int | None = None, style: dict | None = None):
    """Plot in a card with a maximize button (opens the figure in a large pop-up)."""
    fig = prettify_fig(fig)
    h = height or (fig.layout.height or PANEL_HEIGHT_PX)
    return html.Div(className="plot-card", style={**CARD_STYLE, **(style or {})}, children=[
        html.Button(html.Img(src=MAXIMIZE_ICON_URI, className="max-icon"),
                    id={"type": "max-btn", "index": idx}, n_clicks=0, title="Maximize this plot",
                    className="card-max-btn"),
        dcc.Graph(id={"type": "card-graph", "index": idx}, figure=fig,
                  style={"height": f"{h}px"},
                  config={"displaylogo": False,
                          "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}),
    ])


PANEL_COLORS = SERIES_COLORS


GAP_FACTOR = 1.5    # a jump longer than 1.5 x the normal record interval is drawn as a gap


def with_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Insert an empty (NaN) row inside every time gap so Plotly breaks the line there instead
    of drawing a straight line across missing records. The normal interval is the median spacing
    (30 min for the flux table, 10 s for raw TGAMon)."""
    if df is None or len(df) < 3 or TIME_COL not in df:
        return df
    t = df[TIME_COL]
    dt = t.diff()
    step = dt.median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        return df
    gap = dt > step * GAP_FACTOR
    if not gap.any():
        return df
    breaks = pd.DataFrame({TIME_COL: t[gap.shift(-1, fill_value=False)].values + step})
    return pd.concat([df, breaks], ignore_index=True).sort_values(TIME_COL, kind="stable").reset_index(drop=True)


def _ustar_threshold(df) -> str:
    """u* threshold used by the filter, formatted for labels ('' if unknown)."""
    v = df.attrs.get("ustar")
    if v is None and USTAR_THR_COL in df:
        s = df[USTAR_THR_COL].dropna()
        v = s.iloc[0] if not s.empty else None
    return "" if v is None else f"{float(v):g}"


def make_panel_figure(df, panel, units_map, dtick, tickformat, color=None) -> go.Figure:
    """One time-series panel: a column name, or (title, [(col, label), ...]) for several lines."""
    df = with_gaps(df)
    if isinstance(panel, str):
        title = format_title(panel, units_map)
        series = [(panel, panel, color, False)]
        tip = title_tip(panel)
    else:
        title = panel[0]
        series = [(col, lab, COMBO_COLORS[k % len(COMBO_COLORS)], True)
                  for k, (col, lab) in enumerate(panel[1]) if col in df.columns]
        tip = title_tip(*[c for c, _ in panel[1]])
    fig = go.Figure()
    for col, lab, color, show in series:
        fig.add_trace(go.Scatter(
            x=df[TIME_COL], y=df[col], mode="lines+markers",
            marker=dict(size=3, color=color) if color else dict(size=3),
            line=dict(color=color, width=1.5) if color else dict(width=1.5), name=lab, showlegend=show,
            hovertemplate=("%{x|%y/%m/%d %H:%M}<br>" + f"{lab}: " + "%{y}<extra></extra>"),
        ))
        if col in Y_RANGES:
            fig.update_yaxes(range=Y_RANGES[col])
        rm = RM_PREFIX + col
        if rm in df.columns and df[rm].notna().any():         # points removed by the u* filter
            fig.add_trace(go.Scatter(
                x=df[TIME_COL], y=df[rm], mode="markers",
                marker=dict(size=5, color="rgba(120,130,140,0.75)", symbol="x-thin",
                            line=dict(width=1.2, color="rgba(120,130,140,0.75)")),
                name=f"removed (u* \u2264 {_ustar_threshold(df)} m/s)", showlegend=True,
                hovertemplate="%{x|%y/%m/%d %H:%M}<br>removed: %{y}<extra></extra>",
            ))
    tick0 = aligned_tick0(df[TIME_COL].min(), dtick)
    fig.update_xaxes(type="date", tickmode="linear", tick0=tick0, dtick=dtick,
                     tickformat=tickformat, tickangle=30, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        height=PANEL_HEIGHT_PX, margin=dict(l=48, r=34, t=40, b=45),
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)", font=dict(size=10)),
        meta={"tip": tip},
    )
    return fig


def panel_grid(df, vars_list, units_map, dtick, tickformat, title_text, start_idx=0):
    """Grid of individual plot cards (each with its own maximize button)."""
    cards = [plot_card(make_panel_figure(df, p, units_map, dtick, tickformat,
                                         color=SERIES_COLORS[0]), start_idx + i)
             for i, p in enumerate(vars_list)]
    return html.Div(className="section", children=[
        section_header("Time series", title_text),
        html.Div(cards, className="card-grid"),
    ])


def section_header(title, sub=None):
    """Heading above a group of plots."""
    return html.Div(className="section-header", children=[
        html.H2(title, className="section-title"),
        html.Span(sub, className="section-sub") if sub else None,
    ])


# =========================
# App objects
# =========================
# Absolute path to the assets folder next to this file, so photos / setup.html / JS load
# no matter which folder the IDE (VS Code, Spyder, Jupyter) starts Python from.
try:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # interactive window / notebook: no __file__
    APP_DIR = os.getcwd()
ASSETS_DIR = os.path.join(APP_DIR, "assets")
app = Dash(__name__, suppress_callback_exceptions=True, assets_folder=ASSETS_DIR)
server = app.server          # Render / gunicorn entry point:  gunicorn app:server
app.title = "TGA310 Cattle CH4 Emission Measurement"


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
                _DATA["units"] = {"FH2O": "mmolH2O m-2 s-1", "ET": "mm hour-1", "TKE": "m2 s-2",
                                  "e_amb": "kPa", "T_DP_1_1_1": "deg C",
                                  **read_units_map_from_toa5_text(text)}
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
        className="empty-state",
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
    page_file = os.path.join(ASSETS_DIR, *SETUP_PAGE.split("/"))
    if not os.path.isfile(page_file):
        return html.Div(style={"textAlign": "center", "marginTop": "40px", "color": "#8a6508"}, children=[
            html.H4("Site and Setup page not found"),
            html.P(f"Expected: {page_file}"),
            html.P("Put the 'assets' folder (with setup/setup.html and the photos) next to app.py."),
        ])
    return html.Iframe(
        src=app.get_asset_url(SETUP_PAGE),
        style={"width": "100%", "height": "calc(100vh - 170px)", "minHeight": "600px",
               "border": "none"},
        title="Site setup",
    )


# =========================
# TGAMon data (TGA310 health table): local LoggerNet file, else GitHub
# =========================
_TGAMON = {"df": pd.DataFrame(), "units": {}, "sig": None, "loaded_at": 0.0, "error": "", "source": ""}
_TGAMON_LOCK = threading.Lock()


def _tgamon_parse_toa5(text: str):
    units = read_units_map_from_toa5_text(text)
    df = pd.read_csv(StringIO(text), skiprows=[0, 2, 3], header=0,
                     na_values=["NAN", "NaN", "nan", ""], low_memory=False)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    if TIME_COL not in df.columns:
        raise ValueError(f"'{TIME_COL}' not found in TGAMon file")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], format="ISO8601", errors="coerce")
    return df, units


def _tgamon_parse_tob3(raw: bytes, data_offset: int = 5120, meta_words: int = 3,
                       trailer_words: int = 1, nan_placeholder: float = 3.39999e38):
    """TOB3 reader from the TGAMonitr dashboard (fixed 5120-byte header, 3 meta words + 1 trailer)."""
    import csv
    head = raw[:data_offset].decode("utf-8", errors="replace").splitlines()
    rows = [next(csv.reader([ln])) for ln in head[:6]]
    rec_bytes = int(rows[1][2])
    labels = [x.strip() for x in rows[2]]
    units = [x.strip() for x in rows[3]] if len(rows) > 3 else []
    n_words = rec_bytes // 4
    n_data = n_words - meta_words - trailer_words
    labels = (labels + [f"var_{i + 1}" for i in range(len(labels), n_data)])[:n_data]
    n_rec = (len(raw) - data_offset) // rec_bytes
    body = raw[data_offset:data_offset + n_rec * rec_bytes]
    u4 = np.frombuffer(body, dtype="<u4").reshape(n_rec, n_words)
    f4 = np.frombuffer(body, dtype="<f4").reshape(n_rec, n_words)
    secs = u4[:, 0].astype(np.int64)
    data = f4[:, meta_words:n_words - trailer_words].astype(np.float64)
    ok = (secs > 0) & (~np.all(data == 0, axis=1)) if data.size else secs > 0
    secs, recnum, data = secs[ok], u4[ok, 2].astype(np.int64), data[ok]
    data[(data >= nan_placeholder) | (data <= -nan_placeholder)] = np.nan
    df = pd.DataFrame(data, columns=labels)
    df.insert(0, TIME_COL, pd.Timestamp("1990-01-01") + pd.to_timedelta(secs, unit="s"))
    df.insert(1, "RECORD", recnum)
    return df, dict(zip(labels, units[:len(labels)])) if units else {}


def _tgamon_parse(raw: bytes):
    first = raw[:64].decode("ascii", errors="ignore").lstrip().strip('"').upper()
    if first.startswith("TOB3"):
        df, units = _tgamon_parse_tob3(raw)
    else:
        df, units = _tgamon_parse_toa5(raw.decode("utf-8", errors="ignore"))
    df = df.dropna(subset=[TIME_COL])
    for c in df.columns:
        if c != TIME_COL:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(subset=[TIME_COL], keep="last").sort_values(TIME_COL).reset_index(drop=True)
    return df, units


def load_tgamon(force: bool = False):
    """Return (df, units_map, source). Local file (TGAMON_LOCAL_FILE or the LoggerNet path) is
    re-read when it changes; otherwise GitHub is tried every REFRESH_MINUTES. Never raises."""
    local = os.environ.get("TGAMON_LOCAL_FILE", "").strip().strip('"') or LOCAL_TGAMON_FILE
    with _TGAMON_LOCK:
        try:
            if os.path.isfile(local):
                st = os.stat(local)
                sig = (local, st.st_size, st.st_mtime_ns)
                if force or sig != _TGAMON["sig"]:
                    with open(local, "rb") as f:
                        raw = f.read()
                    _TGAMON["df"], _TGAMON["units"] = _tgamon_parse(raw)
                    _TGAMON.update(sig=sig, source=local, error="", loaded_at=time.time())
                    print(f"[TGAMON] Loaded {len(_TGAMON['df'])} records from {local}", flush=True)
            elif force or time.time() - _TGAMON["loaded_at"] > REFRESH_MINUTES * 60:
                _TGAMON["loaded_at"] = time.time()
                r = requests.get(TGAMON_GITHUB_RAW_URL, params={"t": int(time.time() // 60)}, timeout=60)
                if r.status_code == 200:
                    _TGAMON["df"], _TGAMON["units"] = _tgamon_parse(r.content)
                    _TGAMON.update(source="GitHub: " + TGAMON_FILENAME, error="")
                else:
                    _TGAMON["error"] = (f"{TGAMON_FILENAME} not found locally ({local}) "
                                        f"or on GitHub (HTTP {r.status_code}).")
        except Exception as e:
            _TGAMON["error"] = f"Could not read TGAMon data: {e}"
            print(f"[TGAMON] {_TGAMON['error']}", flush=True)
        return _TGAMON["df"], _TGAMON["units"], _TGAMON["source"]


def make_tgamon_figure(df, spec, units_map, dtick, tickformat) -> go.Figure:
    """Time series for one TGAMon plot spec (single / secondary / triple y-axes)."""
    df = with_gaps(df)
    vars_ = [v for v in spec["vars"] if v in df.columns]
    y2 = [v for v in spec.get("secondary_y", []) if v in vars_]
    y3 = [v for v in spec.get("third_y", []) if v in vars_ and v not in y2]
    axis_of = {v: "y3" if v in y3 else "y2" if v in y2 else "y" for v in vars_}
    fig = go.Figure()
    trace = go.Scattergl if len(df) > 5000 else go.Scatter     # WebGL only for long 10 s series
    for i, v in enumerate(vars_):
        fig.add_trace(trace(
            x=df[TIME_COL], y=df[v], mode="lines", name=v, yaxis=axis_of[v],
            line=dict(width=1.2, color=PANEL_COLORS[i % len(PANEL_COLORS)]),
            hovertemplate="%{x|%y/%m/%d %H:%M:%S}<br>" + v + ": %{y}<extra></extra>"))
    used = [a for a in ("y", "y2", "y3") if a in axis_of.values()]

    def ttl(axis):
        vs = [v for v in vars_ if axis_of[v] == axis]
        return ", ".join(format_title(v, units_map) for v in vs) if len(vs) <= 2 else ""
    lay = {"yaxis": dict(title=dict(text=ttl("y"), font=dict(size=10)), tickfont=dict(size=10))}
    if "y2" in used:
        lay["yaxis2"] = dict(title=dict(text=ttl("y2"), font=dict(size=10)), overlaying="y", side="right",
                             showgrid=False, tickfont=dict(size=10))
    if "y3" in used:
        lay["yaxis3"] = dict(title=dict(text=ttl("y3"), font=dict(size=10)), overlaying="y", side="right",
                             anchor="free", position=1.0, showgrid=False, tickfont=dict(size=9))
    fig.update_layout(**lay)
    tick0 = aligned_tick0(df[TIME_COL].min(), dtick)
    fig.update_xaxes(type="date", tickmode="linear", tick0=tick0, dtick=dtick, tickformat=tickformat,
                     tickangle=30, tickfont=dict(size=10), domain=[0, 0.78 if "y3" in used else 1])
    many = len(vars_) > 1
    fig.update_layout(
        title=dict(text=spec.get("title", vars_[0]), x=0.5, xanchor="center", font=dict(size=14, color="#333")),
        height=PANEL_HEIGHT_PX + (40 if many else 0), showlegend=many,
        margin=dict(l=45, r=45 if "y2" in used else 10, t=40, b=45),
        legend=dict(orientation="h", x=0, y=-0.35, font=dict(size=9)))
    return fig


def tgamon_frame(df, start_date, end_date, display):
    """TGAMon rows for the date range, as 30-min means or original 10 s records."""
    dff = filter_df_by_datepicker_range(df, start_date, end_date)
    note_extra = ""
    if dff.empty:
        return dff, display, note_extra
    span_days = (dff[TIME_COL].max() - dff[TIME_COL].min()).total_seconds() / 86400
    if display == "raw" and span_days > TGAMON_RAW_MAX_DAYS:
        display = "30min"
        note_extra = f" · range > {TGAMON_RAW_MAX_DAYS} days, switched to 30 min"
    if display == "30min":
        dff = dff.set_index(TIME_COL).resample("30min").mean(numeric_only=True).reset_index()
    return dff, display, note_extra


def tgamon_content(start_date, end_date, display, custom_specs):
    df, units_map, source = load_tgamon()
    if df.empty:
        return html.Div(style={"textAlign": "center", "marginTop": "40px", "color": "#555"}, children=[
            html.H4("No TGAMon data"),
            html.P(_TGAMON["error"] or "The TGAMon file has not been loaded yet."),
            html.P(f"Expected locally at {LOCAL_TGAMON_FILE} (or set TGAMON_LOCAL_FILE), "
                   f"or pushed to GitHub as {TGAMON_FILENAME}.", style={"fontSize": "12px", "color": "#999"}),
        ])
    dff, display, note_extra = tgamon_frame(df, start_date, end_date, display)
    if dff.empty:
        return html.Div("No TGAMon data for the selected date range.",
                        style={"textAlign": "center", "marginTop": "30px"})
    dtick, tickformat = choose_axis_settings(dff)
    specs = []
    covered = set()
    for sp in TGAMON_PLOT_SPECS:
        vs = [v for v in sp["vars"] if v in dff.columns]
        if vs:
            specs.append({**sp, "vars": vs}); covered.update(vs)
    if len(specs) < 5:            # variable names differ from the list: show every variable
        specs += [{"vars": [c], "title": c} for c in dff.columns
                  if c not in covered and c not in (TIME_COL, "RECORD")
                  and pd.api.types.is_numeric_dtype(dff[c]) and dff[c].notna().any()]
    cards = [plot_card(make_tgamon_figure(dff, sp, units_map, dtick, tickformat), i)
             for i, sp in enumerate(specs)]
    note = (f"Source: {source} · {len(dff):,} rows ({'30 min means' if display == '30min' else 'original 10 s'})"
            f" · last record {df[TIME_COL].max():%Y-%m-%d %H:%M:%S}{note_extra}")
    return html.Div([
        html.Div(note, style={"textAlign": "center", "fontSize": "12px", "color": "#666"}),
        html.Div(id="custom-section",
                 children=custom_section(dff, specs_for_tab(custom_specs, TGAMON_TAB), units_map, dtick, tickformat)),
        html.Div(cards, style={"display": "grid", "gap": "10px", "marginTop": "8px",
                               "gridTemplateColumns": f"repeat({TGAMON_PLOTS_PER_ROW}, minmax(0, 1fr))"}),
    ])


# =========================
# Layout
# =========================
TAB_STYLE = {"padding": "9px 20px", "fontSize": "14px", "fontWeight": "500", "color": "#5f6b76",
             "backgroundColor": "#eef2f7", "border": "1px solid #dde3ea", "borderBottom": "none",
             "borderRadius": "8px 8px 0 0", "height": "auto", "lineHeight": "20px", "marginRight": "6px"}
TAB_SELECTED_STYLE = {**TAB_STYLE, "fontWeight": "600", "color": "#1f63b8", "backgroundColor": "#f5f7fa",
                      "borderTop": "3px solid #2a78d6"}


# SSITC flag filters (AmeriFlux: 0 = best, 1 = usable, 2 = bad); value = highest flag kept
# "> 0" excludes flags 1 and 2 (keeps 0); "> 1" excludes flag 2 (keeps 0 and 1)
QC_OPTIONS = [{"label": "None", "value": "all"},
              {"label": "> 0", "value": 0},
              {"label": "> 1", "value": 1}]
QC_NOTE = {o["value"]: o["label"] + " excluded" for o in QC_OPTIONS}
# Toolbar groups: look is in assets/theme.css; these dicts only switch them on/off per tab
FLUX_CONTROLS_STYLE = {"display": "contents"}
TGAMON_CONTROLS_STYLE = {"display": "contents"}
WD_CONTROLS_STYLE = {"display": "contents"}


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


RM_PREFIX = "_ustar_removed_"
USTAR_THR_COL = "_ustar_threshold"   # carries the u* threshold to the plots (df.attrs is lost on concat)


def filter_flux_df(df: pd.DataFrame, qc_limits, sector, ustar="none") -> pd.DataFrame:
    """For each flux, QC grade <= limit keeps the value (worse grades set to NaN);
    u* filter blanks fluxes, u*, TKE and mixing ratios when USTAR <= threshold (weak turbulence);
    sector keeps only the selected wind directions."""
    d = df.copy()
    for (_id, _lab, col, qc_col), lim in zip(QC_FILTERS, qc_limits):
        if lim not in (None, "all") and col in d and qc_col in d:
            cols = [c for c in ([col, "FH2O", ET_COL] if col == LE_COL else [col]) if c in d]
            d.loc[~(d[qc_col] <= lim), cols] = float("nan")
    if ustar not in (None, "none") and USTAR_COL in d:
        cols = [c for c in [q[2] for q in QC_FILTERS] + [ET_COL, "FH2O", "TAU", "Bowen_ratio"]
                + USTAR_EXTRA_COLS if c in d]
        low = d[USTAR_COL] <= float(ustar)                 # weak turbulence -> removed
        for c in cols:                                      # keep removed values to show in grey
            d[RM_PREFIX + c] = d[c].where(low)
        d.loc[low | d[USTAR_COL].isna(), cols] = float("nan")
        d.attrs["ustar"] = float(ustar)
        d[USTAR_THR_COL] = float(ustar)
    return filter_wd(d, sector)


def _empty_fig(title, msg="Not enough data"):
    f = go.Figure()
    f.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                     showarrow=False, font=dict(color="#888"))
    f.update_layout(title=dict(text=title, x=0.5), height=460,
                    xaxis=dict(visible=False), yaxis=dict(visible=False))
    return f


ROSE_SECTORS = 36                                   # 10-degree petals
ROSE_WS_BINS = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, np.inf]


def make_wind_rose(d: pd.DataFrame) -> go.Figure:
    """Colourful wind rose: 10-degree sectors, stacked by wind-speed class (Viridis, calm = dark),
    radius = % of half-hours."""
    title = "Wind rose"
    if WD_COL not in d or WS_COL not in d:
        return _empty_fig(title)
    x = d[[WD_COL, WS_COL]].dropna()
    if x.empty:
        return _empty_fig(title)
    from plotly.colors import sample_colorscale
    w = 360 / ROSE_SECTORS
    sec = (((x[WD_COL] + w / 2) % 360) // w).astype(int)
    edges = ROSE_WS_BINS
    used = int(np.searchsorted(edges, x[WS_COL].max(), side="right"))   # drop empty top classes
    edges = edges[:max(used, 2) + 1] if used < len(edges) - 1 else edges
    nb = len(edges) - 1
    labels = [f"{edges[i]:g}\u2013{edges[i + 1]:g}" if np.isfinite(edges[i + 1]) else f"> {edges[i]:g}"
              for i in range(nb)]
    cols = sample_colorscale("Viridis", list(np.linspace(0.05, 1.0, nb)))
    spd = pd.cut(x[WS_COL], edges, labels=False, right=False)
    n = len(x)
    theta = np.arange(ROSE_SECTORS) * w
    f = go.Figure()
    for i, (lab, col) in enumerate(zip(labels, cols)):
        freq = np.bincount(sec[spd == i], minlength=ROSE_SECTORS)[:ROSE_SECTORS] / n * 100
        f.add_trace(go.Barpolar(r=freq, theta=theta, width=w * 0.92, name=f"{lab} m/s", marker_color=col,
                                marker_line=dict(color="rgba(30,30,40,0.55)", width=0.6),
                                hovertemplate="%{theta}\u00b0: %{r:.2f}%<extra>" + lab + " m/s</extra>"))
    # discrete colour bar for wind speed (invisible carrier trace)
    ticks = list(range(nb))
    cs = []
    for i, c in enumerate(cols):
        cs += [[i / nb, c], [(i + 1) / nb, c]]
    f.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                           marker=dict(color=[0], cmin=-0.5, cmax=nb - 0.5, colorscale=cs, showscale=True,
                                       colorbar=dict(title="Wind speed<br>(m s\u207b\u00b9)", thickness=14,
                                                     len=0.75, tickvals=ticks, ticktext=labels,
                                                     tickfont=dict(size=10)))))
    f.update_layout(
        title=dict(text=f"{title} \u00b7 n = {n}", x=0.5), height=MAP_HEIGHT_PX, showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40), barmode="stack",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        meta={"tip": "Wind rose: direction the wind comes from (10\u00b0 sectors); petal length = % of "
                     "half-hours; colour = wind-speed class\n" + title_tip(WD_COL, WS_COL)},
        polar=dict(bgcolor="white",
                   angularaxis=dict(direction="clockwise", rotation=90, tickmode="array",
                                    tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                                    ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                                    tickfont=dict(size=13, color="#1f2a33"), gridcolor="#d9dee4"),
                   radialaxis=dict(ticksuffix="%", angle=67.5, tickangle=67.5, nticks=5,
                                   tickfont=dict(size=10, color="#c0392b"), gridcolor="#d9dee4")),
    )
    return f


def make_fch4_vs_wd(d: pd.DataFrame) -> go.Figure:
    title = "FCH4 vs wind direction"
    if WD_COL not in d or FCH4_COL not in d:
        return _empty_fig(title)
    x = d[[WD_COL, FCH4_COL]].dropna()
    if x.empty:
        return _empty_fig(title)
    y = x[FCH4_COL] * FCH4_TO_NMOL            # nmol m-2 s-1
    f = go.Figure()
    f.add_trace(go.Scatter(x=x[WD_COL], y=y, mode="markers", name="30-min",
                           marker=dict(size=6, color=SERIES_COLORS[0], opacity=0.55),
                           hovertemplate="WD %{x:.0f}°<br>FCH4 %{y:.1f}<extra></extra>"))
    # sector means (22.5 deg)
    sec = ((x[WD_COL] // 22.5) * 22.5 + 11.25)
    m = y.groupby(sec).mean()
    f.add_trace(go.Scatter(x=m.index, y=m.values, mode="lines+markers", name="Sector mean",
                           line=dict(color=SERIES_COLORS[1], width=2), marker=dict(size=8)))
    f.update_layout(
        title=dict(text=title, x=0.5), height=460, margin=dict(l=60, r=40, t=60, b=50),
        meta={"tip": title_tip(FCH4_COL, WD_COL)},
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
    xc = x[FC_COL] * FC_TO_UMOL               # umol m-2 s-1
    yc = x[FCH4_COL] * FCH4_TO_NMOL           # nmol m-2 s-1
    slope, intercept = np.polyfit(xc, yc, 1)
    r2 = float(np.corrcoef(xc, yc)[0, 1] ** 2)
    xs = np.linspace(xc.min(), xc.max(), 50)
    f = go.Figure()
    f.add_trace(go.Scatter(x=xc, y=yc, mode="markers", name="30-min",
                           marker=dict(size=6, color=SERIES_COLORS[0], opacity=0.55),
                           hovertemplate="FC %{x:.2f}<br>FCH4 %{y:.1f}<extra></extra>"))
    f.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Linear fit",
                           line=dict(color=SERIES_COLORS[1], width=2)))
    f.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper", xanchor="left", yanchor="top",
                     showarrow=False, align="left", bgcolor="rgba(255,255,255,0.8)",
                     text=(f"y = {slope:.3f} x {'+' if intercept >= 0 else '-'} {abs(intercept):.3f}"
                           f"<br>R² = {r2:.3f}, n = {len(x)}"))
    f.update_layout(
        title=dict(text=title, x=0.5), height=460, margin=dict(l=60, r=40, t=60, b=50),
        xaxis=dict(title="FC (µmol m-2 s-1)"), yaxis=dict(title="FCH4 (nmol m-2 s-1)"),
        meta={"tip": title_tip(FCH4_COL, FC_COL)},
        legend=dict(x=0.99, xanchor="right", y=0.01, yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.7)"),
    )
    return f


SECT16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
          "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
SECT8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _sector_index(wd: pd.Series, n: int) -> pd.Series:
    w = 360 / n
    return (((wd + w / 2) % 360) // w).astype(int)


# =========================
# Site maps: wind rose and flux footprint drawn on satellite imagery around the EC tower
# =========================
import base64
import struct
import zlib

TOWER_LAT, TOWER_LON = 41.764950, -111.870001
MAP_HEIGHT_PX = 540
# Satellite tiles (Esri World Imagery, no key needed); loaded by the viewer's browser
SAT_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
SAT_ATTRIBUTION = "Imagery © Esri, Maxar, Earthstar Geographics"
WS_BINS = [0, 1, 2, 3, 5, np.inf]
WS_LABELS = ["0–1", "1–2", "2–3", "3–5", "> 5"]
WS_COLORS = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
M_PER_DEG_LAT = 111320.0
M_PER_DEG_LON = 111320.0 * math.cos(math.radians(TOWER_LAT))


def _to_lonlat(x_east, y_north):
    """Metres east / north of the tower -> (lon, lat)."""
    return (TOWER_LON + np.asarray(x_east, dtype=float) / M_PER_DEG_LON,
            TOWER_LAT + np.asarray(y_north, dtype=float) / M_PER_DEG_LAT)


def _zoom_for(radius_m, px=MAP_HEIGHT_PX):
    """Map zoom that shows about +/- radius_m around the tower."""
    m_per_px = 2.0 * radius_m / (px - 70)                 # plot area is the card minus the title
    # MapLibre uses 512-px tiles: metres/pixel at zoom z = 78271.52 * cos(lat) / 2**z
    return float(np.clip(math.log2(78271.52 * math.cos(math.radians(TOWER_LAT)) / m_per_px), 10, 19))


def _map_layout(fig, title, radius_m, tip):
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=MAP_HEIGHT_PX, margin=dict(l=10, r=10, t=50, b=10), meta={"tip": tip},
        map=dict(style="white-bg", center=dict(lat=TOWER_LAT, lon=TOWER_LON), zoom=_zoom_for(radius_m),
                 layers=[dict(below="traces", sourcetype="raster", source=[SAT_TILES],
                              sourceattribution=SAT_ATTRIBUTION)] + list(fig.layout.map.layers or [])),
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.85)",
                    font=dict(size=11)),
    )
    return fig


def _tower_trace():
    return go.Scattermap(lat=[TOWER_LAT], lon=[TOWER_LON], mode="markers", name="EC tower",
                         marker=dict(size=13, color="#e34948"),
                         hovertemplate=f"EC tower<br>{TOWER_LAT:.6f}, {TOWER_LON:.6f}<extra></extra>")


def _ring(r_m, n=90):
    th = np.linspace(0, 2 * np.pi, n)
    return _to_lonlat(r_m * np.sin(th), r_m * np.cos(th))


def make_wind_rose_map(d: pd.DataFrame) -> go.Figure:
    """Wind rose (16 sectors x wind-speed classes) drawn on satellite imagery, centred on the tower.
    Petal length = frequency (%), scaled so the longest petal reaches ROSE_RADIUS_M."""
    title = "Wind rose on site map"
    x = d[[WD_COL, WS_COL]].dropna() if {WD_COL, WS_COL} <= set(d.columns) else pd.DataFrame()
    fig = go.Figure()
    if x.empty:
        fig.add_trace(_tower_trace())
        return _map_layout(fig, title + " (no wind data)", 250, title_tip(WD_COL, WS_COL))
    sec = _sector_index(x[WD_COL], 16).to_numpy()
    spd = pd.cut(x[WS_COL], WS_BINS, labels=False, right=False).to_numpy()
    n = len(x)
    freq = np.array([[100.0 * np.sum((sec == s) & (spd == b)) / n for b in range(len(WS_LABELS))]
                     for s in range(16)])                       # (sector, speed class) in %
    total = freq.sum(axis=1)
    rose_r = 220.0                                               # metres for the longest petal
    m_per_pct = rose_r / max(total.max(), 1e-9)
    half = np.deg2rad(22.5 / 2 - 1.2)                           # small gap between petals
    cum = np.zeros(16)
    for b, (lab, col) in enumerate(zip(WS_LABELS, WS_COLORS)):
        lons, lats, hx, hy, ht = [], [], [], [], []
        for s in range(16):
            r0, r1 = cum[s] * m_per_pct, (cum[s] + freq[s, b]) * m_per_pct
            if freq[s, b] <= 0:
                continue
            c = np.deg2rad(s * 22.5)
            a = np.linspace(c - half, c + half, 8)
            xs = np.r_[r0 * np.sin(a), r1 * np.sin(a[::-1]), r0 * np.sin(a[0])]
            ys = np.r_[r0 * np.cos(a), r1 * np.cos(a[::-1]), r0 * np.cos(a[0])]
            lo, la = _to_lonlat(xs, ys)
            lons += list(lo) + [None]; lats += list(la) + [None]
            mx, my = _to_lonlat((r0 + r1) / 2 * np.sin(c), (r0 + r1) / 2 * np.cos(c))
            hx.append(mx); hy.append(my)
            ht.append(f"{SECT16[s]} · {lab} m/s<br>{freq[s, b]:.1f}% of half-hours")
        cum += freq[:, b]
        if not lons:
            continue
        fig.add_trace(go.Scattermap(lon=lons, lat=lats, mode="lines", fill="toself", fillcolor=col,
                                    line=dict(color="rgba(255,255,255,0.9)", width=1), name=f"{lab} m/s",
                                    legendgroup=lab, hoverinfo="skip"))
        fig.add_trace(go.Scattermap(lon=hx, lat=hy, mode="markers", marker=dict(size=10, opacity=0),
                                    legendgroup=lab, showlegend=False, text=ht,
                                    hovertemplate="%{text}<extra></extra>"))
    # frequency rings with % labels
    step = next(s for s in (1, 2, 5, 10, 20, 25, 50) if total.max() / s <= 4)
    for p in np.arange(step, total.max() + 1e-9, step):
        lo, la = _ring(p * m_per_pct)
        fig.add_trace(go.Scattermap(lon=lo, lat=la, mode="lines", hoverinfo="skip", showlegend=False,
                                    line=dict(color="rgba(255,255,255,0.75)", width=1)))
        tx, ty = _to_lonlat(p * m_per_pct * math.sin(math.radians(30)), p * m_per_pct * math.cos(math.radians(30)))
        fig.add_trace(go.Scattermap(lon=[tx], lat=[ty], mode="text", text=[f"{p:g}%"], hoverinfo="skip",
                                    showlegend=False, textfont=dict(size=11, color="white")))
    fig.add_trace(_tower_trace())
    return _map_layout(fig, f"Wind rose on site map · n = {n}", rose_r * 1.15,
                       "Wind rose on satellite map, centred on the EC tower\n"
                       "Petal direction = where the wind comes from; length = % of half-hours; "
                       "colour = wind-speed class\n" + title_tip(WD_COL, WS_COL))


# ---- Approximate 2-D flux footprint for each half-hour, from FETCH_MAX / FETCH_90 ----
# Along the wind: gamma distribution with its peak at FETCH_MAX and 90% of the footprint within
# FETCH_90. Across the wind: Gaussian spread sigma_y = (V_SIGMA / WS) * distance.
# (An approximation of the source area; a full model such as Kljun et al. 2015 also needs the
# measurement height and roughness length.)
FP_N = 161                                     # grid cells per side

def _gamma_ratio_table():
    """For gamma shape k with mode 1: ratio of the 90th percentile to the mode."""
    ks = np.linspace(1.03, 40, 400)
    xs = np.linspace(0, 400, 40001)
    out = []
    for k in ks:
        th = 1.0 / (k - 1)
        logpdf = (k - 1) * np.log(np.maximum(xs, 1e-300)) - xs / th - math.lgamma(k) - k * math.log(th)
        cdf = np.cumsum(np.exp(logpdf)) * (xs[1] - xs[0])
        out.append(xs[min(np.searchsorted(cdf / cdf[-1], 0.9), len(xs) - 1)])
    return ks, np.array(out)

_GK, _GR = _gamma_ratio_table()               # ratio decreases as k increases
_FP_CACHE = {}


def _footprints(x: pd.DataFrame, ext: float):
    """Sum over half-hours of the normalised 2-D footprints (climatology), plus the sums needed for
    the flux-weighted source map, on a grid of +/- ext metres. Only running sums are kept in memory."""
    g = np.linspace(-ext, ext, FP_N)
    GX, GY = np.meshgrid(g, g)
    xmax = np.maximum(x["FETCH_MAX"].to_numpy(float), 0.5)
    x90 = np.maximum(x["FETCH_90"].to_numpy(float), xmax * 1.3)
    k = np.interp(x90 / xmax, _GR[::-1], _GK[::-1])               # shape from the 90%/peak ratio
    th = xmax / (k - 1)
    a = np.deg2rad(x[WD_COL].to_numpy(float))
    spread = np.clip(x["V_SIGMA"].to_numpy(float) / np.maximum(x[WS_COL].to_numpy(float), 0.3), 0.05, 1.0)
    fch4 = x[FCH4_COL].to_numpy(float)
    clim = np.zeros((FP_N, FP_N)); W = np.zeros((FP_N, FP_N)); S = np.zeros((FP_N, FP_N))
    for i in range(len(x)):
        ux, uy = math.sin(a[i]), math.cos(a[i])                 # towards the upwind (source) side
        xu = GX * ux + GY * uy
        yc = -GX * uy + GY * ux
        xp = np.clip(xu, 1e-3, None)
        sig = spread[i] * xp
        logf = ((k[i] - 1) * np.log(xp) - xp / th[i] - math.lgamma(k[i]) - k[i] * math.log(th[i])
                - 0.5 * (yc / sig) ** 2 - np.log(sig))
        f = np.where(xu > 0, np.exp(logf), 0.0)
        s = f.sum()
        if s <= 0:
            continue
        f /= s
        clim += f
        if np.isfinite(fch4[i]):                                  # flux-weighted source map
            W += f
            S += f * float(np.clip(fch4[i], -2000, 2000))
    return g, clim, W, S


def _level_for(f, frac):
    v = np.sort(f.ravel())[::-1]
    c = np.cumsum(v)
    return v[min(np.searchsorted(c, frac * c[-1]), len(v) - 1)]


def _outline(g, f, frac, n_ang=144):
    """Star-shaped outline (seen from the tower) of the area holding `frac` of footprint f."""
    lev = _level_for(f, frac)
    inside = f >= lev
    GX, GY = np.meshgrid(g, g)
    ang = (np.degrees(np.arctan2(GX, GY)) + 360) % 360
    rad = np.hypot(GX, GY)
    b = (ang // (360 / n_ang)).astype(int)
    rmax = np.zeros(n_ang)
    np.maximum.at(rmax, b[inside], rad[inside])
    rmax = np.where(rmax > 0, rmax, np.nan)
    rmax = pd.Series(rmax).interpolate(limit_direction="both")              # fill empty bins
    rmax = pd.concat([rmax.iloc[-3:], rmax, rmax.iloc[:3]]).rolling(5, center=True).median().iloc[3:-3]
    rmax = rmax.to_numpy()                                                    # smooth (wraps around N)
    th = np.deg2rad((np.arange(n_ang) + 0.5) * 360 / n_ang)
    xs, ys = np.r_[rmax * np.sin(th), rmax[0] * np.sin(th[0])], np.r_[rmax * np.cos(th), rmax[0] * np.cos(th[0])]
    return _to_lonlat(xs, ys), float(np.nanmax(rmax))


def _png_rgba(rgba: np.ndarray) -> str:
    """Tiny PNG encoder (no Pillow / matplotlib needed) -> data URI."""
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[r].tobytes() for r in range(h))
    def chunk(t, data):
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _rgb_table(scale, n=256):
    from plotly.colors import sample_colorscale, unlabel_rgb
    cols = sample_colorscale(scale, list(np.linspace(0, 1, n)))
    return np.array([unlabel_rgb(c) for c in cols], dtype=np.uint8)


def make_footprint_map(d: pd.DataFrame) -> go.Figure:
    """Flux footprint climatology (areas holding 50, 80 and 90% of the footprint of the selected
    half-hours) on satellite imagery around the tower."""
    need = [WD_COL, WS_COL, "V_SIGMA", "FETCH_MAX", "FETCH_90"]
    title = "Flux footprint (90%)"
    tip = ("Flux footprint climatology of the selected half-hours on the site map\n"
           "Yellow = area supplying 90% of the measured flux; white = 80% and 50%\n"
           "Approximate footprints from FETCH_MAX, FETCH_90, WD, WS and V_SIGMA\n"
           + title_tip("FETCH_90", "FETCH_MAX", WD_COL))
    fig = go.Figure()
    if not set(need) <= set(d.columns):
        fig.add_trace(_tower_trace())
        return _map_layout(fig, title + " (footprint columns missing)", 250, tip)
    x = d[need].dropna()
    x = x[(x["FETCH_90"] > 0) & (x["FETCH_MAX"] > 0)].assign(**{FCH4_COL: np.nan})
    if len(x) < 3:
        fig.add_trace(_tower_trace())
        return _map_layout(fig, title + " (not enough data)", 250, tip)
    ext = float(np.clip(np.nanpercentile(x["FETCH_90"], 99) * 1.4, 100, 1200))
    key = (len(x), round(ext), pd.util.hash_pandas_object(x[need], index=False).sum())
    if key not in _FP_CACHE:
        if len(_FP_CACHE) > 12:
            _FP_CACHE.clear()
        _FP_CACHE[key] = _footprints(x, ext)
    g, clim, _W, _S = _FP_CACHE[key]
    clim = clim / clim.sum()
    r90 = 0.0
    for frac, width, col, fill in ((0.9, 2.6, "#ffd23f", "rgba(255,210,63,0.18)"),
                                   (0.8, 1.6, "rgba(255,255,255,0.95)", None),
                                   (0.5, 1.4, "rgba(255,255,255,0.8)", None)):
        (lo, la), r = _outline(g, clim, frac)
        r90 = max(r90, r)
        fig.add_trace(go.Scattermap(lon=lo, lat=la, mode="lines", name=f"{int(frac * 100)}% footprint",
                                    line=dict(color=col, width=width), hoverinfo="skip",
                                    fill="toself" if fill else None, fillcolor=fill))
    fig.add_trace(_tower_trace())
    return _map_layout(fig, f"{title} \u00b7 n = {len(x)}", max(r90, 60) * 1.1, tip)


SOURCE_FETCH_COL = "FETCH_90"        # footprint distance used on the source-direction map


def make_source_map(d: pd.DataFrame) -> go.Figure:
    """Flux footprint and source direction on the satellite map: each half-hour is placed upwind of the
    tower (wind direction) at its 90% footprint distance (FETCH_90) and coloured by FCH4; the dotted line
    is the median 90% fetch per 22.5 deg sector. Follows the dashboard's QC / u* / wind filters."""
    pct = SOURCE_FETCH_COL.split("_")[-1]
    title = "Flux footprint and source direction"
    tip = (f"Each dot is a half-hour, drawn upwind of the tower at the distance holding {pct}% of the "
           f"flux footprint ({SOURCE_FETCH_COL}); colour = CH\u2084 flux\n"
           f"Dotted line = median {pct}% fetch per wind sector\n" + title_tip(FCH4_COL, SOURCE_FETCH_COL, WD_COL))
    fig = go.Figure()
    if not {WD_COL, SOURCE_FETCH_COL, FCH4_COL} <= set(d.columns):
        fig.add_trace(_tower_trace())
        return _map_layout(fig, title + " (columns missing)", 250, tip)
    x = d[[TIME_COL, WD_COL, SOURCE_FETCH_COL, FCH4_COL]].dropna()
    x = x[x[SOURCE_FETCH_COL] > 0]
    if len(x) < 3:
        fig.add_trace(_tower_trace())
        return _map_layout(fig, title + " (not enough data)", 250, tip)
    a = np.deg2rad(x[WD_COL].to_numpy(float) % 360)
    r = x[SOURCE_FETCH_COL].to_numpy(float)
    lon, lat = _to_lonlat(r * np.sin(a), r * np.cos(a))
    # median distance per sector (closed dotted line)
    sec = _sector_index(x[WD_COL], 16)
    m = x[SOURCE_FETCH_COL].groupby(sec).median().reindex(range(16))
    if m.notna().sum() >= 2:
        m = m.interpolate(limit_direction="both").to_numpy()
        th = np.deg2rad(np.arange(17) * 22.5)
        mr = np.append(m, m[0])
        # dotted black line: Scattermap lines can't be dashed, so draw small dots along the path
        xs_, ys_ = mr * np.sin(th), mr * np.cos(th)
        seg = np.hypot(np.diff(xs_), np.diff(ys_))
        step = max(float(np.nanmax(mr)) / 28, 1.0)            # dot spacing in metres
        px, py = [], []
        for i in range(16):
            k = max(int(seg[i] // step), 1)
            t = np.arange(k) / k
            px += list(xs_[i] + t * (xs_[i + 1] - xs_[i])); py += list(ys_[i] + t * (ys_[i + 1] - ys_[i]))
        mlo, mla = _to_lonlat(px, py)
        fig.add_trace(go.Scattermap(lon=mlo, lat=mla, mode="markers", name=f"median {pct}% fetch",
                                    marker=dict(size=4, color="#1f2a33"), hoverinfo="skip"))
    # distance rings (like a polar plot): thin circles, distance labels, compass spokes and labels
    rview = float(np.nanpercentile(r, 95))
    step = next(v for v in (10, 20, 25, 50, 100, 200, 250, 500) if rview / v <= 4.5)
    rings = np.arange(step, rview + step * 0.999, step)
    ring_lo, ring_la = [], []
    for rr in rings:
        lo_, la_ = _ring(rr)
        ring_lo += list(lo_) + [None]; ring_la += list(la_) + [None]
    rout = rings[-1]
    for ang in range(0, 360, 45):                                  # spokes
        sx, sy = _to_lonlat([0, rout * math.sin(math.radians(ang))], [0, rout * math.cos(math.radians(ang))])
        ring_lo += list(sx) + [None]; ring_la += list(sy) + [None]
    fig.add_trace(go.Scattermap(lon=ring_lo, lat=ring_la, mode="lines", hoverinfo="skip", showlegend=False,
                                line=dict(color="rgba(255,255,255,0.3)", width=1)))      # faint grid
    olo, ola = _ring(rout)
    fig.add_trace(go.Scattermap(lon=olo, lat=ola, mode="lines", hoverinfo="skip", showlegend=False,
                                line=dict(color="rgba(255,255,255,0.55)", width=1.2)))   # outer ring
    tl = math.radians(67.5)
    lx, ly = _to_lonlat(rings * math.sin(tl), rings * math.cos(tl))
    fig.add_trace(go.Scattermap(lon=lx, lat=ly, mode="text", text=[f"{v:g} m" for v in rings],
                                hoverinfo="skip", showlegend=False,
                                textfont=dict(size=11, color="rgba(255,255,255,0.85)")))
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    cx, cy = _to_lonlat([rout * 1.08 * math.sin(math.radians(a)) for a in range(0, 360, 45)],
                        [rout * 1.08 * math.cos(math.radians(a)) for a in range(0, 360, 45)])
    fig.add_trace(go.Scattermap(lon=cx, lat=cy, mode="text", text=names, hoverinfo="skip", showlegend=False,
                                textfont=dict(size=13, color="white")))

    lim = float(np.nanpercentile(np.abs(x[FCH4_COL]), 95)) or 1.0
    order = np.argsort(np.abs(x[FCH4_COL].to_numpy()))           # strongest fluxes drawn on top
    xs = x.iloc[order]
    lon, lat = lon[order], lat[order]
    fig.add_trace(go.Scattermap(
        lon=lon, lat=lat, mode="markers", name="half-hours (colour = FCH4)",
        marker=dict(size=8, color=xs[FCH4_COL].clip(-lim, lim), colorscale="RdBu_r", cmid=0,
                    cmin=-lim, cmax=lim, opacity=0.85,
                    colorbar=dict(title="FCH4<br>nmol m\u207b\u00b2 s\u207b\u00b9", thickness=12, len=0.6)),
        customdata=np.c_[xs[FCH4_COL], xs[WD_COL], xs[SOURCE_FETCH_COL], xs[TIME_COL].dt.strftime("%y/%m/%d %H:%M")],
        hovertemplate="%{customdata[3]}<br>WD %{customdata[1]:.0f}\u00b0 \u00b7 " + pct
                      + "% fetch %{customdata[2]:.0f} m<br>FCH4 %{customdata[0]:.1f}<extra></extra>"))
    fig.data = tuple(t for t in fig.data if t.name != f"median {pct}% fetch") + \
        tuple(t for t in fig.data if t.name == f"median {pct}% fetch")       # dotted line on top
    fig.add_trace(_tower_trace())
    fig = _map_layout(fig, f"{title} \u00b7 n = {len(x)}", max(rout * 1.25, 60), tip)
    fig.update_layout(legend=dict(bgcolor="rgba(255,255,255,0.85)"))
    return fig


def make_fch4_surface(d: pd.DataFrame) -> go.Figure:
    """Smoothed FCH4 surface (top view) over wind direction x time of day.
    Each grid point is a Gaussian-weighted mean of nearby half-hours
    (sigma 30 deg in direction, 1.5 h in time, both circular); areas with no
    nearby data are left blank. Dots = measured half-hours."""
    title = "Mean CH\u2084 flux by wind direction and hour of day"
    if WD_COL not in d or FCH4_COL not in d:
        return _empty_fig(title)
    x = d[[TIME_COL, WD_COL, FCH4_COL]].dropna()
    if len(x) < 3:
        return _empty_fig(title)
    wd = x[WD_COL].to_numpy()
    hr = (x[TIME_COL].dt.hour + x[TIME_COL].dt.minute / 60).to_numpy()
    f = (x[FCH4_COL] * FCH4_TO_NMOL).to_numpy()              # nmol m-2 s-1

    gw, gh = np.arange(0, 361, 10), np.arange(0, 24.01, 0.5)
    sw, sh = 30.0, 1.5
    dw = np.abs(gw[None, :, None] - wd[None, None, :]); dw = np.minimum(dw, 360 - dw)
    dh = np.abs(gh[:, None, None] - hr[None, None, :]); dh = np.minimum(dh, 24 - dh)
    k = np.exp(-0.5 * ((dw / sw) ** 2 + (dh / sh) ** 2))    # (hours, dirs, points)
    wsum = k.sum(axis=2)
    Z = np.where(wsum >= 0.5, (k * f).sum(axis=2) / np.where(wsum > 0, wsum, 1), np.nan)

    lim = float(np.nanpercentile(np.abs(Z), 98)) if np.isfinite(Z).any() else 1.0
    fig = go.Figure(go.Contour(
        x=gw, y=gh, z=Z, colorscale="RdBu_r", zmid=0, zmin=-lim, zmax=lim,
        contours=dict(coloring="heatmap"), connectgaps=False, line=dict(width=0.5),
        colorbar=dict(title="nmol m-2 s-1", thickness=12),
        hovertemplate="WD %{x}\u00b0<br>%{y:.1f} h<br>FCH4 %{z:.1f}<extra>smoothed</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=wd, y=hr, mode="markers", marker=dict(size=5, color="black", opacity=0.6),
        customdata=f, showlegend=False,
        hovertemplate="WD %{x:.0f}\u00b0<br>%{y:.1f} h<br>FCH4 %{customdata:.1f}<extra>measured</extra>",
    ))
    fig.update_layout(
        title=dict(text=title + f" \u00b7 n = {len(f)}", x=0.5), height=460,
        meta={"tip": "Smoothed mean methane flux by wind direction and hour of day "
                     "(dots = measured half-hours)\n" + title_tip(FCH4_COL, WD_COL)},
        margin=dict(l=55, r=20, t=60, b=50),
        xaxis=dict(title="Wind direction (\u00b0)", range=[0, 360],
                   tickvals=[0, 90, 180, 270, 360], ticktext=["0 N", "90 E", "180 S", "270 W", "360 N"]),
        yaxis=dict(title="Hour of day", range=[0, 24], dtick=3),
    )
    return fig


DIURNAL_VARS = [  # (column, label, factor, unit)
    (FCH4_COL, "FCH4", FCH4_TO_NMOL, "nmol m-2 s-1"),
    (FC_COL, "FC", FC_TO_UMOL, "µmol m-2 s-1"),
    (LE_COL, "LE", 1, "W m-2"),
    ("FH2O", "FH2O", 1, "mmol m-2 s-1"),
    (H_COL, "H", 1, "W m-2"),
]


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def make_diurnal(d: pd.DataFrame) -> go.Figure:
    """Mean by time of day (30-min bins) with a shaded +/- 1 standard deviation band."""
    fig = make_subplots(rows=1, cols=len(DIURNAL_VARS), horizontal_spacing=0.06,
                        subplot_titles=[f"{lab} ({u})" for _c, lab, _k, u in DIURNAL_VARS])
    colors = SERIES_COLORS[:5]
    hod = d[TIME_COL].dt.hour + d[TIME_COL].dt.minute / 60
    first = True
    for i, (col, lab, k, _u) in enumerate(DIURNAL_VARS, start=1):
        if col not in d:
            continue
        g = (d[col] * k).groupby(hod).agg(["mean", "std", "count"])
        g = g[g["count"] > 0].sort_index()
        if g.empty:
            continue
        sd = g["std"].fillna(0)
        up, dn = g["mean"] + sd, g["mean"] - sd
        c = colors[(i - 1) % len(colors)]
        # shaded SD band: upper edge, then lower edge filled up to it
        fig.add_trace(go.Scatter(x=g.index, y=up, mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False, legendgroup="sd"),
                      row=1, col=i)
        fig.add_trace(go.Scatter(x=g.index, y=dn, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=_rgba(c, 0.25),
                                 name="± 1 SD", legendgroup="sd", showlegend=first,
                                 hoverinfo="skip"), row=1, col=i)
        fig.add_trace(go.Scatter(x=g.index, y=g["mean"], mode="lines+markers",
                                 line=dict(color=c, width=2), marker=dict(size=4),
                                 name="Mean", legendgroup="mean", showlegend=first,
                                 customdata=np.stack([sd, g["count"]], axis=-1),
                                 hovertemplate="%{x:.1f} h<br>mean %{y:.2f}<br>SD %{customdata[0]:.2f}"
                                               "<br>n = %{customdata[1]}<extra>" + lab + "</extra>"),
                      row=1, col=i)
        fig.update_xaxes(range=[0, 24], dtick=6, title_text="Hour of day", row=1, col=i)
        first = False
    fig.update_layout(title=dict(text="Diurnal cycle (mean ± 1 SD)", x=0.5), height=400,
                      meta={"tip": "Mean by time of day (30-min bins), shaded ± 1 standard deviation\n"
                                   + title_tip(*[c for c, *_ in DIURNAL_VARS]),
                            "ann_tips": [title_tip(c) for c, *_ in DIURNAL_VARS]},
                      margin=dict(l=40, r=20, t=90, b=50),
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.12, yanchor="bottom"))
    fig.update_annotations(font=dict(size=13))
    return fig


# =========================
# Custom "Add Plot" (like the TGA data monitor): pick variables, optional 2nd/3rd/4th y-axis
# =========================
CUSTOM_EXCLUDE = {TIME_COL, "RECORD", "TIMESTAMP_START", "TIMESTAMP_END"}
AXIS_COLORS = SERIES_COLORS[:4]
BTN = {}                       # buttons are styled by class in assets/theme.css


def custom_var_options(df: pd.DataFrame):
    cols = [c for c in df.columns if c not in CUSTOM_EXCLUDE and not c.startswith("_")
            and not c.upper().endswith(("_QC", "_SSITC_TEST"))
            and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]
    return [{"label": c, "value": c} for c in cols]


# ---- Add Plot: plot types and what each of the 4 dropdowns means for that type ----
CUSTOM_DD_IDS = ["all-variables-dropdown", "secondary-y-dropdown", "third-y-dropdown", "fourth-y-dropdown"]
DD_LABEL_STYLE = {}            # .field-label in assets/theme.css
PLOT_TYPE_OPTIONS = [
    {"label": "Time series (up to 4 y-axes)", "value": "ts"},
    {"label": "Scatter / comparison (X vs Y)", "value": "scatter"},
    {"label": "Linear regression", "value": "reg"},
    {"label": "Box plot (by group)", "value": "box"},
    {"label": "Fingerprint (day \u00d7 hour)", "value": "fp"},
    {"label": "Diurnal median + IQR (up to 4 y-axes)", "value": "diurnal"},
    {"label": "Polar (variable vs wind direction)", "value": "polar"},
]
BOX_GROUP_OPTIONS = [
    {"label": "Hour of day", "value": "hour"},
    {"label": "Day / night", "value": "daynight"},
    {"label": "Wind direction (8 sectors)", "value": "wd8"},
    {"label": "u* class", "value": "ustar"},
    {"label": "Date", "value": "day"},
]
# per plot type: (label, multi, placeholder, options kind) for dropdowns 1-4 (None = hidden)
PLOT_TYPE_FIELDS = {
    "ts": [("Y variable(s)", True, "Choose one or more variables", "var"),
           ("Secondary Y-axis", True, "Variable for secondary Y-axis", "var"),
           ("Third Y-axis", True, "Variable for third Y-axis", "var"),
           ("Fourth Y-axis", True, "Variable for 4th axis", "var")],
    "scatter": [("X variable", False, "e.g. USTAR", "var"),
                ("Y variable(s)", True, "e.g. FCH4", "var"),
                ("Colour points by (optional)", False, "e.g. TA_1_1_1", "var"),
                None],
    "reg": [("X variable (independent)", False, "e.g. FC", "var"),
            ("Y variable (dependent)", False, "e.g. FCH4", "var"),
            None, None],
    "box": [("Variable", False, "e.g. FCH4", "var"),
            ("Group by", False, "Choose grouping", "group"),
            None, None],
    "fp": [("Variable", False, "e.g. FCH4", "var"), None, None, None],
    "diurnal": [("Left axis", False, "e.g. CH4", "var"),
                ("Right axis (optional)", False, "e.g. FCH4", "var"),
                ("2nd right axis (optional)", False, "e.g. USTAR", "var"),
                ("3rd right axis (optional)", False, "e.g. TA_1_1_1", "var")],
    "polar": [("Radial variable", False, "e.g. FCH4", "var"),
              ("Angle (direction) variable", False, "Default: WD", "var"),
              ("Colour points by (optional)", False, "e.g. USTAR", "var"),
              None],
}
PLOT_TYPE_HINTS = {
    "ts": "Variables against time; extra axes are colour-coded.",
    "scatter": "Y against X, with binned medians (IQR bars), e.g. FCH4 vs u*.",
    "reg": "Least-squares line with equation, R\u00b2, RMSE and n.",
    "box": "Distribution of a variable per group (median, quartiles, outliers).",
    "fp": "Heat map of a variable: date on x, time of day on y.",
    "diurnal": "Hourly median with IQR shading; each variable on its own colour-coded axis.",
    "polar": "Variable against wind direction on a polar plot (half-hours + 16-sector mean).",
}


def _one(v):
    """Dropdown value -> single column name (accepts list or string)."""
    if isinstance(v, (list, tuple)):
        return v[0] if v else None
    return v or None


def _many(v):
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def build_custom_spec(selected, y2, y3, y4, ptype="ts"):
    """Turn the 4 dropdown values into a plot spec for the chosen plot type (None = incomplete)."""
    if ptype == "scatter":
        x, ys = _one(selected), [v for v in _many(y2) if v != _one(selected)]
        return {"type": "scatter", "x": x, "y": ys, "color": _one(y3)} if x and ys else None
    if ptype == "reg":
        x, y = _one(selected), _one(y2)
        return {"type": "reg", "x": x, "y": y} if x and y and x != y else None
    if ptype == "box":
        y, g = _one(selected), _one(y2)
        return {"type": "box", "y": y, "group": g} if y and g else None
    if ptype == "fp":
        v = _one(selected)
        return {"type": "fp", "var": v} if v else None
    if ptype == "polar":
        r, th = _one(selected), _one(y2) or WD_COL
        return {"type": "polar", "r": r, "theta": th, "color": _one(y3)} if r and r != th else None
    if ptype == "diurnal":
        vs = []
        for v in (_one(selected), _one(y2), _one(y3), _one(y4)):
            if v and v not in vs:
                vs.append(v)
        return {"type": "diurnal", "vars": vs} if vs else None
    # time series: axis variables are added to the plot if not already chosen (TGA monitor logic)
    vars_ = _many(selected)
    for axis in (_many(y2), _many(y3), _many(y4)):
        for v in axis:
            if v not in vars_:
                vars_.append(v)
    if not vars_:
        return None
    y2 = [v for v in _many(y2) if v in vars_]
    y3 = [v for v in _many(y3) if v in vars_ and v not in y2]
    y4 = [v for v in _many(y4) if v in vars_ and v not in y2 and v not in y3]
    return {"type": "ts", "vars": vars_, "y2": y2, "y3": y3, "y4": y4}


def spec_columns(spec) -> list:
    t = spec.get("type", "ts")
    if t == "scatter":
        return [spec["x"], *spec["y"]] + ([spec["color"]] if spec.get("color") else [])
    if t == "reg":
        return [spec["x"], spec["y"]]
    if t == "box":
        return [spec["y"]]
    if t == "fp":
        return [spec["var"]]
    if t == "polar":
        return [spec["r"], spec["theta"]] + ([spec["color"]] if spec.get("color") else [])
    return list(spec.get("vars", []))          # "ts" and "diurnal"


def _binned_median(x: pd.Series, y: pd.Series, nbins: int = 10):
    """Median and IQR of y in equal-count bins of x (bins with < 3 points dropped)."""
    ok = x.notna() & y.notna()
    if ok.sum() < 10:
        return None
    q = pd.qcut(x[ok], q=min(nbins, int(ok.sum() // 5)), duplicates="drop")
    g = y[ok].groupby(q, observed=True)
    xm = x[ok].groupby(q, observed=True).median()
    res = pd.DataFrame({"x": xm, "med": g.median(), "q1": g.quantile(.25), "q3": g.quantile(.75),
                        "n": g.count()})
    return res[res["n"] >= 3]


def make_scatter_figure(df, spec, units_map) -> go.Figure:
    x = spec["x"]
    ys = [v for v in spec["y"] if v in df.columns]
    cvar = spec.get("color") if spec.get("color") in df.columns else None
    fig = go.Figure()
    for i, y in enumerate(ys):
        col = PANEL_COLORS[i % len(PANEL_COLORS)]
        marker = dict(size=6, color=col, opacity=0.55, line=dict(width=0.3, color="#333"))
        if cvar and len(ys) == 1:
            marker.update(color=df[cvar], colorscale="Viridis", opacity=0.8, showscale=True,
                          colorbar=dict(title=format_title(cvar, units_map), thickness=12))
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y], mode="markers", marker=marker, name=y,
            customdata=df[TIME_COL].dt.strftime("%y/%m/%d %H:%M"),
            hovertemplate="%{customdata}<br>" + x + ": %{x}<br>" + y + ": %{y}<extra></extra>"))
        b = _binned_median(df[x], df[y])
        if b is not None and len(b):
            fig.add_trace(go.Scatter(
                x=b["x"], y=b["med"], mode="lines+markers", name=f"{y} binned median",
                line=dict(color="black" if len(ys) == 1 else col, width=2.5),
                marker=dict(size=8, symbol="diamond"),
                error_y=dict(type="data", symmetric=False, array=b["q3"] - b["med"],
                             arrayminus=b["med"] - b["q1"], thickness=1.2, width=4),
                customdata=b["n"], hovertemplate="median %{y:.3g}<br>n = %{customdata}<extra></extra>"))
    n = int(df[[x, *ys]].dropna().shape[0]) if ys else 0
    fig.update_layout(
        title=dict(text=f"{', '.join(ys)} vs {x} \u00b7 n = {n}", x=0.5, font=dict(size=14)),
        xaxis_title=format_title(x, units_map),
        yaxis_title=format_title(ys[0], units_map) if len(ys) == 1 else "",
        height=420, margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", font=dict(size=11)))
    return fig


def make_regression_figure(df, spec, units_map) -> go.Figure:
    x, y = spec["x"], spec["y"]
    d = df[[TIME_COL, x, y]].dropna()
    fig = go.Figure(go.Scatter(
        x=d[x], y=d[y], mode="markers", name="half-hours",
        marker=dict(size=6, color="#1f77b4", opacity=0.55, line=dict(width=0.3, color="#333")),
        customdata=d[TIME_COL].dt.strftime("%y/%m/%d %H:%M"),
        hovertemplate="%{customdata}<br>" + x + ": %{x}<br>" + y + ": %{y}<extra></extra>"))
    title = f"{y} vs {x}"
    if len(d) >= 3 and d[x].nunique() > 1:
        slope, icpt = np.polyfit(d[x], d[y], 1)
        pred = slope * d[x] + icpt
        ss_res = float(((d[y] - pred) ** 2).sum()); ss_tot = float(((d[y] - d[y].mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = math.sqrt(ss_res / len(d))
        xl = np.array([d[x].min(), d[x].max()])
        sign = "+" if icpt >= 0 else "\u2212"
        fig.add_trace(go.Scatter(x=xl, y=slope * xl + icpt, mode="lines", name="least-squares fit",
                                 line=dict(color="#d62728", width=2.5)))
        fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.98, xanchor="left", yanchor="top",
                           showarrow=False, align="left", bgcolor="rgba(255,255,255,0.8)",
                           text=(f"y = {slope:.4g} x {sign} {abs(icpt):.4g}<br>"
                                 f"R\u00b2 = {r2:.3f} \u00b7 RMSE = {rmse:.3g} \u00b7 n = {len(d)}"))
    else:
        title += " (not enough data)"
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=14)),
                      xaxis_title=format_title(x, units_map), yaxis_title=format_title(y, units_map),
                      height=420, margin=dict(l=60, r=30, t=60, b=50), showlegend=False)
    return fig


def _box_groups(df, how):
    """(group labels per row, ordered category list, axis title)."""
    t = df[TIME_COL]
    if how == "hour":
        return t.dt.hour, list(range(24)), "Hour of day"
    if how == "day":
        g = t.dt.strftime("%m-%d")
        return g, sorted(g.dropna().unique()), "Date"
    if how == "daynight":
        if "daytime" in df:
            day = df["daytime"] >= 0.5
        elif "sun_elevation" in df:
            day = df["sun_elevation"] > 0
        else:
            day = t.dt.hour.between(7, 18)
        return day.map({True: "Day", False: "Night"}), ["Day", "Night"], ""
    if how == "wd8" and WD_COL in df:
        names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        idx = ((df[WD_COL] + 22.5) % 360 // 45)
        return idx.map(lambda k: names[int(k)] if pd.notna(k) else None), names, "Wind direction"
    if how == "ustar" and USTAR_COL in df:
        bins, labs = [-np.inf, 0.1, 0.2, 0.3, np.inf], ["< 0.1", "0.1\u20130.2", "0.2\u20130.3", "\u2265 0.3"]
        return pd.cut(df[USTAR_COL], bins, labels=labs, right=False).astype(object), labs, "u* (m s-1)"
    return pd.Series("all", index=df.index), ["all"], ""


def make_box_figure(df, spec, units_map) -> go.Figure:
    y, how = spec["y"], spec["group"]
    g, order, gtitle = _box_groups(df, how)
    fig = go.Figure()
    for k in order:
        vals = df.loc[g == k, y].dropna()
        if len(vals):
            fig.add_trace(go.Box(y=vals, name=f"{k}", boxpoints="outliers", marker=dict(size=3),
                                 line=dict(width=1.2), fillcolor=_rgba(SERIES_COLORS[0], 0.2),
                                 marker_color=SERIES_COLORS[0], hovertemplate=f"{k}: " + "%{y}<extra></extra>"))
    # n per group above each box (only when there are few groups, to avoid clutter)
    if len(fig.data) <= 12:
        for tr in fig.data:
            fig.add_annotation(x=tr.name, y=1.0, yref="paper", text=f"n={len(tr.y)}",
                               showarrow=False, font=dict(size=9, color="#666"), yanchor="bottom")
    glabel = next((o["label"] for o in BOX_GROUP_OPTIONS if o["value"] == how), how)
    fig.update_layout(title=dict(text=f"{y} by {glabel.lower()}", x=0.5, font=dict(size=14)),
                      yaxis_title=format_title(y, units_map), xaxis_title=gtitle,
                      xaxis=dict(type="category"), showlegend=False,
                      height=420, margin=dict(l=60, r=30, t=60, b=70))
    return fig


DIURNAL_AXIS_COLORS = SERIES_COLORS[:4]


def make_diurnal_multi_figure(df, spec, units_map) -> go.Figure:
    """Hourly median (line) and interquartile range (shaded) for up to 4 variables,
    each on its own colour-coded y-axis (left, right, and two extra right axes)."""
    vars_ = [v for v in spec["vars"] if v in df.columns][:4]
    hour = df[TIME_COL].dt.hour
    x = np.arange(24) + 0.5                                  # centre of each hour
    axes = ["y", "y2", "y3", "y4"][:len(vars_)]
    n_right = len(vars_) - 1
    x_end = {0: 1.0, 1: 1.0, 2: 0.86, 3: 0.74}[max(n_right, 0)]
    free_pos = {2: {"y3": 0.95}, 3: {"y3": 0.86, "y4": 0.98}}.get(n_right, {})
    fig = go.Figure()
    for i, (v, ax) in enumerate(zip(vars_, axes)):
        col = DIURNAL_AXIS_COLORS[i]
        g = df[v].groupby(hour)
        med = g.median().reindex(range(24)).values
        q1 = g.quantile(.25).reindex(range(24)).values
        q3 = g.quantile(.75).reindex(range(24)).values
        n = g.count().reindex(range(24)).fillna(0).astype(int).values
        fig.add_trace(go.Scatter(x=x, y=q3, yaxis=ax, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=q1, yaxis=ax, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=_rgba(col, 0.15),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=x, y=med, yaxis=ax, name=v, mode="lines+markers",
            line=dict(color=col, width=2.2, dash="dash" if i == 2 else "solid"),
            marker=dict(size=5, symbol=["circle", "square", "diamond", "triangle-up"][i]),
            customdata=np.c_[q1, q3, n],
            hovertemplate=(f"{v}<br>hour %{{x:.1f}}<br>median %{{y:.3g}}<br>"
                           "IQR %{customdata[0]:.3g} to %{customdata[1]:.3g}<br>n = %{customdata[2]}<extra></extra>")))
    lay = {}
    for i, (v, ax) in enumerate(zip(vars_, axes)):
        col = DIURNAL_AXIS_COLORS[i]
        cfg = dict(title=dict(text=format_title(v, units_map), font=dict(size=11, color=col)),
                   tickfont=dict(size=10, color=col), tickmode="auto", nticks=6)
        if ax != "y":
            cfg.update(overlaying="y", side="right", showgrid=False)
            if ax in free_pos:
                cfg.update(anchor="free", position=free_pos[ax])
        lay["yaxis" if ax == "y" else "yaxis" + ax[1:]] = cfg
    fig.update_layout(**lay)
    if vars_ and df[vars_[0]].min() < 0 < df[vars_[0]].max():
        fig.add_hline(y=0, line=dict(color="#666", width=0.8, dash="dot"))
    fig.update_xaxes(range=[0, 24], dtick=3, title_text="Hour of day", domain=[0, x_end])
    fig.update_layout(
        title=dict(text="Diurnal median (IQR shaded): " + ", ".join(vars_), x=0.5, font=dict(size=14)),
        height=420, margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(size=11),
                    traceorder="normal"))
    return fig


def make_fingerprint_figure(df, spec, units_map) -> go.Figure:
    v = spec["var"]
    d = df[[TIME_COL, v]].dropna()
    fig = go.Figure()
    if d.empty:
        return _empty_fig(f"{v} fingerprint")
    day = d[TIME_COL].dt.normalize()
    hh = d[TIME_COL].dt.hour + d[TIME_COL].dt.minute / 60
    Z = d.assign(day=day, hh=hh).pivot_table(index="hh", columns="day", values=v, aggfunc="mean")
    Z = Z.reindex(index=np.arange(0, 24, 0.5),
                  columns=pd.date_range(day.min(), day.max(), freq="D"))
    lo, hi = np.nanpercentile(d[v], [2, 98])
    diverging = lo < 0 < hi and abs(lo) > 0.2 * abs(hi)   # both signs matter (e.g. fluxes)
    lim = max(abs(lo), abs(hi))
    fig.add_trace(go.Heatmap(
        x=Z.columns, y=Z.index, z=Z.values, hoverongaps=False,
        colorscale="RdBu_r" if diverging else "Viridis",
        zmin=-lim if diverging else lo, zmax=lim if diverging else hi, zmid=0 if diverging else None,
        colorbar=dict(title=format_title(v, units_map), thickness=12),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f} h<br>" + v + ": %{z:.3g}<extra></extra>"))
    fig.update_layout(title=dict(text=f"{v} fingerprint (date \u00d7 time of day)", x=0.5, font=dict(size=14)),
                      xaxis=dict(title="Date", tickformat="%m-%d"),
                      yaxis=dict(title="Time of day (h)", range=[0, 24], dtick=3),
                      height=420, margin=dict(l=60, r=30, t=60, b=50))
    return fig


def make_polar_figure(df, spec, units_map) -> go.Figure:
    """Radial variable vs a direction variable (deg, compass: 0 = N, clockwise).
    Dots = half-hours (optionally coloured by a 3rd variable); red line = 16-sector mean."""
    r, th, c = spec["r"], spec.get("theta") or WD_COL, spec.get("color")
    title = f"{r} vs {th}"
    cols = [th, r] + ([c] if c and c not in (r, th) else [])
    d = df[cols].dropna(subset=[th, r])
    if d.empty:
        return _empty_fig(title)
    theta = d[th] % 360
    fig = go.Figure()
    mk = dict(size=5, opacity=0.65, color="#1f77b4")
    if c and c in d:
        mk.update(color=d[c], colorscale="Viridis", showscale=True,
                  colorbar=dict(title=format_title(c, units_map), thickness=12, x=1.08))
    fig.add_trace(go.Scatterpolar(
        r=d[r], theta=theta, mode="markers", marker=mk, name="30-min",
        hovertemplate=f"{th} " + "%{theta:.0f}\u00b0<br>" + r + " %{r:.3g}<extra></extra>"))
    # 16-sector mean (22.5 deg sectors centred on N, NNE, ...), closed loop
    idx = _sector_index(theta, 16)
    m = d[r].groupby(idx).mean().reindex(range(16))
    n = d[r].groupby(idx).count().reindex(range(16)).fillna(0).astype(int)
    if m.notna().any():
        ang = np.arange(16) * 22.5
        fig.add_trace(go.Scatterpolar(
            r=np.append(m.values, m.values[0]), theta=np.append(ang, 360),
            mode="lines+markers", name="Sector mean", line=dict(color="#d62728", width=2),
            marker=dict(size=5), customdata=np.append(n.values, n.values[0]),
            hovertemplate="%{theta:.1f}\u00b0<br>mean %{r:.3g}<br>n = %{customdata}<extra></extra>"))
    lo, hi = float(d[r].min()), float(d[r].max())
    pad = 0.05 * (hi - lo or 1)
    fig.update_layout(
        title=dict(text=f"{fmt_units(format_title(r, units_map))} vs {th} \u00b7 n = {len(d)}"
                   + ("<br><sup>centre = minimum value (radial axis starts below 0)</sup>" if lo < 0 else ""),
                   x=0.5, font=dict(size=14)),
        height=460, margin=dict(l=60, r=80, t=80, b=50),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.05, yanchor="top"),
        polar=dict(angularaxis=dict(direction="clockwise", rotation=90, tickmode="array",
                                    tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                                    ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
                   radialaxis=dict(range=[lo - pad, hi + pad], angle=45, nticks=5,
                                   tickfont=dict(size=9))),
    )
    return fig


def make_custom_figure(df, spec, units_map, dtick, tickformat) -> go.Figure:
    df = with_gaps(df)
    vars_ = [v for v in spec["vars"] if v in df.columns]
    y2, y3, y4 = set(spec["y2"]), set(spec["y3"]), set(spec["y4"])
    axis_of = {v: ("y4" if v in y4 else "y3" if v in y3 else "y2" if v in y2 else "y") for v in vars_}
    used = [a for a in ("y", "y2", "y3", "y4") if a in axis_of.values()]
    n_right = len(used) - 1
    # plot area shrinks as axes are added; extra right-hand axes sit in the freed space
    x_end = {0: 1.0, 1: 1.0, 2: 0.84, 3: 0.72}[n_right]
    free_pos = {2: {"y3": 0.93}, 3: {"y3": 0.83, "y4": 0.95}}.get(n_right, {})

    fig = go.Figure()
    color_of = {v: PANEL_COLORS[i % len(PANEL_COLORS)] for i, v in enumerate(vars_)}
    for v in vars_:
        fig.add_trace(go.Scatter(
            x=df[TIME_COL], y=df[v], mode="lines+markers", marker=dict(size=3),
            line=dict(color=color_of[v]), name=v, yaxis=axis_of[v],
            hovertemplate="%{x|%y/%m/%d %H:%M}<br>" + v + ": %{y}<extra></extra>",
        ))

    def ax_cfg(axis):
        vs = [v for v in vars_ if axis_of[v] == axis]
        col = color_of[vs[0]] if len(vs) == 1 else "#333"      # colour-code single-variable axes
        return dict(title=dict(text=", ".join(format_title(v, units_map) for v in vs),
                               font=dict(size=11, color=col)),
                    tickfont=dict(size=10, color=col))

    lay = {"yaxis": ax_cfg("y")}
    for a in used[1:]:
        cfg = {**ax_cfg(a), "overlaying": "y", "side": "right", "showgrid": False}
        if a in free_pos:
            cfg.update(anchor="free", position=free_pos[a])
        lay["yaxis" + a[1:]] = cfg
    fig.update_layout(**lay)

    tick0 = aligned_tick0(df[TIME_COL].min(), dtick)
    fig.update_xaxes(type="date", tickmode="linear", tick0=tick0, dtick=dtick, tickformat=tickformat,
                     tickangle=30, domain=[0, x_end], tickfont=dict(size=10))
    fig.update_layout(
        title=dict(text=" vs ".join(vars_), x=0.5, font=dict(size=14)),
        height=380, margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(size=11)),
    )
    return fig


def spec_home(sp) -> str:
    """Tab an added plot is shown on (no tab = flux tab; the old Meteorology tab is now the flux tab)."""
    t = sp.get("tab", FLUX_TAB)
    return FLUX_TAB if (t == MET_TAB or t in OLD_FLUX_TABS) else t


def specs_for_tab(specs, tab):
    """Added-plot specs that belong to a tab."""
    return [sp for sp in (specs or []) if spec_home(sp) == tab]


CUSTOM_MAKERS = {"scatter": make_scatter_figure, "reg": make_regression_figure,
                 "box": make_box_figure, "fp": make_fingerprint_figure,
                 "diurnal": make_diurnal_multi_figure, "polar": make_polar_figure}


def valid_custom_specs(df, specs):
    """Added-plot specs whose variables all exist in the data (their order sets the card index)."""
    return [sp for sp in (specs or []) if all(c in df.columns for c in spec_columns(sp) if c)]


def custom_figure(df, sp, units_map, dtick, tickformat) -> go.Figure:
    t = sp.get("type", "ts")
    fig = (CUSTOM_MAKERS[t](df, sp, units_map) if t in CUSTOM_MAKERS
           else make_custom_figure(df, sp, units_map, dtick, tickformat))
    fig.update_layout(meta={**(fig.layout.meta or {}),
                            "tip": title_tip(*dict.fromkeys(c for c in spec_columns(sp) if c))})
    return fig


def custom_section(df, specs, units_map, dtick, tickformat):
    specs = valid_custom_specs(df, specs)
    if not specs:
        return None
    cards = [plot_card(custom_figure(df, sp, units_map, dtick, tickformat), CUSTOM_IDX0 + i)
             for i, sp in enumerate(specs)]
    return html.Div(className="section", children=[
        section_header("Added plots"),
        html.Div(cards, className="card-grid card-grid-wide"),
    ])


CUSTOM_IDX0 = 1000      # card index of the first added plot


SAVE_BTN_STYLE = {}            # .btn classes in assets/theme.css


def field(label, *controls, cls=""):
    """Toolbar control with a small label above it."""
    return html.Div(className=("field " + cls).strip(), children=[
        html.Label(label, className="field-label"),
        html.Div(list(controls), className="field-control"),
    ])


def serve_layout():
    """Built on every page load, so the date picker always reflects the latest data."""
    df, _ = load_data()
    var_opts = custom_var_options(df) if not df.empty else []
    if df.empty:
        min_d = max_d = None
    else:
        min_d = df[TIME_COL].min().date()
        max_d = df[TIME_COL].max().date()

    return html.Div(className="app-shell", children=[
        # ---------- Header ----------
        html.Header(className="app-header", children=[
            html.Div(className="header-inner", children=[
                html.Div(className="brand", children=[
                    html.Div([
                        html.H1("TGA310 Cattle CH₄ Emission Measurement", className="app-title"),
                        html.Div("Eddy covariance flux monitoring · USU test site · "
                                 "30-min AmeriFlux-format data", className="app-subtitle"),
                    ]),
                ]),
                html.Div(className="status", children=[
                    html.Span(className="status-dot"),
                    html.Span(id="last-updated", className="status-text"),
                ]),
            ]),
            # Tabs sit in the header, under the title
            html.Div(className="header-inner", children=[
                dcc.Tabs(id="tabs", value=SETUP_TAB, className="app-tabs",
                         parent_className="app-tabs-parent",
                         style={"border": "none", "height": "auto"}, children=[
                    dcc.Tab(label=n, value=n, style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE)
                    for n in [SETUP_TAB] + tab_names + ([TGAMON_TAB] if ENABLE_TGAMON else [])
                ]),
            ]),
        ]),

        html.Main(className="app-main", children=[
            # ---------- Toolbar: date range, filters, actions ----------
            html.Div(id="range-row", className="toolbar", style=RANGE_ROW_STYLE, children=[
                field("Date range", dcc.DatePickerRange(
                    id="dp-range", min_date_allowed=min_d, max_date_allowed=max_d,
                    start_date=min_d, end_date=max_d, display_format="YYYY-MM-DD", clearable=True)),
                html.Div(id="flux-controls", style=FLUX_CONTROLS_STYLE, children=[
                    html.Div(className="toolbar-divider"),
                    *[field(lab.replace("_QC", " QC"),
                            dcc.Dropdown(id=qid, options=QC_OPTIONS, value="all", clearable=False,
                                         searchable=False, className="dd-small"))
                      for qid, lab, _c, _q in QC_FILTERS],
                    field("u* filter", dcc.Dropdown(id="ustar-filter", options=USTAR_OPTIONS, value="none",
                                                    clearable=False, searchable=False, className="dd-small")),
                ]),
                html.Div(id="wd-controls", style=WD_CONTROLS_STYLE, children=[
                    field("Wind direction", html.Details(id="wd-details", className="wd-details", children=[
                        html.Summary(id="wd-summary", children="All ▾", className="wd-summary"),
                        html.Div(className="wd-panel", children=[
                            html.Div(className="wd-actions", children=[
                                html.Button("Select all", id="wd-all", n_clicks=0, className="btn btn-xs"),
                                html.Button("Clear", id="wd-none", n_clicks=0, className="btn btn-xs"),
                                html.Button("Done", id="wd-done", n_clicks=0,
                                            className="btn btn-xs btn-primary", style={"marginLeft": "auto"}),
                            ]),
                            dcc.Checklist(
                                id="wd-sector",
                                options=[{"label": " " + k, "value": k} for k in WD_SECTORS],
                                value=list(WD_SECTORS),      # default: all directions
                                className="wd-checklist",
                                labelStyle={"display": "block"},
                            ),
                        ]),
                    ])),
                ]),
                html.Div(id="tgamon-controls", style={**TGAMON_CONTROLS_STYLE, "display": "none"}, children=[
                    field("Display", dcc.Dropdown(id="tgamon-display", options=TGAMON_DISPLAY_OPTIONS,
                                                  value="30min", clearable=False, className="dd-medium")),
                ]),
                html.Div(className="toolbar-actions", children=[
                    html.Button("+ Add plot", id="toggle-variable-panel-btn", n_clicks=0,
                                className="btn btn-outline"),
                    html.Button("Save plots (PDF)", id="save-pdf-btn", n_clicks=0,
                                title="Save all plots on this tab to a PDF file",
                                className="btn btn-primary"),
                ]),
            ]),

            # ---------- Add-plot panel ----------
            html.Div(id="variable-panel", style={"display": "none"}, className="add-panel", children=[
                html.Div(className="add-panel-head", children=[
                    html.Div("Create an additional plot", className="add-panel-title"),
                    field("Plot type", dcc.Dropdown(id="plot-type", options=PLOT_TYPE_OPTIONS, value="ts",
                                                    clearable=False, style={"width": "280px"})),
                    html.Span(id="plot-type-hint", className="hint"),
                ]),
                html.Div(className="add-panel-grid", children=[
                    html.Div(id=f"dd-wrap-{i}", children=[
                        html.Div(id=f"dd-label-{i}", className="field-label", style=DD_LABEL_STYLE),
                        dcc.Dropdown(id=dd_id, options=var_opts, value=[], multi=True),
                    ])
                    for i, dd_id in enumerate(CUSTOM_DD_IDS, start=1)
                ]),
                html.Div(className="add-panel-actions", children=[
                    html.Button("Add plot", id="add-plot-btn", n_clicks=0, className="btn btn-primary"),
                    html.Button("Clear added plots (this tab)", id="clear-selected-vars-btn", n_clicks=0,
                                className="btn btn-ghost"),
                ]),
            ]),
            dcc.Store(id="custom-specs", storage_type="session", data=[]),
            dcc.Loading(type="dot", color="#2a78d6", delay_show=400,
                        target_components={"tab-content": "children"},
                        overlay_style={"visibility": "visible", "opacity": 0.55, "filter": "blur(1px)"},
                        children=html.Div(id="tab-content", className="tab-body")),
        ]),

        html.Footer(className="app-footer", children=[
            html.Span("Campbell Scientific TGA310 trace gas analyzer · eddy covariance system"),
            html.Span(f"Data refresh every {REFRESH_MINUTES} min · hover a plot title for the full "
                      "variable name"),
        ]),

        # Pop-up for a maximized plot
        html.Div(id="modal-overlay", style=MODAL_HIDDEN, children=[
            html.Div(className="modal-box", children=[
                html.Div(className="modal-head", children=[
                    html.Button("✕  Close", id="close-modal-btn", n_clicks=0, className="btn btn-ghost"),
                ]),
                dcc.Graph(id="zoom-graph", style={"flex": "1 1 auto", "minHeight": 0},
                          config={"displaylogo": False}),
            ]),
        ]),
        html.Div(id="title-tooltip", className="title-tooltip"),
        # Re-check GitHub while the page is open
        dcc.Interval(id="refresh", n_intervals=0,
                     interval=(RETRY_SECONDS_WHEN_EMPTY * 1000 if df.empty
                               else REFRESH_MINUTES * 60 * 1000)),
    ])


RANGE_ROW_STYLE = {}          # toolbar is styled by class; only shown / hidden from callbacks

# Google font for the page (the rest of the styling is assets/theme.css)
app.index_string = app.index_string.replace("</head>", """<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>""")

app.layout = serve_layout


# =========================
# Callbacks
# =========================
@app.callback(
    Output("range-row", "style"),
    Output("last-updated", "style"),
    Output("flux-controls", "style"),
    Output("wd-controls", "style"),
    Output("save-pdf-btn", "style"),
    Output("tgamon-controls", "style"),
    Input("tabs", "value"),
)
def toggle_range(tab_value):
    """Date picker hidden on Setup; QC filters on the flux tab; wind direction on flux + met tabs."""
    lu = {}                        # "Last record" sits in the header and is always shown
    hide = {"display": "none"}
    fc = FLUX_CONTROLS_STYLE if tab_value == FLUX_TAB else {**FLUX_CONTROLS_STYLE, **hide}
    wd = WD_CONTROLS_STYLE if tab_value == FLUX_TAB else {**WD_CONTROLS_STYLE, **hide}
    tg = TGAMON_CONTROLS_STYLE if tab_value == TGAMON_TAB else {**TGAMON_CONTROLS_STYLE, **hide}
    if tab_value == SETUP_TAB:
        return {**RANGE_ROW_STYLE, **hide}, lu, fc, wd, {**SAVE_BTN_STYLE, **hide}, tg
    return RANGE_ROW_STYLE, lu, fc, wd, SAVE_BTN_STYLE, tg


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
        return (None, None, None, None, "Waiting for data",
                RETRY_SECONDS_WHEN_EMPTY * 1000)

    new_min = df[TIME_COL].min().date()
    new_max = df[TIME_COL].max().date()
    tg = load_tgamon()[0] if ENABLE_TGAMON else pd.DataFrame()   # picker also covers TGAMon days
    if not tg.empty:
        new_min = min(new_min, tg[TIME_COL].min().date())
        new_max = max(new_max, tg[TIME_COL].max().date())
    last_rec = df[TIME_COL].max().strftime("%Y-%m-%d %H:%M")

    if start_date is None:                       # data just arrived
        start_date = new_min
    if end_date is None or (old_max is not None and str(end_date)[:10] == str(old_max)[:10]):
        end_date = new_max
    return (new_min, new_max, start_date, end_date,
            f"Last record {last_rec}",
            REFRESH_MINUTES * 60 * 1000)


PANEL_STYLE = {"display": "block"}      # .add-panel in assets/theme.css


@app.callback(
    Output("variable-panel", "style"),
    Output("toggle-variable-panel-btn", "style"),
    Input("toggle-variable-panel-btn", "n_clicks"),
    Input("tabs", "value"),
)
def toggle_variable_panel(n, tab_value):
    """Add Plot panel: opened/closed by the button; on the Flux and Meteorology and TGAMon tabs."""
    btn = {}
    if tab_value not in (FLUX_TAB, TGAMON_TAB):    # Add Plot on the flux and TGAMon tabs
        return {"display": "none"}, {**btn, "display": "none"}
    return (PANEL_STYLE if (n or 0) % 2 == 1 else {"display": "none"}), btn


@app.callback(
    Output("custom-specs", "data"),
    Output("all-variables-dropdown", "value"),
    Output("secondary-y-dropdown", "value"),
    Output("third-y-dropdown", "value"),
    Output("fourth-y-dropdown", "value"),
    Input("add-plot-btn", "n_clicks"),
    Input("clear-selected-vars-btn", "n_clicks"),
    State("all-variables-dropdown", "value"),
    State("secondary-y-dropdown", "value"),
    State("third-y-dropdown", "value"),
    State("fourth-y-dropdown", "value"),
    State("custom-specs", "data"),
    State("plot-type", "value"),
    State("tabs", "value"),
    prevent_initial_call=True,
)
def update_custom_specs(_add, _clear, sel, y2, y3, y4, specs, ptype, tab_value):
    """Added plots are kept per tab: each spec remembers the tab it was created on."""
    from dash import ctx, no_update
    if ctx.triggered_id == "clear-selected-vars-btn":           # clear only this tab's plots
        keep = [sp for sp in (specs or []) if spec_home(sp) != tab_value]
        return keep, no_update, no_update, no_update, no_update
    spec = build_custom_spec(sel, y2, y3, y4, ptype or "ts")
    if spec is None:
        return no_update, no_update, no_update, no_update, no_update
    spec["tab"] = tab_value
    empty = [[] if f and f[1] else None for f in PLOT_TYPE_FIELDS.get(ptype or "ts")]
    return (specs or []) + [spec], *[e if e is not None else None for e in empty]


@app.callback(
    *[Output(f"dd-wrap-{i}", "style") for i in range(1, 5)],
    *[Output(f"dd-label-{i}", "children") for i in range(1, 5)],
    *[Output(dd, "multi") for dd in CUSTOM_DD_IDS],
    *[Output(dd, "placeholder") for dd in CUSTOM_DD_IDS],
    *[Output(dd, "options") for dd in CUSTOM_DD_IDS],
    *[Output(dd, "value", allow_duplicate=True) for dd in CUSTOM_DD_IDS],
    Output("plot-type-hint", "children"),
    Input("plot-type", "value"),
    Input("tabs", "value"),
    prevent_initial_call="initial_duplicate",
)
def configure_plot_type(ptype, tab_value=None):
    """Relabel / show / hide the 4 variable dropdowns to match the selected plot type;
    variables come from the TGAMon file on the TGAMon tab, else from the AmeriFlux table."""
    fields = PLOT_TYPE_FIELDS.get(ptype or "ts", PLOT_TYPE_FIELDS["ts"])
    df = load_tgamon()[0] if (ENABLE_TGAMON and tab_value == TGAMON_TAB) else load_data()[0]
    var_opts = custom_var_options(df) if not df.empty else []
    styles, labels, multis, holders, opts, vals = [], [], [], [], [], []
    for f in fields:
        if f is None:
            styles.append({"display": "none"}); labels.append(""); multis.append(True)
            holders.append(""); opts.append(var_opts); vals.append([])
        else:
            lab, multi, ph, kind = f
            styles.append({"display": "block"}); labels.append(lab); multis.append(multi)
            holders.append(ph); opts.append(BOX_GROUP_OPTIONS if kind == "group" else var_opts)
            vals.append([] if multi else None)
    return (*styles, *labels, *multis, *holders, *opts, *vals, PLOT_TYPE_HINTS.get(ptype, ""))


# ---------- Flux tab: every card, with the columns it depends on ----------
# The full tab is only rebuilt when the tab, date range or data change. Changing a QC flag,
# the u* filter or the wind-direction filter re-draws only the cards that use the affected
# columns, and adding / clearing a plot only re-draws the "Added plots" section.

def flux_frame(start_date, end_date):
    """(date-filtered data, units, panel list) for the flux tab; data is None when nothing to show."""
    df, units_map = load_data()
    if df.empty:
        return None, units_map, []
    dff = filter_df_by_datepicker_range(df, start_date, end_date)
    vars_list = [p for p in pages.get(FLUX_TAB, []) if any(v in df.columns for v in panel_vars(p))]
    return dff, units_map, vars_list


def flux_card_specs(vars_list, custom_specs, dff):
    """[(card index, set of columns used, maker(dfq, units, dtick, tickformat) -> figure), ...]"""
    cards = []
    for i, sp in enumerate(valid_custom_specs(dff, specs_for_tab(custom_specs, FLUX_TAB))):
        cards.append((CUSTOM_IDX0 + i, set(c for c in spec_columns(sp) if c),
                      lambda d, u, dt, tf, sp=sp: custom_figure(d, sp, u, dt, tf)))
    for i, p in enumerate(vars_list):
        cards.append((i, set(panel_vars(p)),
                      lambda d, u, dt, tf, p=p: make_panel_figure(d, p, u, dt, tf, color=SERIES_COLORS[0])))
    n = len(vars_list)
    cards += [
        (n, {WD_COL, WS_COL}, lambda d, *_: make_wind_rose(d)),
        (n + 1, {FCH4_COL, WD_COL}, lambda d, *_: make_fch4_vs_wd(d)),
        (n + 2, {FCH4_COL, FC_COL}, lambda d, *_: make_fch4_fc_regression(d)),
        (n + 3, {FCH4_COL, WD_COL}, lambda d, *_: make_fch4_surface(d)),
        (n + 4, {c for c, *_ in DIURNAL_VARS}, lambda d, *_: make_diurnal(d)),
        (n + 6, {FCH4_COL, WD_COL, SOURCE_FETCH_COL}, lambda d, *_: make_source_map(d)),
    ]
    return cards


def flux_notes(dff, dfq, qc_limits, sector, ustar):
    notes = [f"{lab}: {'none' if lim in (None, 'all') else QC_NOTE.get(lim, lim)}"
             for (_i, lab, _c, _q), lim in zip(QC_FILTERS, qc_limits)] + [
             ("u*: none" if ustar in (None, "none") else
              f"u* ≤ {ustar} m/s removed ({int((dff[USTAR_COL] <= float(ustar)).sum()) if USTAR_COL in dff else 0} half-hours)"),
             f"Wind direction: {sectors_label(sector)}",
             f"{n_records(dfq)} of {len(dff)} records",
             "valid " + ", ".join(f"{lab.replace('_QC', '')}: {int(dfq[c].notna().sum()) if c in dfq else 0}"
                                  for _i, lab, c, _q in QC_FILTERS)]
    chips = [html.Span(t, className="chip" + (" chip-strong" if "records" in t else "")) for t in notes]
    if n_records(dfq) == 0:
        chips.append(html.Span("No data left with these filters", className="chip chip-warn"))
    return chips


def columns_changed_by(trigger_ids):
    """Columns whose values change when the given filter controls change (None = all columns)."""
    cols = set()
    for tid in trigger_ids:
        if tid == "wd-sector":
            return None                                   # removes whole records
        if tid == "ustar-filter":
            cols |= set([q[2] for q in QC_FILTERS] + [ET_COL, "FH2O", "TAU", "Bowen_ratio"] + USTAR_EXTRA_COLS)
        for qid, _lab, col, _q in QC_FILTERS:
            if tid == qid:
                cols |= {col, "FH2O", ET_COL} if col == LE_COL else {col}
    return cols


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("dp-range", "start_date"),
    Input("dp-range", "end_date"),
    Input("refresh", "n_intervals"),
    Input("tgamon-display", "value"),
    *[State(qid, "value") for qid, _l, _c, _q in QC_FILTERS],
    State("wd-sector", "value"),
    State("ustar-filter", "value"),
    State("custom-specs", "data"),
)
def render_tab(tab_value, start_date, end_date, _n, tgamon_display, *args):
    """Builds the whole tab (only when the tab, dates or data change)."""
    nq = len(QC_FILTERS)
    qc_limits, sector, ustar, custom_specs = list(args[:nq]), args[nq], args[nq + 1], args[nq + 2]
    if tab_value == SETUP_TAB:
        return with_placeholders(setup_layout())
    if tab_value == TGAMON_TAB and ENABLE_TGAMON:
        return with_placeholders(tgamon_content(start_date, end_date, tgamon_display or "30min", custom_specs),
                                 custom=False)

    dff, units_map, vars_list = flux_frame(start_date, end_date)
    if dff is None:
        return with_placeholders(no_data_message())
    if dff.empty:
        return with_placeholders(empty_state("No data for the selected date range",
                                             "Pick a different start or end date in the toolbar above."))
    if not vars_list:
        return with_placeholders(empty_state("No variables to display on this tab"))

    dtick, tickformat = choose_axis_settings(dff)
    title_range = f"{start_date or ''} → {end_date or ''}".strip(" →")
    dfq = filter_flux_df(dff, qc_limits, sector, ustar)
    figs = {idx: make(dfq, units_map, dtick, tickformat)
            for idx, _cols, make in flux_card_specs(vars_list, custom_specs, dff)}
    n = len(vars_list)
    customs = sorted(i for i in figs if i >= CUSTOM_IDX0)
    return html.Div([
        html.Div(id="filter-summary", className="filter-summary",
                 children=flux_notes(dff, dfq, qc_limits, sector, ustar)),
        html.Div(id="custom-section", children=custom_block([plot_card(figs[i], i) for i in customs])),
        html.Div(className="section", children=[
            section_header("Time series", title_range),
            html.Div([plot_card(figs[i], i) for i in range(n)], className="card-grid"),
        ]),
        html.Div(className="section", children=[
            section_header("Wind rose, flux footprint and source direction",
                           f"EC tower at {TOWER_LAT:.6f}, {TOWER_LON:.6f}"),
            html.Div(className="card-grid card-grid-maps",
                     children=[plot_card(figs[n], n), plot_card(figs[n + 6], n + 6)]),
        ]),
        html.Div(className="section", children=[
            section_header("Wind and flux analysis", "CH\u2084 flux by wind direction, time of day and CO\u2082 flux"),
            html.Div(className="card-grid card-grid-analysis",
                     children=[plot_card(figs[i], i) for i in range(n + 1, n + 4)]),
        ]),
        html.Div(className="section", children=[
            section_header("Diurnal cycle"),
            plot_card(figs[n + 4], n + 4),
        ]),
    ])


def with_placeholders(content, custom=True):
    """Hidden 'filter-summary' / 'custom-section' boxes so the filter and add-plot callbacks
    always have their targets, even on tabs that don't show them."""
    hidden = {"display": "none"}
    extra = [html.Div(id="filter-summary", style=hidden)]
    if custom:
        extra.append(html.Div(id="custom-section", style=hidden))
    return html.Div([content, *extra])


def custom_block(cards):
    if not cards:
        return None
    return html.Div(className="section", children=[
        section_header("Added plots"),
        html.Div(cards, className="card-grid card-grid-wide"),
    ])


@app.callback(
    Output({"type": "card-graph", "index": ALL}, "figure"),
    Output("filter-summary", "children"),
    *[Input(qid, "value") for qid, _l, _c, _q in QC_FILTERS],
    Input("wd-sector", "value"),
    Input("ustar-filter", "value"),
    State("tabs", "value"),
    State("dp-range", "start_date"),
    State("dp-range", "end_date"),
    State("custom-specs", "data"),
    State({"type": "card-graph", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def apply_flux_filters(*args):
    """QC / u* / wind-direction change: re-draw only the plots that use the affected columns;
    every other plot keeps its current figure (and zoom)."""
    from dash import ctx, no_update
    nq = len(QC_FILTERS)
    qc_limits, sector, ustar = list(args[:nq]), args[nq], args[nq + 1]
    tab_value, start_date, end_date, custom_specs, ids = args[nq + 2:]
    ids = ids or []
    if tab_value != FLUX_TAB or not ids:
        return [no_update] * len(ids), no_update
    dff, units_map, vars_list = flux_frame(start_date, end_date)
    if dff is None or dff.empty:
        return [no_update] * len(ids), no_update

    changed = columns_changed_by([t.split(".")[0] for t in ctx.triggered_prop_ids])
    dfq = filter_flux_df(dff, qc_limits, sector, ustar)
    dtick, tickformat = choose_axis_settings(dff)
    specs = {idx: (cols, make) for idx, cols, make in flux_card_specs(vars_list, custom_specs, dff)}
    out = []
    for cid in ids:
        idx = cid.get("index")
        if idx not in specs or (changed is not None and not (specs[idx][0] & changed)):
            out.append(no_update)
            continue
        out.append(prettify_fig(specs[idx][1](dfq, units_map, dtick, tickformat)))
    return out, flux_notes(dff, dfq, qc_limits, sector, ustar)


@app.callback(
    Output("custom-section", "children"),
    Input("custom-specs", "data"),
    State("tabs", "value"),
    State("dp-range", "start_date"),
    State("dp-range", "end_date"),
    *[State(qid, "value") for qid, _l, _c, _q in QC_FILTERS],
    State("wd-sector", "value"),
    State("ustar-filter", "value"),
    State("tgamon-display", "value"),
    prevent_initial_call=True,
)
def update_added_plots(custom_specs, tab_value, start_date, end_date, *args):
    """Add / clear plot: re-draw only the 'Added plots' section, not the whole tab."""
    from dash import no_update
    nq = len(QC_FILTERS)
    qc_limits, sector, ustar, tgamon_display = list(args[:nq]), args[nq], args[nq + 1], args[nq + 2]
    if tab_value == TGAMON_TAB and ENABLE_TGAMON:
        df, units_map, _src = load_tgamon()
        dff, _d, _x = tgamon_frame(df, start_date, end_date, tgamon_display or "30min")
        if dff.empty:
            return no_update
        dtick, tickformat = choose_axis_settings(dff)
        return custom_section(dff, specs_for_tab(custom_specs, TGAMON_TAB), units_map, dtick, tickformat)
    if tab_value != FLUX_TAB:
        return no_update
    dff, units_map, _vars = flux_frame(start_date, end_date)
    if dff is None or dff.empty:
        return no_update
    dfq = filter_flux_df(dff, qc_limits, sector, ustar)
    dtick, tickformat = choose_axis_settings(dff)
    specs = valid_custom_specs(dff, specs_for_tab(custom_specs, FLUX_TAB))
    return custom_block([plot_card(custom_figure(dfq, sp, units_map, dtick, tickformat), CUSTOM_IDX0 + i)
                         for i, sp in enumerate(specs)])


def empty_state(title, detail=None):
    return html.Div(className="empty-state", children=[
        html.Div(title, className="empty-title"),
        html.Div(detail, className="empty-detail") if detail else None,
    ])


# "Done" closes the wind-direction pop-up (runs in the browser; works even without assets/wd_dropdown.js)
app.clientside_callback(
    """
    function(n) {
        var d = document.getElementById("wd-details");
        if (d && n) { d.open = false; d.removeAttribute("open"); }
        return window.dash_clientside.no_update;
    }
    """,
    Output("wd-done", "title"),
    Input("wd-done", "n_clicks"),
    prevent_initial_call=True,
)


# Maximize: copy the clicked card's figure into the pop-up (runs in the browser, no server call)
app.clientside_callback(
    """
    function(maxClicks, closeClicks, figs) {
        const hidden = %s;
        const shown = Object.assign({}, hidden, {display: "flex"});
        const ctx = dash_clientside.callback_context;
        if (!ctx.triggered || !ctx.triggered.length) { return [dash_clientside.no_update, dash_clientside.no_update]; }
        const t = ctx.triggered[0];
        if (t.prop_id.startsWith("close-modal-btn")) { return [hidden, {}]; }
        if (!t.value) { return [dash_clientside.no_update, dash_clientside.no_update]; }
        const id = JSON.parse(t.prop_id.split(".")[0]);
        const btns = ctx.inputs_list[0];
        const pos = btns.findIndex(b => b.id.index === id.index);
        if (pos < 0 || !figs[pos]) { return [dash_clientside.no_update, dash_clientside.no_update]; }
        const fig = JSON.parse(JSON.stringify(figs[pos]));
        fig.layout = Object.assign({}, fig.layout, {height: null, autosize: true});
        if (fig.layout.title && fig.layout.title.font) { fig.layout.title.font.size = 18; }
        return [shown, fig];
    }
    """ % __import__("json").dumps(MODAL_HIDDEN),
    Output("modal-overlay", "style"),
    Output("zoom-graph", "figure"),
    Input({"type": "max-btn", "index": ALL}, "n_clicks"),
    Input("close-modal-btn", "n_clicks"),
    State({"type": "card-graph", "index": ALL}, "figure"),
    prevent_initial_call=True,
)


# =========================
# Local run (Render uses gunicorn instead)
# =========================
# VS Code: open this folder, run  pip install -r requirements.txt , then press F5 / "Run Python File".
# Optional: set FLUX_LOCAL_FILE=C:\Campbellsci\LoggerNet\Data\Cattle_Experiment_Eagle_TGA310_AmrFlux.dat
# to read the LoggerNet file directly instead of GitHub.
if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8050"))
    print(f"Dash app running at: http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}")
    app.server.run(host=host, port=port, debug=False, use_reloader=False)
