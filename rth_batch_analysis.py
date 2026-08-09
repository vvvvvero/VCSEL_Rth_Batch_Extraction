#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
VCSEL thermal resistance (Rth) batch analysis.

Works on any wafer / test run that follows this layout under --base
(folder and wafer-prefix names are auto-detected, not hardcoded):

  B_LIV_T\B_LIV_{T}Deg\ <prefix>_row_RR_col_CC_site_SSS_<date>_<time>.csv
      columns: Site,Row,Column,Point,...,Voltage_V,Current_A,Optical_Power_W,...

  Spectrometer_T\<prefix>_Col{c}_Row{r}_{pulse}ms_{T}Deg_sync_<date>_<time>\
      summary.csv                      -> Point,...,Voltage_V,Current_A,Peak_nm,Peak_Intensity,Avg_Intensity,...
      spectra\spectrum_point_NNNN.csv  -> Relative_Time_s,Wavelength_nm,Intensity

Physics
-------
  Pdiss = I * V - P_opt(I)          (P_opt interpolated from the LIV sweep)

  C1 = d(lambda_peak) / d(T_plate)  at constant Pdiss              [nm/K]
  C2 = d(lambda_peak) / d(Pdiss)    at constant T_plate            [nm/W  -> reported also nm/mW]

  Rth = C2 / C1                     [K/W]  ( = K/mW when C2 is in nm/mW )

Outputs (under --base\RTH_Analysis, or --out if given):
  results_device.csv      one row per device: C1, C2@Tref, Rth, Ith, quality metrics
  results_C2_by_T.csv     one row per device+temperature
  results_lambda_vs_T.csv the lambda(T) points at Pdiss_ref used for the C1 fit
  plots\<device>\         per-device figures (LIV vs T, spectra vs T, thermal fits)
  summary_*.png           wafer-level summary figures

Usage:
  python rth_batch_analysis.py --base D:\path\to\WAFER_ID          # everything
  python rth_batch_analysis.py --base D:\...\WAFER_ID --devices R10C1,R11C2
  python rth_batch_analysis.py --base D:\...\WAFER_ID --no-device-plots  # fits + summaries only (fast)
  python rth_batch_analysis.py --base D:\...\WAFER_ID --max-devices 5   # quick smoke test

See README.md for the full option list, the fit method, and tuning knobs.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import glob
import math
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.exceptions.RankWarning
                        if hasattr(np, "exceptions") else RuntimeWarning)

# ======================================================================
# CONFIGURATION
# ======================================================================

# Defaults below are only used when the script is imported (e.g. from a
# debug session) without going through main(). Normal runs must pass
# --base on the command line -- nothing wafer-specific is hardcoded.
BASE_DIR   = "."
LIV_SUBDIR = "B_LIV_T"
SPEC_SUBDIR = "Spectrometer_T"
OUT_SUBDIR  = "RTH_Analysis"
LIV_DIR    = os.path.join(BASE_DIR, LIV_SUBDIR)
SPEC_DIR   = os.path.join(BASE_DIR, SPEC_SUBDIR)
OUT_DIR    = os.path.join(BASE_DIR, OUT_SUBDIR)

# --- spectrometer run selection -------------------------------------
PULSE_FILTER      = "0.01ms"   # only use runs with this pulse-width tag; None = any
EXCLUDE_TOKENS    = ("aft",)   # skip post-stress / re-measured runs
# when several runs exist for the same (device, T): keep the latest timestamp

# --- optical power source for Pdiss ---------------------------------
# "20C"   : always use the 20 degC LIV curve  (user default -- one consistent P_opt(I))
# "match" : use the LIV measured at the same plate temperature, fall back to 20 degC
POPT_SOURCE       = "20C"
LIV_FALLBACK_T    = 20

# --- lasing / fit window --------------------------------------------
ITH_MARGIN        = 1.20   # only fit points with I > ITH_MARGIN * Ith
SNR_MIN           = 3.0    # require Peak_Intensity / Avg_Intensity > SNR_MIN
MIN_FIT_POINTS    = 6      # minimum spectra points for a C2 fit
SAT_COUNTS        = 60000  # flag possible detector saturation above this
I_MAX_MARGIN      = 1.02   # drop spectrometer points beyond LIV Imax * this

# --- C1 extraction ---------------------------------------------------
MIN_T_FOR_C1      = 3      # need at least this many temperatures
PREF_FRAC         = 0.50   # Pdiss_ref = lo + PREF_FRAC*(hi-lo) inside the common overlap
C2_T_REF          = 20     # temperature whose C2 is used for the headline Rth

# --- plotting --------------------------------------------------------
N_SPEC_CURVES     = 6      # spectra overlaid per temperature panel
DB_FLOOR          = -40.0  # dB axis floor
DB_NORM           = "per_panel"   # "per_panel" (show intensity growth) or "per_curve"
DPI               = 130

TEMP_CMAP         = "turbo"
CURR_CMAP         = "viridis"

# ======================================================================
# PARSING HELPERS
# ======================================================================

RE_LIV_FILE  = re.compile(r"row_(\d+)_col_(\d+)", re.I)
RE_LIV_DIR   = re.compile(r"B_LIV_(\d+)\s*Deg", re.I)
RE_SPEC_DIR  = re.compile(
    r"Col(\d+)_Row(\d+)_([0-9.]+)ms_(\d+)\s*Deg", re.I)
RE_TIMESTAMP = re.compile(r"(\d{8}_\d{6})")


def dev_key(row: int, col: int) -> str:
    return "R%02d_C%02d" % (row, col)


def _timestamp(name: str) -> str:
    m = RE_TIMESTAMP.search(name)
    return m.group(1) if m else ""


# ======================================================================
# INDEXING
# ======================================================================

def index_liv(liv_dir: str) -> dict:
    """-> {(row, col): {T: filepath}}"""
    idx = defaultdict(dict)
    for sub in sorted(os.listdir(liv_dir)):
        subpath = os.path.join(liv_dir, sub)
        if not os.path.isdir(subpath):
            continue
        mt = RE_LIV_DIR.search(sub)
        if not mt:
            continue
        T = int(mt.group(1))
        for f in glob.glob(os.path.join(subpath, "*.csv")):
            mf = RE_LIV_FILE.search(os.path.basename(f))
            if not mf:
                continue
            row, col = int(mf.group(1)), int(mf.group(2))
            prev = idx[(row, col)].get(T)
            # keep the latest measurement if duplicated
            if prev is None or _timestamp(os.path.basename(f)) > _timestamp(os.path.basename(prev)):
                idx[(row, col)][T] = f
    return dict(idx)


def index_spectra(spec_dir: str) -> dict:
    """-> {(row, col): {T: folderpath}}"""
    idx = defaultdict(dict)
    for name in sorted(os.listdir(spec_dir)):
        path = os.path.join(spec_dir, name)
        if not os.path.isdir(path):
            continue
        low = name.lower()
        if any(tok in low for tok in EXCLUDE_TOKENS):
            continue
        m = RE_SPEC_DIR.search(name)
        if not m:
            continue                       # e.g. the older 5505_B_C10_* debug runs
        col, row, pulse, T = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
        if PULSE_FILTER is not None and (pulse + "ms") != PULSE_FILTER:
            continue
        if not os.path.isfile(os.path.join(path, "summary.csv")):
            continue
        prev = idx[(row, col)].get(T)
        if prev is None or _timestamp(name) > _timestamp(os.path.basename(prev)):
            idx[(row, col)][T] = path
    return dict(idx)


# ======================================================================
# LOADERS
# ======================================================================

def load_liv(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    need = {"Voltage_V", "Current_A", "Optical_Power_W"}
    if not need.issubset(df.columns):
        return None
    df = df[["Current_A", "Voltage_V", "Optical_Power_W"]].apply(
        pd.to_numeric, errors="coerce").dropna()
    df = df[df["Current_A"] >= 0].sort_values("Current_A")
    # collapse duplicate current points (np.interp needs strictly increasing x)
    df = df.groupby("Current_A", as_index=False).mean()
    return df if len(df) >= 5 else None


def load_summary(folder: str) -> pd.DataFrame | None:
    f = os.path.join(folder, "summary.csv")
    try:
        df = pd.read_csv(f, comment="#", skip_blank_lines=True)
    except Exception:
        return None
    if "Peak_nm" not in df.columns or "Current_A" not in df.columns:
        return None
    for c in ("Point", "Setpoint", "Voltage_V", "Current_A",
              "Peak_nm", "Peak_Intensity", "Avg_Intensity"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Current_A", "Peak_nm"])


def load_spectrum(folder: str, point: int) -> pd.DataFrame | None:
    f = os.path.join(folder, "spectra", "spectrum_point_%04d.csv" % int(point))
    if not os.path.isfile(f):
        return None
    try:
        df = pd.read_csv(f, comment="#", skip_blank_lines=True)
    except Exception:
        return None
    if "Wavelength_nm" not in df.columns or "Intensity" not in df.columns:
        return None
    df = df[["Wavelength_nm", "Intensity"]].apply(pd.to_numeric, errors="coerce").dropna()
    return df if len(df) > 10 else None


# ======================================================================
# LIV ANALYSIS
# ======================================================================

def liv_metrics(df: pd.DataFrame) -> dict:
    """
    Threshold current, slope efficiency and rollover point.

    Ith is taken by max-slope linear extrapolation: the L-I curve is fitted
    over the window where dP/dI stays close to its maximum (i.e. the straight
    part of the ramp, safely below thermal rollover) and extrapolated to P=0.
    Fitting the top 50% of the ramp instead would sit in the sub-linear
    rollover region and yields a positive intercept / negative Ith.
    """
    I = df["Current_A"].to_numpy(dtype=float)
    P = df["Optical_Power_W"].to_numpy(dtype=float)
    out = {"Ith_A": np.nan, "SE_W_per_A": np.nan,
           "I_rollover_A": np.nan, "Pmax_W": np.nan, "Ith_method": "none"}
    if len(I) < 8 or not np.isfinite(P).any():
        return out

    k = int(np.argmax(P))
    out["I_rollover_A"] = I[k]
    out["Pmax_W"] = P[k]

    Ir, Pr = I[:k + 1], P[:k + 1]
    if len(Ir) < 6 or Pr.max() <= 0:
        return out

    # light smoothing before differentiating (measurement noise on P)
    w = 3
    kern = np.ones(w) / w
    Ps = np.convolve(Pr, kern, mode="same")
    Ps[:w] = Pr[:w]
    Ps[-w:] = Pr[-w:]
    with np.errstate(invalid="ignore", divide="ignore"):
        dPdI = np.gradient(Ps, Ir)
    dPdI[~np.isfinite(dPdI)] = 0.0

    j = int(np.argmax(dPdI))
    smax = dPdI[j]
    if smax <= 0:
        return out

    # grow a contiguous window around the max-slope point
    lo = hi = j
    while lo > 0 and dPdI[lo - 1] >= 0.70 * smax:
        lo -= 1
    while hi < len(dPdI) - 1 and dPdI[hi + 1] >= 0.70 * smax:
        hi += 1
    sel = slice(lo, hi + 1)

    if hi - lo + 1 >= 4:
        slope, intercept = np.polyfit(Ir[sel], Pr[sel], 1)
        if slope > 0:
            ith = -intercept / slope
            out["SE_W_per_A"] = float(slope)
            if 0 <= ith < I.max():
                out["Ith_A"] = float(ith)
                out["Ith_method"] = "max_slope"
                return out

    # fallback: current where the L-I crosses 10% of the peak power
    tgt = 0.10 * Pr.max()
    above = np.nonzero(Pr >= tgt)[0]
    if len(above):
        i0 = above[0]
        if i0 == 0:
            out["Ith_A"] = float(Ir[0])
        else:
            f = (tgt - Pr[i0 - 1]) / max(Pr[i0] - Pr[i0 - 1], 1e-18)
            out["Ith_A"] = float(Ir[i0 - 1] + f * (Ir[i0] - Ir[i0 - 1]))
        out["Ith_method"] = "P10pct"
    return out


def popt_interp(liv: pd.DataFrame, I_query: np.ndarray) -> np.ndarray:
    """P_opt at the requested currents; NaN beyond the measured LIV range."""
    Il = liv["Current_A"].to_numpy()
    Pl = liv["Optical_Power_W"].to_numpy()
    P = np.interp(I_query, Il, Pl, left=np.nan, right=np.nan)
    P[I_query > Il.max() * I_MAX_MARGIN] = np.nan
    P[I_query < 0] = np.nan
    return P


# ======================================================================
# PER-(DEVICE, TEMPERATURE) SPECTRAL ANALYSIS
# ======================================================================

def analyse_run(folder: str, liv: pd.DataFrame, ith: float) -> dict | None:
    """Return the lasing-window lambda(Pdiss) table and the C2 linear fit."""
    s = load_summary(folder)
    if s is None or len(s) < MIN_FIT_POINTS:
        return None

    I = s["Current_A"].to_numpy(dtype=float)
    V = s["Voltage_V"].to_numpy(dtype=float) if "Voltage_V" in s else \
        s["Setpoint"].to_numpy(dtype=float)
    lam = s["Peak_nm"].to_numpy(dtype=float)
    pk = s["Peak_Intensity"].to_numpy(dtype=float) if "Peak_Intensity" in s \
        else np.full_like(I, np.nan)
    av = s["Avg_Intensity"].to_numpy(dtype=float) if "Avg_Intensity" in s \
        else np.full_like(I, np.nan)
    pts = s["Point"].to_numpy(dtype=int) if "Point" in s else np.arange(len(s))

    Popt = popt_interp(liv, I)
    Pdiss = I * V - Popt

    snr = np.where(av > 0, pk / av, np.nan)

    lasing = np.isfinite(Pdiss) & (Pdiss > 0) & np.isfinite(lam)
    lasing &= (snr > SNR_MIN)
    if np.isfinite(ith):
        lasing &= (I > ITH_MARGIN * ith)
    # stay below thermal rollover of the LIV so P_opt stays meaningful
    lasing &= (I <= liv["Current_A"].max() * I_MAX_MARGIN)

    tbl = pd.DataFrame({
        "Point": pts, "Current_A": I, "Voltage_V": V,
        "Popt_W": Popt, "Pdiss_W": Pdiss, "Peak_nm": lam,
        "Peak_Intensity": pk, "SNR": snr, "lasing": lasing,
    })

    res = {"table": tbl, "n_fit": int(lasing.sum()),
           "C2_nm_per_W": np.nan, "C2_intercept_nm": np.nan, "C2_R2": np.nan,
           "Pdiss_lo_W": np.nan, "Pdiss_hi_W": np.nan,
           "saturated": bool(np.nanmax(pk) > SAT_COUNTS) if np.isfinite(pk).any() else False}

    if lasing.sum() < MIN_FIT_POINTS:
        return res

    x = Pdiss[lasing]
    y = lam[lasing]
    order = np.argsort(x)
    x, y = x[order], y[order]
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    res.update(C2_nm_per_W=float(slope),
               C2_intercept_nm=float(intercept),
               C2_R2=(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
               Pdiss_lo_W=float(x.min()), Pdiss_hi_W=float(x.max()))
    return res


# ======================================================================
# C1 / Rth
# ======================================================================

def choose_pdiss_ref(ranges: dict) -> tuple:
    """
    ranges: {T: (Pdiss_lo, Pdiss_hi)}
    Pick the largest set of temperatures whose Pdiss fit windows overlap,
    then take a point inside that common window.
    Returns (Pref, used_temperatures, lo, hi) or (nan, [], nan, nan).
    """
    Ts = sorted(ranges)
    best = None
    n = len(Ts)
    for i in range(n):
        for j in range(n, i, -1):
            sub = Ts[i:j]
            if len(sub) < MIN_T_FOR_C1:
                continue
            lo = max(ranges[t][0] for t in sub)
            hi = min(ranges[t][1] for t in sub)
            if hi > lo:
                cand = (len(sub), -i, sub, lo, hi)   # prefer more T, then lower start
                if best is None or cand[:2] > best[:2]:
                    best = cand
    if best is None:
        return np.nan, [], np.nan, np.nan
    _, _, sub, lo, hi = best
    return lo + PREF_FRAC * (hi - lo), sub, lo, hi


def fit_C1(per_T: dict) -> dict:
    """
    per_T: {T: run-result dict with a valid C2 fit}
    lambda at the common Pdiss_ref is evaluated from each temperature's own
    C2 regression line (less noisy than interpolating raw points).
    """
    ranges = {T: (r["Pdiss_lo_W"], r["Pdiss_hi_W"])
              for T, r in per_T.items() if np.isfinite(r["C2_nm_per_W"])}
    out = {"C1_nm_per_K": np.nan, "C1_R2": np.nan, "Pdiss_ref_W": np.nan,
           "C1_T_used": [], "lambda_at_ref": {}}
    if len(ranges) < MIN_T_FOR_C1:
        return out

    Pref, Ts_used, _, _ = choose_pdiss_ref(ranges)
    if not np.isfinite(Pref):
        return out

    Tarr, Larr = [], []
    for T in Ts_used:
        r = per_T[T]
        lam = r["C2_nm_per_W"] * Pref + r["C2_intercept_nm"]
        Tarr.append(float(T))
        Larr.append(float(lam))
        out["lambda_at_ref"][T] = float(lam)

    Tarr, Larr = np.asarray(Tarr), np.asarray(Larr)
    slope, intercept = np.polyfit(Tarr, Larr, 1)
    yhat = slope * Tarr + intercept
    ss_res = float(np.sum((Larr - yhat) ** 2))
    ss_tot = float(np.sum((Larr - Larr.mean()) ** 2))

    out.update(C1_nm_per_K=float(slope),
               C1_intercept_nm=float(intercept),
               C1_R2=(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
               Pdiss_ref_W=float(Pref),
               C1_T_used=[int(t) for t in Ts_used])
    return out


# ======================================================================
# PLOTTING
# ======================================================================

def _tcolors(temps):
    cmap = plt.get_cmap(TEMP_CMAP)
    if len(temps) == 1:
        return {temps[0]: cmap(0.5)}
    norm = Normalize(min(temps), max(temps))
    return {T: cmap(norm(T)) for T in temps}


def plot_liv(dev, liv_by_T, metrics_by_T, outdir):
    temps = sorted(liv_by_T)
    if not temps:
        return
    colors = _tcolors(temps)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax2 = ax.twinx()
    for T in temps:
        df = liv_by_T[T]
        I_mA = df["Current_A"] * 1e3
        ax.plot(I_mA, df["Optical_Power_W"] * 1e3, "-", color=colors[T],
                lw=1.6, label="%d $^\\circ$C" % T)
        ax2.plot(I_mA, df["Voltage_V"], "--", color=colors[T], lw=1.0, alpha=0.55)
        ith = metrics_by_T.get(T, {}).get("Ith_A", np.nan)
        if np.isfinite(ith):
            ax.axvline(ith * 1e3, color=colors[T], lw=0.6, ls=":", alpha=0.5)

    ax.set_xlabel("Current (mA)")
    ax.set_ylabel("Optical power (mW)")
    ax2.set_ylabel("Voltage (V)")
    ax.set_title("%s   L-I-V vs plate temperature\n(solid = L, dashed = V, dotted = $I_{th}$)"
                 % dev, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, title="Plate T", title_fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "%s_LIV_vs_T.png" % dev), dpi=DPI)
    plt.close(fig)


def plot_spectra(dev, spec_by_T, run_by_T, outdir):
    temps = sorted(spec_by_T)
    if not temps:
        return
    ncol = min(4, len(temps))
    nrow = int(math.ceil(len(temps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.5 * nrow),
                             squeeze=False, sharex=False)
    cmap = plt.get_cmap(CURR_CMAP)

    for k, T in enumerate(temps):
        ax = axes[k // ncol][k % ncol]
        folder = spec_by_T[T]
        run = run_by_T.get(T)
        if run is None:
            ax.set_visible(False)
            continue
        tbl = run["table"]
        las = tbl[tbl["lasing"]]
        if len(las) == 0:
            las = tbl[np.isfinite(tbl["Peak_nm"])].tail(N_SPEC_CURVES)
        if len(las) == 0:
            ax.set_visible(False)
            continue

        sel = las.iloc[np.linspace(0, len(las) - 1,
                                   min(N_SPEC_CURVES, len(las))).astype(int)]
        currents = sel["Current_A"].to_numpy() * 1e3
        norm = Normalize(currents.min(), currents.max() if currents.max() > currents.min()
                         else currents.min() + 1)

        curves = []
        for _, r in sel.iterrows():
            sp = load_spectrum(folder, int(r["Point"]))
            if sp is None:
                continue
            w = sp["Wavelength_nm"].to_numpy()
            y = sp["Intensity"].to_numpy().astype(float)
            base = np.percentile(y, 5)            # dark / stray-light floor
            y = np.maximum(y - base, 1e-3)
            curves.append((r["Current_A"] * 1e3, w, y))

        if not curves:
            ax.set_visible(False)
            continue

        gmax = max(c[2].max() for c in curves)
        for I_mA, w, y in curves:
            ref = gmax if DB_NORM == "per_panel" else y.max()
            ax.plot(w, 10 * np.log10(y / ref), lw=0.9,
                    color=cmap(norm(I_mA)), label="%.1f mA" % I_mA)

        # zoom around the emission band
        allw = np.concatenate([c[1] for c in curves])
        pk = np.concatenate([[c[1][np.argmax(c[2])]] for c in curves])
        lo, hi = pk.min() - 6, pk.max() + 6
        ax.set_xlim(max(allw.min(), lo), min(allw.max(), hi))
        ax.set_ylim(DB_FLOOR, 3)
        ax.set_title("%d $^\\circ$C" % T, fontsize=10)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Relative intensity (dB)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2, loc="upper left")

    for k in range(len(temps), nrow * ncol):
        axes[k // ncol][k % ncol].set_visible(False)

    fig.suptitle("%s   spectra vs current at each plate temperature "
                 "(0 dB = %s max)" % (dev, "panel" if DB_NORM == "per_panel" else "curve"),
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(outdir, "%s_spectra_vs_T.png" % dev), dpi=DPI)
    plt.close(fig)


def plot_thermal_fits(dev, run_by_T, c1, outdir):
    temps = sorted(t for t in run_by_T if np.isfinite(run_by_T[t]["C2_nm_per_W"]))
    if not temps:
        return
    colors = _tcolors(temps)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # (a) lambda vs Pdiss with the C2 fits
    ax = axes[0]
    for T in temps:
        r = run_by_T[T]
        tbl = r["table"]
        las = tbl[tbl["lasing"]]
        ax.plot(las["Pdiss_W"] * 1e3, las["Peak_nm"], "o", ms=3,
                color=colors[T], alpha=0.65)
        xs = np.linspace(r["Pdiss_lo_W"], r["Pdiss_hi_W"], 50)
        ax.plot(xs * 1e3, r["C2_nm_per_W"] * xs + r["C2_intercept_nm"],
                "-", color=colors[T], lw=1.5,
                label="%d $^\\circ$C: %.4f nm/mW" % (T, r["C2_nm_per_W"] / 1e3))
    if np.isfinite(c1.get("Pdiss_ref_W", np.nan)):
        ax.axvline(c1["Pdiss_ref_W"] * 1e3, color="k", ls="--", lw=1.0,
                   label="$P_{diss,ref}$ = %.2f mW" % (c1["Pdiss_ref_W"] * 1e3))
    ax.set_xlabel("Dissipated power $P_{diss}=IV-P_{opt}$ (mW)")
    ax.set_ylabel("Peak wavelength (nm)")
    ax.set_title("(a) $C_2$: $d\\lambda/dP_{diss}$ at fixed $T$", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # (b) lambda at Pdiss_ref vs plate T -> C1
    ax = axes[1]
    lam_ref = c1.get("lambda_at_ref", {})
    if lam_ref:
        Ts = np.array(sorted(lam_ref))
        Ls = np.array([lam_ref[t] for t in Ts])
        ax.plot(Ts, Ls, "o", ms=6, color="tab:blue")
        if np.isfinite(c1.get("C1_nm_per_K", np.nan)):
            xs = np.linspace(Ts.min(), Ts.max(), 20)
            ax.plot(xs, c1["C1_nm_per_K"] * xs + c1["C1_intercept_nm"], "-",
                    color="tab:red", lw=1.6,
                    label="$C_1$ = %.4f nm/K  ($R^2$=%.4f)"
                          % (c1["C1_nm_per_K"], c1["C1_R2"]))
            ax.legend(fontsize=8)
    ax.set_xlabel("Plate temperature ($^\\circ$C)")
    ax.set_ylabel("Peak wavelength @ $P_{diss,ref}$ (nm)")
    ax.set_title("(b) $C_1$: $d\\lambda/dT$ at fixed $P_{diss}$", fontsize=10)
    ax.grid(alpha=0.3)

    # (c) C2 and the implied Rth vs temperature
    ax = axes[2]
    c2_mW = [run_by_T[T]["C2_nm_per_W"] / 1e3 for T in temps]
    ax.plot(temps, c2_mW, "s-", color="tab:green", ms=6, label="$C_2$")
    ax.set_xlabel("Plate temperature ($^\\circ$C)")
    ax.set_ylabel("$C_2$ (nm/mW)", color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")
    ax.grid(alpha=0.3)
    C1 = c1.get("C1_nm_per_K", np.nan)
    if np.isfinite(C1) and C1 != 0:
        ax2 = ax.twinx()
        ax2.plot(temps, [c / C1 for c in c2_mW], "^--", color="tab:purple", ms=6)
        ax2.set_ylabel("$R_{th}=C_2/C_1$ (K/mW)", color="tab:purple")
        ax2.tick_params(axis="y", labelcolor="tab:purple")
    ax.set_title("(c) $C_2$ and $R_{th}$ vs temperature", fontsize=10)

    fig.suptitle("%s   thermal-resistance extraction" % dev, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(outdir, "%s_thermal_fits.png" % dev), dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- summary

def plot_summary(dev_df: pd.DataFrame, c2_df: pd.DataFrame, outdir: str,
                 wafer_label: str = "wafer"):
    # --- C2 vs T, every device + the population mean -----------------
    if len(c2_df):
        fig, ax = plt.subplots(figsize=(8, 5.2))
        for dev, g in c2_df.groupby("device"):
            g = g.sort_values("T_plate_C")
            ax.plot(g["T_plate_C"], g["C2_nm_per_mW"], "-", lw=0.7,
                    color="0.7", alpha=0.6, zorder=1)
        stat = c2_df.groupby("T_plate_C")["C2_nm_per_mW"].agg(["mean", "std", "count"])
        ax.errorbar(stat.index, stat["mean"], yerr=stat["std"], fmt="o-",
                    color="tab:red", lw=2.2, ms=7, capsize=4, zorder=3,
                    label="mean $\\pm$ 1$\\sigma$")
        for T, r in stat.iterrows():
            ax.annotate("%.4f\n(n=%d)" % (r["mean"], r["count"]),
                        (T, r["mean"]), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=7)
        ax.set_xlabel("Plate temperature ($^\\circ$C)")
        ax.set_ylabel("$C_2 = d\\lambda/dP_{diss}$ (nm/mW)")
        ax.set_title("$C_2$ vs plate temperature -- all devices (grey) and population mean")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "summary_C2_vs_T.png"), dpi=DPI)
        plt.close(fig)

        # boxplot version
        fig, ax = plt.subplots(figsize=(8, 5))
        temps = sorted(c2_df["T_plate_C"].unique())
        data = [c2_df.loc[c2_df["T_plate_C"] == T, "C2_nm_per_mW"].dropna().values
                for T in temps]
        labels = [str(int(t)) for t in temps]
        try:                                    # matplotlib >= 3.9
            ax.boxplot(data, tick_labels=labels, showmeans=True)
        except TypeError:                       # matplotlib < 3.9
            ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_xlabel("Plate temperature ($^\\circ$C)")
        ax.set_ylabel("$C_2$ (nm/mW)")
        ax.set_title("$C_2$ distribution per temperature (n=%d devices)"
                     % c2_df["device"].nunique())
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "summary_C2_boxplot.png"), dpi=DPI)
        plt.close(fig)

    # --- histograms of C1 / C2@Tref / Rth ----------------------------
    specs = [("C1_nm_per_K", "$C_1$ (nm/K)"),
             ("C2_at_Tref_nm_per_mW", "$C_2$ @ %d$^\\circ$C (nm/mW)" % C2_T_REF),
             ("Rth_K_per_W", "$R_{th}$ (K/W)")]
    avail = [(c, l) for c, l in specs
             if c in dev_df and dev_df[c].notna().sum() > 1]
    if avail:
        fig, axes = plt.subplots(1, len(avail), figsize=(5 * len(avail), 4))
        axes = np.atleast_1d(axes)
        for ax, (col, lab) in zip(axes, avail):
            v = dev_df[col].dropna().values
            ax.hist(v, bins=min(25, max(5, len(v) // 3)),
                    color="tab:blue", edgecolor="k", alpha=0.8)
            ax.axvline(np.median(v), color="tab:red", lw=1.8,
                       label="median = %.4g" % np.median(v))
            ax.set_xlabel(lab)
            ax.set_ylabel("devices")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.suptitle("%s -- thermal parameter distributions" % wafer_label, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(outdir, "summary_histograms.png"), dpi=DPI)
        plt.close(fig)

    # --- wafer maps ---------------------------------------------------
    maps = [("Rth_K_per_W", "$R_{th}$ (K/W)"),
            ("C1_nm_per_K", "$C_1$ (nm/K)"),
            ("C2_at_Tref_nm_per_mW", "$C_2$ @ %d$^\\circ$C (nm/mW)" % C2_T_REF)]
    maps = [(c, l) for c, l in maps if c in dev_df and dev_df[c].notna().any()]
    if maps and {"row", "col"}.issubset(dev_df.columns):
        rmin, rmax = int(dev_df["row"].min()), int(dev_df["row"].max())
        cmin, cmax = int(dev_df["col"].min()), int(dev_df["col"].max())
        fig, axes = plt.subplots(1, len(maps), figsize=(4.6 * len(maps), 6.5))
        axes = np.atleast_1d(axes)
        for ax, (col, lab) in zip(axes, maps):
            grid = np.full((rmax - rmin + 1, cmax - cmin + 1), np.nan)
            for _, r in dev_df.iterrows():
                if np.isfinite(r[col]):
                    grid[int(r["row"]) - rmin, int(r["col"]) - cmin] = r[col]
            im = ax.imshow(grid, cmap="viridis", aspect="auto", origin="upper",
                           extent=(cmin - 0.5, cmax + 0.5, rmax + 0.5, rmin - 0.5))
            fig.colorbar(im, ax=ax, label=lab, fraction=0.046)
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            ax.set_xticks(range(cmin, cmax + 1))
            ax.set_title(lab, fontsize=10)
        fig.suptitle("Wafer maps -- %s" % wafer_label, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(outdir, "summary_wafermaps.png"), dpi=DPI)
        plt.close(fig)


# ======================================================================
# MAIN
# ======================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description="VCSEL Rth batch analysis")
    ap.add_argument("--base", required=True,
                    help=r"wafer test folder, e.g. D:\2026\VCSEL\5505 "
                         r"(must contain the LIV and spectrometer subfolders)")
    ap.add_argument("--out", default=None,
                    help="output folder (default: <base>\\RTH_Analysis)")
    ap.add_argument("--liv-subdir", default=LIV_SUBDIR,
                    help="LIV-vs-temperature subfolder name under --base")
    ap.add_argument("--spec-subdir", default=SPEC_SUBDIR,
                    help="spectrometer-vs-temperature subfolder name under --base")
    ap.add_argument("--label", default=None,
                    help="wafer label for plot titles (default: basename of --base)")
    ap.add_argument("--devices", default=None,
                    help="comma list, e.g. R10_C01,R11_C02 (also accepts R10C1)")
    ap.add_argument("--max-devices", type=int, default=None)
    ap.add_argument("--no-device-plots", action="store_true")
    ap.add_argument("--no-spectra-plots", action="store_true",
                    help="skip the (slow) per-device spectra figures")
    ap.add_argument("--popt-source", choices=["20C", "match"], default=POPT_SOURCE)
    ap.add_argument("--tref", type=int, default=C2_T_REF)
    args = ap.parse_args(argv)

    base = args.base
    liv_dir = os.path.join(base, args.liv_subdir)
    spec_dir = os.path.join(base, args.spec_subdir)
    out_dir = args.out or os.path.join(base, OUT_SUBDIR)
    wafer_label = args.label or os.path.basename(os.path.normpath(base))
    plot_root = os.path.join(out_dir, "plots")
    os.makedirs(plot_root, exist_ok=True)

    for d, lab in ((liv_dir, "LIV"), (spec_dir, "Spectrometer")):
        if not os.path.isdir(d):
            sys.exit("ERROR: %s folder not found: %s" % (lab, d))

    print("Indexing ...")
    liv_idx = index_liv(liv_dir)
    spec_idx = index_spectra(spec_dir)
    print("  LIV         : %d devices, temperatures %s"
          % (len(liv_idx), sorted({t for v in liv_idx.values() for t in v})))
    print("  Spectrometer: %d devices, temperatures %s"
          % (len(spec_idx), sorted({t for v in spec_idx.values() for t in v})))

    devices = sorted(set(spec_idx) & set(liv_idx))
    missing_liv = sorted(set(spec_idx) - set(liv_idx))
    if missing_liv:
        print("  WARNING: %d spectrometer devices have no LIV data: %s"
              % (len(missing_liv), ", ".join(dev_key(*d) for d in missing_liv[:10])))

    if args.devices:
        want = set()
        for tok in args.devices.split(","):
            m = re.search(r"R(\d+)[_ ]*C(\d+)", tok.strip(), re.I)
            if m:
                want.add((int(m.group(1)), int(m.group(2))))
        devices = [d for d in devices if d in want]
    if args.max_devices:
        devices = devices[:args.max_devices]

    print("Analysing %d devices ...\n" % len(devices))

    dev_rows, c2_rows, lam_rows = [], [], []

    for n, (row, col) in enumerate(devices, 1):
        dev = dev_key(row, col)
        print("[%3d/%3d] %s" % (n, len(devices), dev), end="")

        # ---- LIV at every temperature -------------------------------
        liv_by_T, met_by_T = {}, {}
        for T, f in sorted(liv_idx[(row, col)].items()):
            df = load_liv(f)
            if df is not None:
                liv_by_T[T] = df
                met_by_T[T] = liv_metrics(df)
        if not liv_by_T:
            print("   -> no usable LIV, skipped")
            continue

        liv_ref_T = (LIV_FALLBACK_T if LIV_FALLBACK_T in liv_by_T
                     else min(liv_by_T))

        # ---- spectra at every temperature ---------------------------
        spec_by_T = spec_idx[(row, col)]
        run_by_T = {}
        for T, folder in sorted(spec_by_T.items()):
            if args.popt_source == "match" and T in liv_by_T:
                liv_use, liv_T = liv_by_T[T], T
            else:
                liv_use, liv_T = liv_by_T[liv_ref_T], liv_ref_T
            ith = met_by_T.get(liv_T, {}).get("Ith_A", np.nan)
            r = analyse_run(folder, liv_use, ith)
            if r is None:
                continue
            r["liv_T_used"] = liv_T
            r["Ith_A"] = ith
            run_by_T[T] = r

            c2_rows.append({
                "device": dev, "row": row, "col": col, "T_plate_C": T,
                "C2_nm_per_W": r["C2_nm_per_W"],
                "C2_nm_per_mW": r["C2_nm_per_W"] / 1e3
                                if np.isfinite(r["C2_nm_per_W"]) else np.nan,
                "C2_R2": r["C2_R2"], "n_fit_points": r["n_fit"],
                "Pdiss_lo_mW": r["Pdiss_lo_W"] * 1e3,
                "Pdiss_hi_mW": r["Pdiss_hi_W"] * 1e3,
                "Ith_mA": ith * 1e3 if np.isfinite(ith) else np.nan,
                "LIV_T_used_C": liv_T,
                "detector_saturated": r["saturated"],
                "spec_folder": os.path.basename(spec_by_T[T]),
            })

        valid = {T: r for T, r in run_by_T.items() if np.isfinite(r["C2_nm_per_W"])}
        c1 = fit_C1(valid)

        for T, lam in c1.get("lambda_at_ref", {}).items():
            lam_rows.append({"device": dev, "row": row, "col": col,
                             "T_plate_C": T,
                             "Pdiss_ref_mW": c1["Pdiss_ref_W"] * 1e3,
                             "lambda_at_Pref_nm": lam})

        C1 = c1.get("C1_nm_per_K", np.nan)
        Tref = args.tref if args.tref in valid else (min(valid) if valid else None)
        C2_ref_W = valid[Tref]["C2_nm_per_W"] if Tref is not None else np.nan
        C2_ref_mW = C2_ref_W / 1e3 if np.isfinite(C2_ref_W) else np.nan
        Rth_KW = C2_ref_W / C1 if (np.isfinite(C2_ref_W) and np.isfinite(C1) and C1 != 0) \
            else np.nan

        dev_rows.append({
            "device": dev, "row": row, "col": col,
            "n_temps_spectra": len(run_by_T),
            "n_temps_C2_ok": len(valid),
            "C1_nm_per_K": C1,
            "C1_R2": c1.get("C1_R2", np.nan),
            "C1_T_used_C": ";".join(str(t) for t in c1.get("C1_T_used", [])),
            "Pdiss_ref_mW": c1.get("Pdiss_ref_W", np.nan) * 1e3,
            "C2_Tref_C": Tref if Tref is not None else np.nan,
            "C2_at_Tref_nm_per_W": C2_ref_W,
            "C2_at_Tref_nm_per_mW": C2_ref_mW,
            "C2_at_Tref_R2": valid[Tref]["C2_R2"] if Tref is not None else np.nan,
            "Rth_K_per_W": Rth_KW,
            "Rth_K_per_mW": Rth_KW / 1e3 if np.isfinite(Rth_KW) else np.nan,
            "Ith_20C_mA": met_by_T.get(20, {}).get("Ith_A", np.nan) * 1e3,
            "SE_20C_W_per_A": met_by_T.get(20, {}).get("SE_W_per_A", np.nan),
            "Pmax_20C_mW": met_by_T.get(20, {}).get("Pmax_W", np.nan) * 1e3,
            "Imax_LIV_20C_mA": (liv_by_T[20]["Current_A"].max() * 1e3
                                if 20 in liv_by_T else np.nan),
        })

        # ---- per-device figures -------------------------------------
        if not args.no_device_plots:
            devdir = os.path.join(plot_root, dev)
            os.makedirs(devdir, exist_ok=True)
            plot_liv(dev, liv_by_T, met_by_T, devdir)
            plot_thermal_fits(dev, valid, c1, devdir)
            if not args.no_spectra_plots:
                plot_spectra(dev, spec_by_T, run_by_T, devdir)

        print("   C1=%s nm/K   C2@%s=%s nm/mW   Rth=%s K/W"
              % ("%.4f" % C1 if np.isfinite(C1) else "  n/a",
                 Tref if Tref is not None else "--",
                 "%.4f" % C2_ref_mW if np.isfinite(C2_ref_mW) else " n/a",
                 "%.1f" % Rth_KW if np.isfinite(Rth_KW) else " n/a"))

    # ---- write results ----------------------------------------------
    dev_df = pd.DataFrame(dev_rows)
    c2_df = pd.DataFrame(c2_rows)
    lam_df = pd.DataFrame(lam_rows)

    if len(dev_df):
        dev_df = dev_df.sort_values(["col", "row"])
        dev_df.to_csv(os.path.join(out_dir, "results_device.csv"), index=False)
    if len(c2_df):
        c2_df.to_csv(os.path.join(out_dir, "results_C2_by_T.csv"), index=False)
    if len(lam_df):
        lam_df.to_csv(os.path.join(out_dir, "results_lambda_vs_T.csv"), index=False)

    if len(dev_df):
        plot_summary(dev_df, c2_df, out_dir, wafer_label)

    # ---- console summary --------------------------------------------
    print("\n" + "=" * 72)
    print("Results written to: %s" % out_dir)
    if len(dev_df):
        ok = dev_df["Rth_K_per_W"].notna().sum()
        print("Devices analysed: %d   (Rth extracted for %d)" % (len(dev_df), ok))
        for col, lab, fmt in (("C1_nm_per_K", "C1        (nm/K)", "%.4f"),
                              ("C2_at_Tref_nm_per_mW",
                               "C2 @%2d C  (nm/mW)" % args.tref, "%.4f"),
                              ("Rth_K_per_W", "Rth       (K/W) ", "%.1f")):
            v = dev_df[col].dropna()
            if len(v):
                print(("  %s : median " + fmt + "   mean " + fmt +
                       "   std " + fmt + "   [n=%d]")
                      % (lab, v.median(), v.mean(), v.std(), len(v)))
    if len(c2_df):
        print("\nC2 (nm/mW) per plate temperature:")
        g = c2_df.groupby("T_plate_C")["C2_nm_per_mW"].agg(["count", "mean", "std", "median"])
        print(g.to_string(float_format=lambda x: "%.4f" % x))
    print("=" * 72)


if __name__ == "__main__":
    main()
