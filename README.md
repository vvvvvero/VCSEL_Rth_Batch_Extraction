# VCSEL Rth Batch Analysis

Batch-extracts VCSEL thermal resistance (R<sub>th</sub>) from paired
LIV-vs-temperature and spectrometer-vs-temperature wafer test data, across
every device on a wafer.

For each device it produces:

- L–I–V curves at every plate temperature
- Relative-intensity (dB) spectra vs wavelength, at each temperature, for a
  range of currents
- The **C1**/**C2**/**R<sub>th</sub>** thermal fit (method below)
- Wafer-level summary plots (C2 vs temperature, histograms, wafer maps)

## Quickstart

```bash
pip install -r requirements.txt
python rth_batch_analysis.py --base sample_data --label "demo wafer"
```

This runs on the small sample dataset included in this repo (`sample_data/`,
2 devices, 7 temperatures, trimmed spectra — see [Sample data](#sample-data))
and writes results to `sample_data/RTH_Analysis/`.

For a real wafer:

```bash
python rth_batch_analysis.py --base "D:\path\to\WAFER_ID"
```

Useful options:

```bash
python rth_batch_analysis.py --base <dir> --devices R10_C01,R11_C02   # subset
python rth_batch_analysis.py --base <dir> --no-spectra-plots          # much faster
python rth_batch_analysis.py --base <dir> --popt-source match         # P_opt from LIV at the same T
python rth_batch_analysis.py --base <dir> --max-devices 5             # smoke test
python rth_batch_analysis.py --base <dir> --liv-subdir XYZ --spec-subdir ABC  # non-default folder names
python rth_batch_analysis.py --base <dir> --label "5505_B"            # title/legend label
```

Run `python rth_batch_analysis.py --help` for the full list.

## Expected data layout

The script auto-detects device (row, column) and plate temperature from
folder/file names — no wafer ID or lot number is hardcoded anywhere, so any
run that follows this layout under `--base` works:

```
<base>\
  B_LIV_T\
    B_LIV_{T}Deg\
      <anything>_row_{RR}_col_{CC}_site_{SSS}_<date>_<time>.csv
        columns: ...,Voltage_V,Current_A,Optical_Power_W,...

  Spectrometer_T\
    <anything>_Col{c}_Row{r}_{pulse}ms_{T}Deg_sync_<date>_<time>\
      summary.csv
        columns: Point,...,Voltage_V,Current_A,Peak_nm,Peak_Intensity,Avg_Intensity,...
      spectra\
        spectrum_point_0000.csv, spectrum_point_0001.csv, ...
          columns: Relative_Time_s,Wavelength_nm,Intensity
      timed_bias\   (optional, not used by the analysis)
```

`{T}` is the plate temperature in °C, `{RR}`/`{CC}` are zero-padded row/column,
`{pulse}` is a pulse width tag (e.g. `0.01ms`). If your equipment writes
different subfolder names, point `--liv-subdir` / `--spec-subdir` at them —
everything below that level is parsed by regex, not fixed strings.

## Method

### 1. Dissipated power

```
Pdiss = I * V - P_opt(I)
```

`I`, `V` come from the spectrometer run's `summary.csv`. `P_opt(I)` is
linearly interpolated from the LIV sweep. By default the **20 °C** LIV is
used for every temperature (`--popt-source 20C`, the default) so one
consistent `P_opt(I)` enters all temperatures; `--popt-source match` uses the
LIV taken at the same plate temperature instead. Spectrometer points beyond
the LIV's measured current range are dropped (no extrapolated `P_opt`).

### 2. Lasing window

A spectrum point enters the fit only if:

| gate | default |
|---|---|
| `Peak_Intensity / Avg_Intensity` > `SNR_MIN` | 3.0 |
| `I` > `ITH_MARGIN * Ith` | 1.20 |
| `Pdiss` > 0 and `I` within the LIV's measured range | — |
| at least `MIN_FIT_POINTS` surviving points | 6 |

`Ith` comes from max-slope linear extrapolation of the L–I curve: the fit
window is the contiguous region where `dP/dI >= 0.7 * max(dP/dI)` below
rollover, extrapolated to `P = 0`. (Fitting the top 50% of the ramp instead
lands in the sub-linear rollover region and can return a negative threshold —
this failed on the majority of devices in initial testing.)

### 3. C2 — `dλ/dPdiss` at fixed plate temperature

Linear regression of `Peak_nm` vs `Pdiss` over the lasing window, one fit per
(device, temperature). Reported as **nm/W** and **nm/mW**, with `R²` and
point count.

### 4. C1 — `dλ/dT_plate` at fixed dissipated power

1. Each temperature's C2 fit covers a `Pdiss` window `[lo_T, hi_T]`.
2. Pick the largest set of temperatures whose windows overlap
   (≥ `MIN_T_FOR_C1` = 3 by default), then
   `Pdiss_ref = lo + PREF_FRAC * (hi - lo)` inside that common overlap.
3. Evaluate each temperature's C2 regression line at `Pdiss_ref` → `λ(T)`.
   (Using the fitted line rather than a raw interpolated point suppresses
   mode-hop noise.)
4. Linear regression `λ(T)` vs `T_plate` → **C1 (nm/K)**.

### 5. Thermal resistance

```
Rth = C2 / C1          [K/W]     (C2 in nm/W)
    = C2 / C1          [K/mW]    (C2 in nm/mW)
```

The headline `Rth` uses C2 at `--tref` (default 20 °C). `results_C2_by_T.csv`
lets you form `Rth(T)` at any temperature; panel (c) of the per-device fit
figure plots it.

## Outputs — `<out>\` (default `<base>\RTH_Analysis\`)

| file | contents |
|---|---|
| `results_device.csv` | one row per device: `C1_nm_per_K`, `C1_R2`, `Pdiss_ref_mW`, `C2_at_Tref_nm_per_mW`, `Rth_K_per_W`, `Rth_K_per_mW`, `Ith_20C_mA`, `SE_20C_W_per_A`, `Pmax_20C_mW` |
| `results_C2_by_T.csv` | one row per device+temperature: `C2_nm_per_mW`, `C2_R2`, `n_fit_points`, `Pdiss_lo/hi_mW`, `LIV_T_used_C`, `detector_saturated`, source folder |
| `results_lambda_vs_T.csv` | the `λ(T)` points at `Pdiss_ref` that the C1 fit used |
| `summary_C2_vs_T.png` | C2 vs plate T — every device (grey) + population mean ± 1σ |
| `summary_C2_boxplot.png` | C2 distribution per temperature |
| `summary_histograms.png` | C1 / C2@T<sub>ref</sub> / Rth histograms |
| `summary_wafermaps.png` | row×col wafer maps of Rth, C1, C2 |
| `plots\<device>\*_LIV_vs_T.png` | L–I (left axis) + V–I (right axis) for all temperatures, Ith marked |
| `plots\<device>\*_spectra_vs_T.png` | relative intensity (dB) vs wavelength, one panel per T, curves coloured by current |
| `plots\<device>\*_thermal_fits.png` | (a) λ vs Pdiss + C2 fits, (b) λ@Pdiss_ref vs T + C1 fit, (c) C2 and Rth vs T |

## Sample data

`sample_data/` contains a trimmed extract for 2 devices (all 7 plate
temperatures) from a real wafer run, enough to exercise the full pipeline —
LIV parsing, lasing-window detection, C1/C2/Rth fitting, and every plot type.
Spectra files are subsampled (10 of ~56 points per run, biased toward the
higher-current/lasing region); this does **not** affect the numeric C1/C2/Rth
results, since those are computed entirely from `summary.csv`, not the
per-point spectrum files — the subsampling only reduces how many traces the
spectra plot can show.

## Caveats

- **CW LIV, quasi-CW spectrometer bias.** `P_opt` is taken from the DC LIV
  sweep while the spectrometer holds each bias step briefly (~0.1–0.2 s).
  Self-heating conditions are close but not identical; this is the dominant
  systematic on `Pdiss`.
- **Peak wavelength = strongest mode.** Taken from `summary.csv`'s `Peak_nm`.
  Multi-mode devices hop modes, which shows up as steps in the λ-vs-Pdiss
  plot — the linear fit averages over them.
- `Voltage_V` in the spectrometer summary is the source setpoint, not
  necessarily a 4-wire measurement, so any probe/series drop is included in
  `I*V`.
- `detector_saturated` flags runs where `Peak_Intensity` exceeds
  `SAT_COUNTS`; the peak position of a clipped spectrum is unreliable.

## Tuning knobs

All at the top of `rth_batch_analysis.py`: `PULSE_FILTER`, `EXCLUDE_TOKENS`,
`POPT_SOURCE`, `ITH_MARGIN`, `SNR_MIN`, `MIN_FIT_POINTS`, `SAT_COUNTS`,
`I_MAX_MARGIN`, `MIN_T_FOR_C1`, `PREF_FRAC`, `C2_T_REF`, `N_SPEC_CURVES`,
`DB_FLOOR`, `DB_NORM` (`"per_panel"` shows intensity growth with current,
`"per_curve"` normalises each spectrum to its own peak), `DPI`.

## Requirements

Python 3.9+, `numpy`, `pandas`, `matplotlib` (see `requirements.txt`).
