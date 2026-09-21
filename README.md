# Hyperspectral AOD / WVOD Retrieval Software

**Version v1.0.0** · Authors: Liu Liying, Zheng Feng · Contact: feng1214@126.com · License: MIT License

Retrieves aerosol optical depth (AOD), Ångström parameters, 936 nm water vapor optical depth (WVOD), and precipitable water vapor (PWV, in both vertical- and slant-column forms) from hyperspectral solar direct normal irradiance (DNI) observations.

Algorithms follow **QX/T 69-2024 "Aerosol Optical Depth — Sun Photometer Method"** (Eqs. 1–14) and *Theoretical Basis of Aerosol Optical Depth and Ångström Parameter Retrieval*. See `docs/` and the in-app help page for details.

## Key Features

- **Data import**: two Excel file types — svd (spectral direct irradiance) and svg (global horizontal irradiance); spectral range auto-detected from column count (801 → 300–1100 nm, 121 → 280–400 nm, 751 → 950–1700 nm); manual exclusion of arbitrary time periods (excluded data are set to NaN and skipped in calibration and retrieval)
- **Langley calibration**: vectorized regression over ~800 wavelengths (<1 s); Bouguer–Langley cloud/disturbance screening; unified daily fitting window (auto-selected morning/afternoon or user-specified); tiered acceptance criteria — physical constraint 0<τ≤1 as the first veto, R²≥0.9 for normal solutions, σ(lnE₀)≤0.01 with R²≥0.5 for estimated solutions (marked `[estimated]` in plot titles/reports); wavelengths that fail calibration fall back to the Wehrli (1985) standard spectrum
- **AOD retrieval**: time series at 11 characteristic wavelengths (340/380/400/440/500/550/675/780/870/936/1020 nm) plus full-spectrum AOD (total optical depth minus Rayleigh scattering)
- **Water vapor retrieval**: 936 nm baseline method (QX/T Eqs. 8–10) for WVOD; PWV (mm, adjustable a/b coefficients) converted via the slant-path transmittance formulas (QX/T Eqs. 5–7); vertical and slant (= vertical × m) columns output simultaneously
- **svg global irradiance integration**: full spectrum / UV / VIS / NIR / PAR / PPFD / illuminance (CIE 1931 V(λ))
- **Visualization**: linear/log y-axis toggle and per-line show/hide on both time-series and wavelength-distribution plots; Langley fit diagnostics; raw-spectrum and irradiance time-series viewers; dedicated subpages for vertical/slant PWV
- **Three computation modes**: Fast (vectorized) / Slow (minute-by-minute) / Both (same-plot comparison)
- Multiple algorithm options: air mass (4 formulations including Kasten), Rayleigh formulas, gas-absorption toggles, water-vapor baseline methods; parameterized latitude/longitude/altitude (automatic pressure conversion)/ozone column; parameter profile import/export; export of result figures and Excel reports

## Run

```bat
pip install -r requirements.txt
start_all.bat
```

Open http://localhost:8000 in a browser and upload a single-day svd/svg Excel data file.

## File Overview

| File | Description |
|---|---|
| `aod_inversion.py` | Core algorithm library (solar geometry, optical-depth retrieval, Langley calibration, WVOD/PWV) |
| `process_dni_data.py` | Data reading and batch processing (slow minute-by-minute / fast vectorized) |
| `main.py` | FastAPI backend (async tasks + progress queries + Excel export + version endpoint) |
| `frontend_english.html` | Web frontend (ECharts) |
| `help.html` | Retrieval principles and algorithm description page |
| `recompute_all.py` | Batch recomputation script for multi-day data |
| `start_all.bat` | One-click startup (kills any stale process on port 8000, then restarts) |
| `docs/软件使用说明书.md` | User manual — installation, UI parameters, workflow, result interpretation, FAQ (Chinese) |
| `docs/算法原理说明.md` | Retrieval principles and algorithms, review/archive edition, consistent with help.html (Chinese) |
| `CHANGELOG.md` | Version history |

## Known Limitations

- The ozone retrieval module is currently unusable (to be rebuilt; UV differential cross-sections and calibration need refinement)
- The ozone Chappuis band (450–750 nm) is not corrected, so 675 nm AOD is slightly overestimated
- Spectral data inside strong water-vapor/oxygen absorption bands (~690–730, 760, 790–870, 920–980 nm) cannot be used quantitatively
- Points with solar zenith angle θ ≥ 85° are excluded from retrieval; latitude must be accurate to ~0.1°
- Older data batches show ±1-pixel CCD timing jitter in the 870 nm channel (fixed in the new FPGA firmware); calibration for these data follows the estimated-solution criteria and the data are kept uncorrected
