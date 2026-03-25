# NPY Metadata to SVG Plotting Guide

This document complements [EXP_RESULTS_META.md](./EXP_RESULTS_META.md).  
It explains how to convert `.npy` metadata under `reports/exp_results_meta` into `.svg` figures for post-editing (for example, in Inkscape).

## 1. Relationship to `EXP_RESULTS_META.md`

- `EXP_RESULTS_META.md` defines the `.npy` protocol (`kind`, field names, timestamp fields, etc.).
- This file explains how to consume that protocol and render plots.
- The current focus is the `return_target_layer` renderer:
  - [utils/npy_return_target_to_svg.py](./utils/npy_return_target_to_svg.py)

## 2. Quick Start (Focus: return_target)

Single file, default output (main time-series only):

```powershell
python utils\npy_return_target_to_svg.py reports\exp_results_meta\ft\1d\zeroshot\btcusdt\exp1_whitebox_return_target_test_BTCUSDT_1d.npy
```

Batch mode (recursive on a directory):

```powershell
python utils\npy_return_target_to_svg.py reports\exp_results_meta\ft\1d\zeroshot\btcusdt
```

Output behavior:

- SVG files are saved in the same folder as the source `.npy`.
- Filename suffixes are added based on options (for example `__with_inset`, `__with_dual_inset`, `__with_scatter`).

## 3. Field Mapping for `return_target_layer`

The script reads payloads with `kind == "return_target_layer"`:

- `target_ts_ns` -> x-axis timestamps
- `pred_next_return` -> predicted next-return line
- `real_next_return` -> realized next-return line
- `band_lo` / `band_hi` -> uncertainty band

Notes:

- The uncertainty band is rendered as a gradient: darkest near `y=0`, lighter toward upper/lower band edges.
- Band boundary lines (`band_lo`, `band_hi`) are drawn on top of the gradient.
- Timestamp decoding auto-detects `s/ms/us/ns`, so legacy files with mixed units are supported.

## 4. Common Options (`npy_return_target_to_svg.py`)

- `--with-scatter`
  - Show the second subplot (pred vs real scatter with 45-degree line). Default: off.
- `--with-inset`
  - Enable inset zoom panel(s) in the main time-series panel. Default: off.
- `--inset-start-ratio`, `--inset-end-ratio`
  - Window for inset #1, by sample ratio in `[0, 1]`.
- `--inset2-start-ratio`, `--inset2-end-ratio`
  - Optional window for inset #2.
- `--inset-loc`, `--inset2-loc`
  - Inset positions (for example `upper left`, `upper center`, `upper right`, `lower right`, etc.).
- `--inset-borderpad`, `--inset2-borderpad`
  - Inset border paddings.
- `--inset1-size`, `--inset2-size`
  - Inset sizes, format `WxH` in percent (for example `37x34`, `32x26`).
- `--main-y-pad-ratio-upper`, `--main-y-pad-ratio-lower`
  - Independent upper/lower y-axis padding ratios for the main panel (useful to avoid inset/legend overlap).
- `--main-y-pad-ratio`
  - Legacy symmetric y-padding ratio. If upper/lower are provided, upper/lower take priority.
- `--title`
  - Override the title loaded from `.npy`.

## 5. Dual-Inset Example

Inset #1: ratio `0.3~0.5`, placed at upper-left.  
Inset #2: ratio `0.7~0.9`, placed at lower-right.

```powershell
python utils\npy_return_target_to_svg.py reports\exp_results_meta\ft\1d\zeroshot\btcusdt\exp1_whitebox_return_target_test_BTCUSDT_1d.npy `
  --with-inset `
  --inset-start-ratio 0.3 --inset-end-ratio 0.5 --inset-loc "upper left" --inset1-size 34x28 `
  --inset2-start-ratio 0.7 --inset2-end-ratio 0.9 --inset2-loc "lower right" --inset2-size 30x24 `
  --main-y-pad-ratio-upper 0.55 --main-y-pad-ratio-lower 0.20 `
  --title "BTCUSDT 1d | White-box Return Target (Test)"
```

## 6. Other `.npy -> .svg` Scripts (Brief)

These two scripts are available and currently kept simple/stable:

- [utils/npy_forecast_layer_to_svg.py](./utils/npy_forecast_layer_to_svg.py)
  - Handles `kind == "forecast_layer"`.
- [utils/npy_backtest_layer_to_svg.py](./utils/npy_backtest_layer_to_svg.py)
  - Handles `kind == "backtest_layer"`.

They can be further refined later to align with your final paper-style standards.

