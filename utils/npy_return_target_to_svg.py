from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def _decode_ts_ns(v: object) -> pd.DatetimeIndex:
    arr = np.asarray(v)
    if arr.size == 0:
        return pd.DatetimeIndex([], tz="UTC")
    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.to_datetime(arr, utc=True)

    raw = arr.astype("int64")
    m = int(np.nanmax(np.abs(raw)))
    # Auto-detect epoch unit by magnitude to handle legacy mixed payloads.
    # ns: ~1e18, us: ~1e15, ms: ~1e12, s: ~1e9
    if m >= 10**17:
        unit = "ns"
    elif m >= 10**14:
        unit = "us"
    elif m >= 10**11:
        unit = "ms"
    else:
        unit = "s"
    return pd.to_datetime(raw, unit=unit, utc=True)


def _as_float(v: object) -> np.ndarray:
    return np.asarray(v, dtype=float)


def _collect_npy(path_like: str) -> list[Path]:
    p = Path(path_like)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.rglob("*.npy"))
    return []


def _parse_inset_size(size_text: str) -> tuple[str, str]:
    raw = str(size_text).strip().lower()
    for sep in ("x", ",", "*"):
        if sep in raw:
            a, b = raw.split(sep, 1)
            w = float(a.strip())
            h = float(b.strip())
            if w <= 0 or h <= 0:
                raise ValueError
            return f"{w:g}%", f"{h:g}%"
    raise ValueError(
        f"Invalid inset size '{size_text}'. Use format like 37x34 (percent)."
    )


def _fill_band_gradient_from_zero(
    ax: plt.Axes,
    x: pd.DatetimeIndex,
    band_lo: np.ndarray,
    band_hi: np.ndarray,
    label: str = "Pred Band",
    color: str = "lightblue",
    layers: int = 20,
    alpha_center: float = 0.8,
    alpha_edge: float = 0.2,
    draw_edges: bool = True,
    edge_linewidth: float = 1.0,
    edge_alpha: float = 0.85,
) -> None:
    """
    Draw uncertainty band with strongest opacity near y=0 and fading toward both edges.
    """
    base = to_rgba(color)
    layers = max(4, int(layers))

    for k in range(1, layers + 1):
        q_prev = (k - 1) / layers
        q_curr = k / layers
        # Linear alpha decay: darkest near 0, lightest near band edges.
        alpha = alpha_center - (alpha_center - alpha_edge) * (q_prev)
        alpha = float(max(0.0, min(1.0, alpha)))

        # Positive direction: from 0 to band_hi
        y1_up = band_hi * q_prev
        y2_up = band_hi * q_curr
        # Negative direction: from 0 to band_lo
        y1_dn = band_lo * q_prev
        y2_dn = band_lo * q_curr

        draw_label = label if k == 1 else None
        ax.fill_between(x, y1_up, y2_up, color=base, alpha=alpha, linewidth=0.0, label=draw_label)
        ax.fill_between(x, y1_dn, y2_dn, color=base, alpha=alpha, linewidth=0.0)

    if draw_edges:
        ax.plot(x, band_hi, color=base, linewidth=edge_linewidth, alpha=edge_alpha, label=None)
        ax.plot(x, band_lo, color=base, linewidth=edge_linewidth, alpha=edge_alpha, label=None)


def _add_inset(
    ax: plt.Axes,
    x: pd.DatetimeIndex,
    pred: np.ndarray,
    real: np.ndarray,
    band_lo: np.ndarray | None,
    band_hi: np.ndarray | None,
    inset_start_ratio: float,
    inset_end_ratio: float,
    inset_loc: str,
    inset_borderpad: float,
    inset_size: tuple[str, str],
) -> None:
    n = len(x)
    if n < 5:
        return

    lo_r = float(max(0.0, min(1.0, inset_start_ratio)))
    hi_r = float(max(0.0, min(1.0, inset_end_ratio)))
    if hi_r <= lo_r:
        return

    i0 = int(np.floor(lo_r * (n - 1)))
    i1 = int(np.ceil(hi_r * (n - 1)))
    if i1 - i0 < 2:
        return

    xz = x[i0 : i1 + 1]
    pz = pred[i0 : i1 + 1]
    rz = real[i0 : i1 + 1]
    bz_lo = band_lo[i0 : i1 + 1] if band_lo is not None else None
    bz_hi = band_hi[i0 : i1 + 1] if band_hi is not None else None

    y_min = float(np.nanmin(np.concatenate([pz, rz])))
    y_max = float(np.nanmax(np.concatenate([pz, rz])))
    if bz_lo is not None and bz_hi is not None:
        y_min = float(min(y_min, np.nanmin(bz_lo)))
        y_max = float(max(y_max, np.nanmax(bz_hi)))
    pad = (y_max - y_min) * 0.1 if y_max > y_min else 1e-4

    axins = inset_axes(
        ax,
        width=inset_size[0],
        height=inset_size[1],
        loc=inset_loc,
        borderpad=float(inset_borderpad),
    )
    axins.plot(xz, pz, linewidth=1.6, alpha=0.9, color="tab:blue")
    axins.plot(xz, rz, linewidth=1.3, alpha=0.85, color="tab:orange")
    if bz_lo is not None and bz_hi is not None:
        _fill_band_gradient_from_zero(axins, xz, bz_lo, bz_hi, label="Pred Band", color="lightblue")
    axins.set_xlim(xz[0], xz[-1])
    axins.set_ylim(y_min - pad, y_max + pad)
    axins.grid(True, alpha=0.35)
    axins.set_xticks([])
    axins.set_yticks([])

    ax.indicate_inset_zoom(axins, edgecolor="0.25", alpha=0.9)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.25", ls="--", lw=1.1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render return_target_layer .npy into SVG in the same folder."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="reports/exp_results_meta",
        help="A .npy file or a directory (recursive search).",
    )
    parser.add_argument(
        "--with-inset",
        action="store_true",
        help="If set, add a zoom-in inset on the top time-series panel.",
    )
    parser.add_argument(
        "--with-scatter",
        action="store_true",
        help="If set, include the 45-degree scatter subplot. Default is off.",
    )
    parser.add_argument(
        "--inset-start-ratio",
        type=float,
        default=0.6,
        help="Zoom window start as ratio in [0,1], default 0.6.",
    )
    parser.add_argument(
        "--inset-end-ratio",
        type=float,
        default=0.9,
        help="Zoom window end as ratio in [0,1], default 0.9.",
    )
    parser.add_argument(
        "--main-y-pad-ratio",
        type=float,
        default=0.42,
        help="Legacy symmetric y padding ratio for main plot when --with-inset is on. Default 0.42.",
    )
    parser.add_argument(
        "--main-y-pad-ratio-upper",
        type=float,
        default=None,
        help="Upper-side y padding ratio for main plot when --with-inset is on.",
    )
    parser.add_argument(
        "--main-y-pad-ratio-lower",
        type=float,
        default=None,
        help="Lower-side y padding ratio for main plot when --with-inset is on.",
    )
    parser.add_argument(
        "--inset-loc",
        type=str,
        default="upper right",
        choices=[
            "upper left",
            "upper center",
            "upper right",
            "center left",
            "center",
            "center right",
            "lower left",
            "lower center",
            "lower right",
        ],
        help="Inset location on the main axis.",
    )
    parser.add_argument(
        "--inset-borderpad",
        type=float,
        default=1.2,
        help="Inset border padding. Larger value pushes inset away from borders.",
    )
    parser.add_argument(
        "--inset1-size",
        type=str,
        default="37x34",
        help='First inset size in percent, format "WxH" (e.g. 37x34).',
    )
    parser.add_argument(
        "--inset2-start-ratio",
        type=float,
        default=None,
        help="Optional second inset start ratio in [0,1].",
    )
    parser.add_argument(
        "--inset2-end-ratio",
        type=float,
        default=None,
        help="Optional second inset end ratio in [0,1].",
    )
    parser.add_argument(
        "--inset2-loc",
        type=str,
        default="lower left",
        choices=[
            "upper left",
            "upper center",
            "upper right",
            "center left",
            "center",
            "center right",
            "lower left",
            "lower center",
            "lower right",
        ],
        help="Second inset location on the main axis.",
    )
    parser.add_argument(
        "--inset2-borderpad",
        type=float,
        default=1.2,
        help="Second inset border padding.",
    )
    parser.add_argument(
        "--inset2-size",
        type=str,
        default="37x34",
        help='Second inset size in percent, format "WxH" (e.g. 32x28).',
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title override. If set, replace title loaded from npy.",
    )
    args = parser.parse_args()

    npy_files = _collect_npy(args.input)
    if not npy_files:
        print(f"[WARN] no input files found: {args.input}")
        return

    converted = 0
    for npy_path in npy_files:
        out = convert_one_with_options(
            npy_path=npy_path,
            with_inset=bool(args.with_inset),
            with_scatter=bool(args.with_scatter),
            inset_start_ratio=float(args.inset_start_ratio),
            inset_end_ratio=float(args.inset_end_ratio),
            main_y_pad_ratio=float(args.main_y_pad_ratio),
            main_y_pad_ratio_upper=float(args.main_y_pad_ratio_upper) if args.main_y_pad_ratio_upper is not None else None,
            main_y_pad_ratio_lower=float(args.main_y_pad_ratio_lower) if args.main_y_pad_ratio_lower is not None else None,
            inset_loc=str(args.inset_loc),
            inset_borderpad=float(args.inset_borderpad),
            inset2_start_ratio=float(args.inset2_start_ratio) if args.inset2_start_ratio is not None else None,
            inset2_end_ratio=float(args.inset2_end_ratio) if args.inset2_end_ratio is not None else None,
            inset2_loc=str(args.inset2_loc),
            inset2_borderpad=float(args.inset2_borderpad),
            inset1_size=str(args.inset1_size),
            inset2_size=str(args.inset2_size),
            title_override=str(args.title) if args.title is not None else None,
        )
        if out is not None:
            converted += 1
            print(f"[OK] {npy_path} -> {out}")
    print(f"[DONE] converted {converted} return_target_layer file(s).")


def convert_one_with_options(
    npy_path: Path,
    with_inset: bool,
    with_scatter: bool,
    inset_start_ratio: float,
    inset_end_ratio: float,
    main_y_pad_ratio: float,
    main_y_pad_ratio_upper: float | None,
    main_y_pad_ratio_lower: float | None,
    inset_loc: str,
    inset_borderpad: float,
    inset2_start_ratio: float | None,
    inset2_end_ratio: float | None,
    inset2_loc: str,
    inset2_borderpad: float,
    inset1_size: str,
    inset2_size: str,
    title_override: str | None,
) -> Path | None:
    meta = np.load(str(npy_path), allow_pickle=True).item()
    if str(meta.get("kind", "")) != "return_target_layer":
        return None

    x = _decode_ts_ns(meta.get("target_ts_ns", []))
    pred = _as_float(meta.get("pred_next_return", []))
    real = _as_float(meta.get("real_next_return", []))
    band_lo = _as_float(meta.get("band_lo", []))
    band_hi = _as_float(meta.get("band_hi", []))
    title = str(title_override) if title_override is not None else str(meta.get("title", npy_path.stem))

    n = min(len(x), len(pred), len(real))
    if n == 0:
        return None
    x = x[:n]
    pred = pred[:n]
    real = real[:n]

    has_band = len(band_lo) >= n and len(band_hi) >= n
    if has_band:
        band_lo = band_lo[:n]
        band_hi = band_hi[:n]

    inset1_size_parsed = _parse_inset_size(inset1_size)
    inset2_size_parsed = _parse_inset_size(inset2_size)

    if with_scatter:
        fig = plt.figure(figsize=(15, 9))
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
    else:
        fig = plt.figure(figsize=(15, 5.5))
        ax1 = plt.subplot(1, 1, 1)
        ax2 = None

    ax1.plot(x, pred, label="Pred Next Return", linewidth=1.8, alpha=0.9, color="tab:blue")
    ax1.plot(x, real, label="Real Next Return (GT)", linewidth=1.4, alpha=0.85, color="tab:orange")
    if has_band:
        _fill_band_gradient_from_zero(ax1, x, band_lo, band_hi, label="Pred Band q10~q90", color="lightblue")
    if with_inset:
        # Keep extra vertical room so the inset doesn't visually cover the main signal too much.
        y_all = [pred, real]
        if has_band:
            y_all.extend([band_lo, band_hi])
        y_concat = np.concatenate(y_all)
        y_min = float(np.nanmin(y_concat))
        y_max = float(np.nanmax(y_concat))
        span = y_max - y_min
        symmetric_ratio = max(0.0, float(main_y_pad_ratio))
        upper_ratio = symmetric_ratio if main_y_pad_ratio_upper is None else max(0.0, float(main_y_pad_ratio_upper))
        lower_ratio = symmetric_ratio if main_y_pad_ratio_lower is None else max(0.0, float(main_y_pad_ratio_lower))
        if span > 0:
            pad_upper = span * upper_ratio
            pad_lower = span * lower_ratio
        else:
            pad_upper = 1e-4
            pad_lower = 1e-4
        ax1.set_ylim(y_min - pad_lower, y_max + pad_upper)
        _add_inset(
            ax=ax1,
            x=x,
            pred=pred,
            real=real,
            band_lo=band_lo if has_band else None,
            band_hi=band_hi if has_band else None,
            inset_start_ratio=inset_start_ratio,
            inset_end_ratio=inset_end_ratio,
            inset_loc=inset_loc,
            inset_borderpad=inset_borderpad,
            inset_size=inset1_size_parsed,
        )
        if inset2_start_ratio is not None and inset2_end_ratio is not None:
            _add_inset(
                ax=ax1,
                x=x,
                pred=pred,
                real=real,
                band_lo=band_lo if has_band else None,
                band_hi=band_hi if has_band else None,
                inset_start_ratio=float(inset2_start_ratio),
                inset_end_ratio=float(inset2_end_ratio),
                inset_loc=inset2_loc,
                inset_borderpad=inset2_borderpad,
                inset_size=inset2_size_parsed,
            )
    ax1.set_title(title)
    ax1.grid(True)
    ax1.legend()

    if with_scatter and ax2 is not None:
        ax2.scatter(pred, real, s=14, alpha=0.55, label="points")
        lo = float(min(np.nanmin(pred), np.nanmin(real)))
        hi = float(max(np.nanmax(pred), np.nanmax(real)))
        ax2.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.4, label="45-degree line")
        ax2.set_xlabel("Predicted next return")
        ax2.set_ylabel("Real next return")
        ax2.grid(True)
        ax2.legend()

    suffix = []
    if with_inset:
        suffix.append("with_inset")
    if with_inset and inset2_start_ratio is not None and inset2_end_ratio is not None:
        suffix.append("with_dual_inset")
    if with_scatter:
        suffix.append("with_scatter")
    suffix_text = ("__" + "__".join(suffix)) if suffix else ""
    out_name = npy_path.stem + suffix_text + ".svg"
    out_path = npy_path.with_name(out_name)
    if with_inset:
        plt.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.07, hspace=0.30)
    else:
        plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    main()
