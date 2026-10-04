"""Visualize VaR backtesting results against the Halbleib & Pohlmeier (2012) benchmarks.

Produces ``assets/model_validity_frontier.png``, a two-panel figure:

  Panel A -- Model Validity Frontier.  Each model placed at its
    (Kupiec POF p-value, Christoffersen independence p-value) coordinate, with
    the regulatory p = 0.05 thresholds drawn so the acceptance quadrant
    (top-right) separates visually from the rejection zone.  Two evaluation
    windows are shown: the full out-of-sample period and the paper's crash
    period.

  Panel B -- Crash-period violation rates, ours vs the paper's published values.

A NOTE ON WHAT CAN HONESTLY BE PLOTTED
--------------------------------------
The paper does **not** publish (Kupiec p-value, Christoffersen independence
p-value) pairs anywhere, so the paper's models cannot be placed on Panel A's
axes.  What its tables report is:

  * the **percentage rate of violations** (numeric), and
  * a single **conditional coverage** p-value given only as a band --
    "** refers to p-values of conditional coverage test smaller than 0.05,
    * to p-values between 0.05 and 0.10 and no mark refers to p-values larger
    than 0.10" (Tables A3, A5, 3), and
  * the Basel zone, encoded as bold / italic / plain typeface.

Inventing two numeric coordinates per model to place the paper on Panel A would
mean publishing fabricated figures attributed to named authors.  Panel B
therefore carries the paper comparison using the quantity the paper actually
reports numerically, with its significance band annotated.  Every hardcoded
benchmark below cites its source table.

Our own numbers are computed from the cached VaR panel, never hardcoded, so the
figure cannot drift out of step with the engine.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from backtesting import backtest_summary, build_panel
    from data_loader import DATE_COLUMN
    from var_models import ACTUAL_COLUMN, ALPHA, VAR_COLUMN_PREFIX
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from backtesting import backtest_summary, build_panel
    from data_loader import DATE_COLUMN
    from var_models import ACTUAL_COLUMN, ALPHA, VAR_COLUMN_PREFIX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_PATH = ASSETS_DIR / "model_validity_frontier.png"

# Paper's crash period (Section 3.1): Lehman's month through the trough.
CRASH_START = pd.Timestamp("2008-09-01")
CRASH_END = pd.Timestamp("2009-07-01")

REJECTION_LEVEL = 0.05
STANDALONE = ("HS", "GARCH-N", "GARCH-t")
COMBINATIONS = ("CQOM", "CCOM")

# --- Palette (dataviz reference instance; validated all-pairs, light mode) -----
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8982"
GRID = "#e4e3dd"
C_STANDALONE = "#2a78d6"   # categorical slot 1
C_COMBINATION = "#eb6834"  # categorical slot 2
C_CRITICAL = "#d03b3b"     # status: regulatory threshold
C_GOOD = "#0ca30c"         # status: acceptance region


@dataclass(frozen=True)
class PaperBenchmark:
    """A published figure from the paper, with its source recorded."""

    model: str
    violation_rate: float   # percent
    cc_band: str            # '**' p<0.05 | '*' 0.05-0.10 | '' p>0.10
    source: str


# Large-cap index is the closest analogue to the S&P 500 we model.
# Crash period = 1 Sep 2008 - 1 Jul 2009 (217 days in the paper).
PAPER_CRASH: tuple[PaperBenchmark, ...] = (
    PaperBenchmark("HS", 8.75, "**", "Table A5, Large cap, HS 250-day window"),
    PaperBenchmark("GARCH-N", 4.60, "**", "Table A3, Large cap, ND / ARMA-GARCH, 1987"),
    PaperBenchmark("GARCH-t", 1.38, "", "Table A3, Large cap, TD / ARMA-GARCH, 1987"),
    PaperBenchmark("CQOM", 1.20, "", "Table 3, Part A, ARMA-GARCH, Large cap, 1987"),
    PaperBenchmark("CCOM", 0.80, "", "Table 3, Part A, ARMA-GARCH, Large cap, 1987"),
)

BAND_LABEL = {
    "**": "CC p < 0.05 (rejected)",
    "*": "CC p in [0.05, 0.10)",
    "": "CC p > 0.10",
}


def family(model: str) -> str:
    return "Combination" if model in COMBINATIONS else "Stand-alone"


def family_color(model: str) -> str:
    return C_COMBINATION if model in COMBINATIONS else C_STANDALONE


# --- Results assembly ----------------------------------------------------------
def compute_results(alpha: float = ALPHA) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Backtest the full sample and the crash period.

    Combination weights are re-calibrated daily on a rolling window inside
    ``build_panel``, so every series here -- stand-alone and combined alike -- is
    genuinely out-of-sample.  The crash table is simply the crash-window slice of
    that same panel; no separate static pre-crash fit is needed (or wanted), as
    the rolling weights already adapt through the regime shift.
    """
    panel, _ = build_panel(verbose=False)
    dates = pd.to_datetime(panel[DATE_COLUMN])

    full = backtest_summary(panel, alpha=alpha)

    mask = (dates >= CRASH_START) & (dates <= CRASH_END)
    crash_panel = panel[mask].reset_index(drop=True)
    crash_table = backtest_summary(crash_panel, alpha=alpha, basel_window=250)

    weights = panel.attrs.get("weights")
    meta = {
        "full_n": len(panel),
        "full_start": dates.iloc[0].date(),
        "full_end": dates.iloc[-1].date(),
        "crash_n": len(crash_panel),
        "weights": weights,
    }
    return full, crash_table, meta


# --- Panel A -------------------------------------------------------------------
def _merge_coincident(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Join labels of points that land on the same coordinate.

    GARCH-N and GARCH-t produce an identical crash-period hit sequence, so their
    markers coincide exactly; without merging, one label would sit invisibly on
    top of the other.
    """
    merged: dict[tuple[float, float], list[str]] = {}
    for x, y, label in points:
        key = (round(x, 10), round(y, 10))
        merged.setdefault(key, []).append(label)
    return [(x, y, " / ".join(labels)) for (x, y), labels in merged.items()]


def plot_frontier(ax, full: pd.DataFrame, crash: pd.DataFrame, meta: dict) -> None:
    x_floor, x_max, y_max = 1e-11, 2.0, 1.06

    # Acceptance quadrant and rejection band, kept recessive.
    ax.add_patch(
        plt.Rectangle(
            (REJECTION_LEVEL, REJECTION_LEVEL), x_max - REJECTION_LEVEL, y_max - REJECTION_LEVEL,
            facecolor=C_GOOD, alpha=0.055, edgecolor="none", zorder=0.5,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (x_floor, 0.0), REJECTION_LEVEL - x_floor, y_max,
            facecolor=C_CRITICAL, alpha=0.05, edgecolor="none", zorder=0.5,
        )
    )

    ax.axvline(REJECTION_LEVEL, color=C_CRITICAL, ls="--", lw=1.6, zorder=2)
    ax.axhline(REJECTION_LEVEL, color=C_CRITICAL, ls="--", lw=1.6, zorder=2)

    specs = [
        (full, "o", 125, "Full sample"),
        (crash, "^", 135, "Crash period"),
    ]
    for table, marker, size, period in specs:
        for fam, color in (("Stand-alone", C_STANDALONE), ("Combination", C_COMBINATION)):
            sub = table[table["model"].map(family) == fam]
            if sub.empty:
                continue
            ax.scatter(
                sub["p_uc"].clip(lower=x_floor),
                sub["p_ind"],
                marker=marker,
                s=size,
                facecolor=color,
                edgecolor=SURFACE,       # 2px surface ring on overlapping marks
                linewidth=1.8,
                zorder=5,
                label=f"{fam} — {period}",
            )

    # Direct labels so identity is never carried by color alone.  Full-sample
    # labels sit below their marker and crash labels above, which separates the
    # two CQOM points; labels flip to the left of markers near the right edge so
    # they stay inside the axes.
    for table, dy in ((full, -12), (crash, 12)):
        pts = [
            (max(float(r["p_uc"]), x_floor), float(r["p_ind"]), str(r["model"]))
            for _, r in table.iterrows()
        ]
        for x, y, label in _merge_coincident(pts):
            flip = x > 0.15
            offset = 13 if y < 0.15 else (-13 if y > 0.92 else dy)
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(-11 if flip else 11, offset),
                ha="right" if flip else "left",
                va="center",
                fontsize=8.6,
                color=TEXT_PRIMARY,
                zorder=6,
            )

    ax.set_xscale("log")
    ax.set_xlim(x_floor, x_max)
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel("Kupiec POF p-value  (log scale)", fontsize=10.5, color=TEXT_SECONDARY)
    ax.set_ylabel("Christoffersen independence p-value", fontsize=10.5, color=TEXT_SECONDARY)
    ax.set_title(
        "A  Model validity frontier",
        fontsize=12, fontweight="bold", color=TEXT_PRIMARY, loc="left", pad=10,
    )

    ax.text(
        0.32, 0.995, "ACCEPTANCE ZONE", fontsize=8.4, fontweight="bold",
        color=C_GOOD, ha="center", va="top", zorder=4,
    )
    ax.text(
        10 ** -5.4, 0.995, "REJECTION ZONE  (Kupiec)", fontsize=8.4, fontweight="bold",
        color=C_CRITICAL, ha="center", va="top", zorder=4,
    )
    ax.text(
        REJECTION_LEVEL * 1.35, 0.40, "p = 0.05", fontsize=8.2,
        color=C_CRITICAL, ha="left", va="bottom", zorder=4,
    )

    leg = ax.legend(
        loc="upper left", bbox_to_anchor=(0.012, 0.88), fontsize=8.5,
        frameon=True, framealpha=0.96, edgecolor=GRID, facecolor=SURFACE,
        borderpad=0.7, labelspacing=0.5,
    )
    for text in leg.get_texts():
        text.set_color(TEXT_SECONDARY)


# --- Panel B -------------------------------------------------------------------
def plot_rate_comparison(ax, crash: pd.DataFrame, alpha: float = ALPHA) -> None:
    order = list(STANDALONE) + list(COMBINATIONS)
    ours = {str(r["model"]): float(r["hit_rate"]) * 100 for _, r in crash.iterrows()}
    paper = {b.model: b for b in PAPER_CRASH}
    y = np.arange(len(order))[::-1]

    ax.axvline(
        alpha * 100, color=C_CRITICAL, ls="--", lw=1.6, zorder=2,
        label=f"Target p = {alpha:.0%}",
    )

    for yi, model in zip(y, order):
        color = family_color(model)
        ox, px = ours[model], paper[model].violation_rate
        ax.plot([px, ox], [yi, yi], color=GRID, lw=2.0, zorder=3, solid_capstyle="round")
        ax.scatter(
            ox, yi, marker="o", s=135, facecolor=color, edgecolor=SURFACE,
            linewidth=1.8, zorder=5,
        )
        ax.scatter(
            px, yi, marker="D", s=92, facecolor=SURFACE, edgecolor=color,
            linewidth=2.0, zorder=5,
        )
        band = paper[model].cc_band
        if band:
            ax.annotate(
                band, (px, yi), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=10, fontweight="bold", color=C_CRITICAL, zorder=6,
            )
        ax.annotate(
            f"{ox:.2f}%", (ox, yi), textcoords="offset points", xytext=(0, -17),
            ha="center", fontsize=8.4, color=TEXT_PRIMARY, zorder=6,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=10, color=TEXT_PRIMARY)
    ax.set_xlim(0, 9.9)
    ax.set_ylim(-0.75, len(order) - 0.25)
    ax.set_xlabel("Violation rate during crash period (%)", fontsize=10.5, color=TEXT_SECONDARY)
    ax.set_title(
        "B  Crash-period violation rate: ours vs published",
        fontsize=12, fontweight="bold", color=TEXT_PRIMARY, loc="left", pad=10,
    )

    handles = [
        plt.Line2D([], [], marker="o", ls="none", markersize=11,
                   markerfacecolor=TEXT_SECONDARY, markeredgecolor=SURFACE,
                   markeredgewidth=1.8, label="This engine (S&P 500)"),
        plt.Line2D([], [], marker="D", ls="none", markersize=9,
                   markerfacecolor=SURFACE, markeredgecolor=TEXT_SECONDARY,
                   markeredgewidth=2.0, label="Halbleib & Pohlmeier (large cap)"),
        plt.Line2D([], [], ls="--", lw=1.6, color=C_CRITICAL, label=f"Target p = {alpha:.0%}"),
    ]
    leg = ax.legend(
        handles=handles, loc="lower right", fontsize=8.5, frameon=True,
        framealpha=0.95, edgecolor=GRID, facecolor=SURFACE, borderpad=0.7,
    )
    for text in leg.get_texts():
        text.set_color(TEXT_SECONDARY)

    ax.text(
        0.12, -0.64, "**  paper: conditional-coverage p < 0.05",
        fontsize=8, color=TEXT_MUTED, ha="left", va="bottom",
    )


# --- Figure --------------------------------------------------------------------
def build_figure(
    full: pd.DataFrame, crash: pd.DataFrame, meta: dict, alpha: float = ALPHA
) -> plt.Figure:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.9), facecolor=SURFACE)

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8, zorder=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9.2)

    plot_frontier(axes[0], full, crash, meta)
    plot_rate_comparison(axes[1], crash, alpha)

    fig.suptitle(
        "VaR model validity: Basel backtesting vs Halbleib & Pohlmeier (2012)",
        fontsize=15, fontweight="bold", color=TEXT_PRIMARY, x=0.055, ha="left", y=0.975,
    )
    fig.text(
        0.055,
        0.925,
        f"S&P 500, 1-day {1 - alpha:.0%} VaR, 250-day rolling window   |   "
        f"full sample {meta['full_start']} to {meta['full_end']} (S={meta['full_n']})   |   "
        f"crash period {CRASH_START.date()} to {CRASH_END.date()} (S={meta['crash_n']})",
        fontsize=9.3, color=TEXT_SECONDARY, ha="left",
    )
    fig.text(
        0.055,
        0.022,
        "Panel A plots this engine's results only: the paper publishes violation rates and a banded "
        "conditional-coverage p-value, never Kupiec/independence p-value pairs.\n"
        "CQOM/CCOM weights are re-calibrated daily on a rolling 250-day window (data from t-250 to t-1 only), "
        "so every series shown -- stand-alone and combined -- is genuinely out-of-sample.",
        fontsize=8.1, color=TEXT_MUTED, ha="left", va="bottom",
    )

    fig.subplots_adjust(left=0.055, right=0.985, top=0.855, bottom=0.145, wspace=0.185)
    return fig


def main() -> int:
    print("=" * 74)
    print("VaR backtesting visualization")
    print("=" * 74)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[assets] {ASSETS_DIR}")

    print("\n[computing results]")
    full, crash, meta = compute_results()
    print(f"  full sample  : S={meta['full_n']} ({meta['full_start']} -> {meta['full_end']})")
    print(f"  crash period : S={meta['crash_n']}  (paper reports 217 days)")
    w = meta.get("weights")
    if w is not None and len(w):
        g = w.groupby(["method", "model"])["weight"]
        print("  rolling weight dispersion (mean / sd / min / max):")
        for (method, model), s_ in g:
            print(f"    {method:<5} {model:<9} {s_.mean():+8.3f} {s_.std():8.3f} "
                  f"{s_.min():+9.3f} {s_.max():+9.3f}")

    cols = ["model", "n_violations", "hit_rate", "p_uc", "p_ind", "p_cc"]
    print("\n[full sample]")
    print(full[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n[crash period]")
    print(crash[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n[paper benchmarks, crash period]")
    for b in PAPER_CRASH:
        print(f"  {b.model:<9}{b.violation_rate:>6.2f}%  {BAND_LABEL[b.cc_band]:<24} {b.source}")

    print("\n[rendering]")
    fig = build_figure(full, crash, meta)
    fig.savefig(OUTPUT_PATH, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"  saved -> {OUTPUT_PATH.relative_to(PROJECT_ROOT)}  ({size_kb:.0f} KB)")

    print("\n[checks]")
    ok = True

    def check(label: str, cond: bool) -> bool:
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        return cond

    ok &= check("PNG written and non-trivial", OUTPUT_PATH.exists() and size_kb > 20)
    ok &= check("all five models present in both periods",
                set(full["model"]) == set(crash["model"]) == set(STANDALONE) | set(COMBINATIONS))
    ok &= check("paper benchmark covers every model",
                {b.model for b in PAPER_CRASH} == set(STANDALONE) | set(COMBINATIONS))
    def in_unit(table: pd.DataFrame) -> bool:
        vals = table[["p_uc", "p_ind"]].to_numpy(dtype=float)
        return bool(np.all(np.isfinite(vals)) and np.all(vals >= 0.0) and np.all(vals <= 1.0))

    ok &= check("all plotted p-values within [0, 1]", in_unit(full) and in_unit(crash))
    # Report, rather than assert, which models sit in the acceptance zone: that
    # is a research outcome that legitimately moves with the data, not a code
    # invariant.  (An earlier version asserted "both combinations accepted",
    # which only ever held because the weights were fitted in-sample.)
    for label, table in (("full sample", full), ("crash period", crash)):
        accepted = [
            str(r["model"])
            for _, r in table.iterrows()
            if r["p_uc"] > REJECTION_LEVEL and r["p_ind"] > REJECTION_LEVEL
        ]
        print(f"  [INFO] {label}: in acceptance zone -> "
              f"{', '.join(accepted) if accepted else 'none'}")

    # The acceptance criterion for the dynamic-weight refactor.
    cq_crash = crash.loc[crash["model"] == "CQOM", "p_uc"]
    ok &= check(
        f"OBJECTIVE - crash-period CQOM clears p > 0.05 "
        f"(p = {float(cq_crash.iloc[0]):.4f})",
        bool((cq_crash > REJECTION_LEVEL).all()),
    )
    ok &= check(
        "every model has a finite p-value in both periods",
        bool(
            np.isfinite(full[["p_uc", "p_ind"]].to_numpy()).all()
            and np.isfinite(crash[["p_uc", "p_ind"]].to_numpy()).all()
        ),
    )

    print("\n" + ("All checks passed." if ok else "SOME CHECKS FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
