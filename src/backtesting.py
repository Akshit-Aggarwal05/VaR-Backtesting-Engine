"""Basel backtesting harness: coverage tests and the traffic-light classifier.

Implements the VaR evaluation rules used in Halbleib & Pohlmeier (2012),
"Improving the value at risk forecasts: Theory and evidence from the financial
crisis", Journal of Economic Dynamics & Control 36, 1212-1228.

All tests operate on the hit sequence (eq. 2.3, p. 1214):

    H_t = 1(r_t < VaR_{t|t-1}(p)),   t = T+1, ..., T+S

which is binomial with parameter p under a correctly specified model.

  * **Kupiec (1995) POF / unconditional coverage.**  H0: E[H_t] = p.  The paper
    calls this "the unconditional coverage test (Christoffersen, 2003)" and
    notes the Basel Committee's rules "imply testing the null hypothesis
    H0 : E[H_t] = p".  LR_uc ~ chi2(1).

  * **Christoffersen (1998) independence.**  Tests "the degree of 'clustering'
    within the hit sequence", with H0 that a failure at t+1 is independent of a
    failure at t, via a first-order Markov transition matrix.  LR_ind ~ chi2(1).

  * **Conditional coverage.**  The paper: "One can assess the ability of a VaR
    model to provide the correct conditional coverage probability by
    simultaneously testing the two null hypotheses from the unconditional
    coverage and independence tests."  LR_cc = LR_uc + LR_ind ~ chi2(2).

  * **Basel traffic light.**  Zones are defined by quantiles of the
    Binomial(S=250, p=0.01) distribution (p. 1214): the green zone runs to the
    95% quantile and the red zone begins beyond the 99.99% quantile.  The paper
    describes the red boundary as the "99% quantile", but that is loose prose --
    it would imply a yellow ceiling of 6 and contradict the Basel Committee's
    published table.  ``verify_basel_thresholds`` derives the N<=4 / 5-9 / >=10
    cut-offs from the distribution rather than hard-coding them on trust.

Numerical note: every log-likelihood uses ``scipy.special.xlogy``, which defines
0*log(0) = 0.  Without it, the common cases N = 0 and N = S produce NaN instead
of a finite statistic.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binom, chi2

try:
    from data_loader import DATE_COLUMN, load_returns
    from optimizations import combined_frame, fit_combinations, prepare_panel
    from var_models import (
        ACTUAL_COLUMN,
        ALPHA,
        VAR_COLUMN_PREFIX,
        WINDOW,
        default_models,
        rolling_var_forecasts,
    )
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_loader import DATE_COLUMN, load_returns
    from optimizations import combined_frame, fit_combinations, prepare_panel
    from var_models import (
        ACTUAL_COLUMN,
        ALPHA,
        VAR_COLUMN_PREFIX,
        WINDOW,
        default_models,
        rolling_var_forecasts,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data"

# Basel Committee (1996) traffic-light cut-offs for S = 250, p = 0.01.
BASEL_WINDOW = 250
BASEL_ALPHA = 0.01      # the table is defined only for p = 0.01
GREEN_MAX = 4
YELLOW_MAX = 9

ZONE_GREEN = "GREEN"
ZONE_YELLOW = "YELLOW"
ZONE_RED = "RED"
ZONE_INSUFFICIENT = "N/A"


# --- Test results --------------------------------------------------------------
@dataclass
class TestResult:
    """Outcome of a likelihood-ratio test."""

    name: str
    statistic: float
    pvalue: float
    df: int
    identified: bool = True
    detail: dict[str, float] = None

    def __post_init__(self) -> None:
        if self.detail is None:
            self.detail = {}

    def reject(self, level: float = 0.05) -> bool:
        """True if H0 is rejected at ``level``."""
        return bool(np.isfinite(self.pvalue) and self.pvalue < level)


def _as_hits(hits: np.ndarray | pd.Series) -> np.ndarray:
    """Validate and coerce a hit sequence to a 0/1 float array."""
    arr = np.asarray(hits, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("hit sequence is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("hit sequence contains NaN/inf")
    if not np.all((arr == 0.0) | (arr == 1.0)):
        raise ValueError("hit sequence must contain only 0 and 1")
    return arr


def hit_sequence(actual: np.ndarray | pd.Series, var: np.ndarray | pd.Series) -> np.ndarray:
    """H_t = 1(r_t < VaR_t), eq. 2.3."""
    a = np.asarray(actual, dtype=float)
    v = np.asarray(var, dtype=float)
    if a.shape != v.shape:
        raise ValueError(f"shape mismatch: returns {a.shape} vs VaR {v.shape}")
    return (a < v).astype(float)


# --- Kupiec POF ----------------------------------------------------------------
def kupiec_pof(hits: np.ndarray | pd.Series, alpha: float = ALPHA) -> TestResult:
    """Kupiec (1995) proportion-of-failures / unconditional coverage test.

        LR_uc = -2 ln[ L(p) / L(pi_hat) ],   pi_hat = N / S,   LR_uc ~ chi2(1)
    """
    h = _as_hits(hits)
    S = h.size
    N = float(h.sum())
    pi_hat = N / S

    # log L under H0 (rate fixed at alpha) and unrestricted (rate = pi_hat).
    ll_null = xlogy(S - N, 1.0 - alpha) + xlogy(N, alpha)
    ll_alt = xlogy(S - N, 1.0 - pi_hat) + xlogy(N, pi_hat)

    stat = float(2.0 * (ll_alt - ll_null))
    stat = max(stat, 0.0)                      # guard tiny negative round-off
    return TestResult(
        name="Kupiec POF (uncond. coverage)",
        statistic=stat,
        pvalue=float(chi2.sf(stat, 1)),
        df=1,
        identified=True,
        detail={
            "n_obs": float(S),
            "n_violations": N,
            "hit_rate": pi_hat,
            "expected": S * alpha,
            "alpha": alpha,
        },
    )


# --- Christoffersen independence ----------------------------------------------
def transition_counts(hits: np.ndarray | pd.Series) -> dict[str, int]:
    """First-order Markov transition counts of the hit sequence."""
    h = _as_hits(hits)
    prev, nxt = h[:-1], h[1:]
    return {
        "n00": int(np.sum((prev == 0) & (nxt == 0))),
        "n01": int(np.sum((prev == 0) & (nxt == 1))),
        "n10": int(np.sum((prev == 1) & (nxt == 0))),
        "n11": int(np.sum((prev == 1) & (nxt == 1))),
    }


def christoffersen_independence(hits: np.ndarray | pd.Series) -> TestResult:
    """Christoffersen (1998) independence test for violation clustering.

    Compares a first-order Markov chain against an i.i.d. Bernoulli chain:

        pi_01 = n01/(n00+n01),  pi_11 = n11/(n10+n11),  pi = (n01+n11)/n
        LR_ind = -2 ln[ L_iid / L_markov ] ~ chi2(1)

    When no observation follows a violation (n10 = n11 = 0) the transition
    probability pi_11 is not identified; the ``xlogy`` terms then vanish, the
    statistic is exactly 0, and ``identified`` is set False.  This is the normal
    situation at p = 0.01 with few violations, so the flag matters: a p-value of
    1.0 there means "no evidence available", not "independence confirmed".
    """
    h = _as_hits(hits)
    c = transition_counts(h)
    n00, n01, n10, n11 = c["n00"], c["n01"], c["n10"], c["n11"]
    total = n00 + n01 + n10 + n11

    if total == 0:
        return TestResult(
            name="Christoffersen independence",
            statistic=0.0,
            pvalue=1.0,
            df=1,
            identified=False,
            detail={k: float(v) for k, v in c.items()},
        )

    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / total

    ll_iid = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi)
    ll_markov = (
        xlogy(n00, 1.0 - pi_01)
        + xlogy(n01, pi_01)
        + xlogy(n10, 1.0 - pi_11)
        + xlogy(n11, pi_11)
    )

    stat = float(2.0 * (ll_markov - ll_iid))
    stat = max(stat, 0.0)
    identified = (n10 + n11) > 0 and (n00 + n01) > 0
    return TestResult(
        name="Christoffersen independence",
        statistic=stat,
        pvalue=float(chi2.sf(stat, 1)),
        df=1,
        identified=bool(identified),
        detail={
            **{k: float(v) for k, v in c.items()},
            "pi_01": pi_01,
            "pi_11": pi_11,
            "pi": pi,
        },
    )


def christoffersen_conditional_coverage(
    hits: np.ndarray | pd.Series, alpha: float = ALPHA
) -> TestResult:
    """Joint conditional coverage test: LR_cc = LR_uc + LR_ind ~ chi2(2)."""
    uc = kupiec_pof(hits, alpha)
    ind = christoffersen_independence(hits)
    stat = float(uc.statistic + ind.statistic)
    return TestResult(
        name="Christoffersen conditional coverage",
        statistic=stat,
        pvalue=float(chi2.sf(stat, 2)),
        df=2,
        identified=ind.identified,
        detail={"lr_uc": uc.statistic, "lr_ind": ind.statistic},
    )


# --- Basel traffic light -------------------------------------------------------
def basel_zone_from_count(n_violations: int) -> str:
    """Classify a 250-day violation count into its Basel regulatory zone."""
    n = int(n_violations)
    if n <= GREEN_MAX:
        return ZONE_GREEN
    if n <= YELLOW_MAX:
        return ZONE_YELLOW
    return ZONE_RED


@dataclass
class BaselResult:
    """Traffic-light outcome for one 250-day window."""

    zone: str
    n_violations: int
    window: int
    start: object = None
    end: object = None
    sufficient: bool = True
    reason: str = ""


def basel_traffic_light(
    hits: np.ndarray | pd.Series,
    window: int = BASEL_WINDOW,
    dates: pd.Series | None = None,
    alpha: float = ALPHA,
) -> BaselResult:
    """Classify the most recent ``window`` days of the hit sequence.

    The N<=4 / 5-9 / >=10 table is specific to S = 250 and p = 0.01, so the
    classification is withheld (rather than silently mis-applied) when either
    the sample is too short or ``alpha`` differs from the Basel level.
    """
    h = _as_hits(hits)
    if not np.isclose(alpha, BASEL_ALPHA):
        return BaselResult(
            zone=ZONE_INSUFFICIENT,
            n_violations=int(h[-window:].sum()) if h.size >= window else int(h.sum()),
            window=int(min(h.size, window)),
            sufficient=False,
            reason=f"Basel zones are defined for p={BASEL_ALPHA}, not {alpha}",
        )
    if h.size < window:
        return BaselResult(
            zone=ZONE_INSUFFICIENT,
            n_violations=int(h.sum()),
            window=int(h.size),
            sufficient=False,
            reason=f"only {h.size} observations, need {window}",
        )
    recent = h[-window:]
    start = end = None
    if dates is not None:
        d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
        start, end = d.iloc[-window], d.iloc[-1]
    return BaselResult(
        zone=basel_zone_from_count(int(recent.sum())),
        n_violations=int(recent.sum()),
        window=int(window),
        start=start,
        end=end,
        sufficient=True,
    )


def rolling_basel_zones(
    hits: np.ndarray | pd.Series, window: int = BASEL_WINDOW
) -> pd.Series:
    """Basel zone for every rolling ``window`` of the hit sequence."""
    h = _as_hits(hits)
    counts = pd.Series(h).rolling(window).sum()
    return counts.map(lambda n: basel_zone_from_count(n) if np.isfinite(n) else ZONE_INSUFFICIENT)


GREEN_THRESHOLD = 0.95       # Basel Committee (1996): green zone ends here
RED_THRESHOLD = 0.9999       # ... and the red zone begins here


def verify_basel_thresholds(
    window: int = BASEL_WINDOW, alpha: float = ALPHA
) -> dict[str, object]:
    """Derive the N<=4 / 5-9 / >=10 cut-offs from Binomial(window, alpha).

    A zone boundary is the largest N whose cumulative probability is still
    *below* the threshold, so it is ``ppf(threshold) - 1``.

    Note the thresholds.  The paper describes the red zone as lying "outside the
    99% quantile", but that is loose prose: for Binomial(250, 0.01) the 99%
    quantile implies a yellow ceiling of 6, which contradicts the Basel
    Committee's published table.  The actual red-zone boundary is the **99.99%**
    quantile -- P(X<=9) = 0.99975 < 0.9999 <= P(X<=10) = 0.99995 -- which
    reproduces the canonical N<=4 / 5-9 / >=10 cut-offs used here.
    """
    green_derived = int(binom.ppf(GREEN_THRESHOLD, window, alpha)) - 1
    yellow_derived = int(binom.ppf(RED_THRESHOLD, window, alpha)) - 1
    return {
        "green_threshold": GREEN_THRESHOLD,
        "red_threshold": RED_THRESHOLD,
        "green_max_derived": green_derived,
        "yellow_max_derived": yellow_derived,
        "green_max_used": GREEN_MAX,
        "yellow_max_used": YELLOW_MAX,
        "cdf_at_green_max": float(binom.cdf(GREEN_MAX, window, alpha)),
        "cdf_at_green_max_plus1": float(binom.cdf(GREEN_MAX + 1, window, alpha)),
        "cdf_at_yellow_max": float(binom.cdf(YELLOW_MAX, window, alpha)),
        "cdf_at_yellow_max_plus1": float(binom.cdf(YELLOW_MAX + 1, window, alpha)),
        "yellow_max_if_99pct": int(binom.ppf(0.99, window, alpha)) - 1,
        "matches": bool(green_derived == GREEN_MAX and yellow_derived == YELLOW_MAX),
    }


# --- Summary -------------------------------------------------------------------
def backtest_var(
    actual: np.ndarray | pd.Series,
    var: np.ndarray | pd.Series,
    alpha: float = ALPHA,
    dates: pd.Series | None = None,
    basel_window: int = BASEL_WINDOW,
) -> dict[str, object]:
    """Run the full battery on one VaR series."""
    hits = hit_sequence(actual, var)
    uc = kupiec_pof(hits, alpha)
    ind = christoffersen_independence(hits)
    cc = christoffersen_conditional_coverage(hits, alpha)
    basel = basel_traffic_light(hits, window=basel_window, dates=dates, alpha=alpha)
    return {
        "n_obs": int(hits.size),
        "n_violations": int(hits.sum()),
        "hit_rate": float(hits.mean()),
        "expected_violations": float(hits.size * alpha),
        "lr_uc": uc.statistic,
        "p_uc": uc.pvalue,
        "lr_ind": ind.statistic,
        "p_ind": ind.pvalue,
        "ind_identified": ind.identified,
        "lr_cc": cc.statistic,
        "p_cc": cc.pvalue,
        "basel_zone": basel.zone,
        "basel_violations": basel.n_violations,
        "basel_window": basel.window,
        "basel_applicable": basel.sufficient,
        "basel_reason": basel.reason,
        **{f"trans_{k}": v for k, v in ind.detail.items() if k.startswith("n")},
    }


def backtest_summary(
    panel: pd.DataFrame,
    alpha: float = ALPHA,
    basel_window: int = BASEL_WINDOW,
) -> pd.DataFrame:
    """Backtest every VaR column in ``panel`` and return a tidy table."""
    cols = [c for c in panel.columns if c.startswith(VAR_COLUMN_PREFIX)]
    if not cols:
        raise ValueError("no VaR columns found")
    actual = panel[ACTUAL_COLUMN].to_numpy(dtype=float)
    dates = panel[DATE_COLUMN] if DATE_COLUMN in panel.columns else None
    rows = []
    for col in cols:
        res = backtest_var(
            actual, panel[col].to_numpy(dtype=float), alpha, dates, basel_window
        )
        rows.append({"model": col[len(VAR_COLUMN_PREFIX):], **res})
    return pd.DataFrame(rows)


# --- Panel construction (with cache) -------------------------------------------
def _cache_path(test_days: int | None, window: int, alpha: float) -> Path:
    tag = "full" if test_days is None else f"n{test_days}"
    return CACHE_DIR / f"var_panel_{tag}_w{window}_a{alpha}.csv"


def build_panel(
    test_days: int | None = None,
    window: int = WINDOW,
    alpha: float = ALPHA,
    refresh: bool = False,
    include_combinations: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, bool]:
    """Assemble the VaR panel, reusing a cached CSV when one is available.

    GARCH refitting on every rolling window is the expensive step, so the
    stand-alone panel is cached; combination weights are cheap and refitted.
    Returns ``(panel, from_cache)``.
    """
    cache = _cache_path(test_days, window, alpha)
    returns = load_returns()
    slice_df = returns if test_days is None else returns.iloc[:test_days]
    slice_df = slice_df.reset_index(drop=True)

    standalone: pd.DataFrame | None = None
    from_cache = False
    if cache.exists() and not refresh:
        candidate = pd.read_csv(cache, parse_dates=[DATE_COLUMN])
        expected = {ACTUAL_COLUMN, DATE_COLUMN} | {
            f"{VAR_COLUMN_PREFIX}{m.name}" for m in default_models()
        }
        if expected <= set(candidate.columns) and len(candidate) == len(slice_df) - window:
            standalone, from_cache = candidate, True
            if verbose:
                print(f"  reusing cache      : {cache.name}")
        elif verbose:
            print(f"  cache unusable, rebuilding: {cache.name}")

    if standalone is None:
        if verbose:
            print(f"  fitting rolling VaR ({len(slice_df) - window} forecasts) ...")
        rolling = rolling_var_forecasts(
            slice_df,
            models=default_models(),
            window=window,
            alpha=alpha,
            progress_every=500 if verbose else 0,
        )
        standalone = rolling.var_frame
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        out = standalone.copy()
        out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN]).dt.strftime("%Y-%m-%d")
        out.to_csv(cache, index=False)
        if verbose:
            print(f"  cached to          : {cache.name}")

    panel, _, dropped = prepare_panel(standalone)
    if verbose and dropped:
        print(f"  rows dropped (NaN) : {dropped}")

    if include_combinations:
        if verbose:
            print("  fitting CQOM / CCOM combinations ...")
        results = fit_combinations(panel, alpha=alpha)
        panel = combined_frame(panel, results)
    return panel, from_cache


# --- Verification block --------------------------------------------------------
def _check(label: str, ok: bool) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def _self_tests(alpha: float) -> bool:
    """Validate the LR statistics against cases with known answers."""
    ok = True
    rng = np.random.default_rng(7)

    # 1. Hit rate exactly alpha -> LR_uc = 0, p = 1.
    exact = np.zeros(1000)
    exact[:10] = 1.0
    r = kupiec_pof(exact, 0.01)
    ok &= _check(
        f"Kupiec: hit rate == alpha gives LR_uc = 0 (got {r.statistic:.2e}, p={r.pvalue:.3f})",
        abs(r.statistic) < 1e-9 and abs(r.pvalue - 1.0) < 1e-9,
    )

    # 2. Grossly wrong rate -> strong rejection.
    bad = np.zeros(1000)
    bad[:100] = 1.0           # 10% hits against a 1% target
    r = kupiec_pof(bad, 0.01)
    ok &= _check(
        f"Kupiec: 10% hits vs 1% target is rejected (LR={r.statistic:.1f}, p={r.pvalue:.2e})",
        r.pvalue < 1e-10,
    )

    # 3. N = 0 must stay finite (the 0*log(0) case).
    r = kupiec_pof(np.zeros(500), 0.01)
    expected = -2.0 * 500 * np.log(1.0 - 0.01)
    ok &= _check(
        f"Kupiec: N=0 is finite and equals -2*S*ln(1-p) ({r.statistic:.4f} vs {expected:.4f})",
        np.isfinite(r.statistic) and abs(r.statistic - expected) < 1e-9,
    )

    # 4. i.i.d. Bernoulli hits -> independence not rejected.
    iid = (rng.random(5000) < 0.05).astype(float)
    r = christoffersen_independence(iid)
    ok &= _check(
        f"Christoffersen: i.i.d. hits not rejected (LR={r.statistic:.3f}, p={r.pvalue:.3f})",
        r.pvalue > 0.05,
    )

    # 5. Perfectly clustered hits -> independence strongly rejected.
    clustered = np.zeros(1000)
    clustered[100:150] = 1.0
    clustered[400:450] = 1.0
    r = christoffersen_independence(clustered)
    ok &= _check(
        f"Christoffersen: clustered hits rejected (LR={r.statistic:.1f}, p={r.pvalue:.2e})",
        r.pvalue < 1e-10,
    )

    # 6. Unidentified case is flagged rather than silently passing.
    lone = np.zeros(100)
    lone[-1] = 1.0            # the only hit is last, so nothing follows it
    r = christoffersen_independence(lone)
    ok &= _check(
        "Christoffersen: single trailing hit flagged as unidentified",
        (not r.identified) and r.statistic == 0.0,
    )

    # 7. LR_cc is exactly the sum of the two components.
    mixed = (rng.random(2000) < 0.012).astype(float)
    uc, ind, cc = (
        kupiec_pof(mixed, alpha),
        christoffersen_independence(mixed),
        christoffersen_conditional_coverage(mixed, alpha),
    )
    ok &= _check(
        "LR_cc == LR_uc + LR_ind",
        abs(cc.statistic - (uc.statistic + ind.statistic)) < 1e-12,
    )

    # 8. Basel cut-offs derived from the binomial, not assumed.
    thr = verify_basel_thresholds()
    ok &= _check(
        f"Basel zones derived from Binomial(250,0.01): green<={thr['green_max_derived']}, "
        f"yellow<={thr['yellow_max_derived']} "
        f"(CDF {thr['cdf_at_green_max']:.4f}/{thr['cdf_at_yellow_max']:.6f})",
        bool(thr["matches"]),
    )
    ok &= _check(
        "Basel classifier maps 4->GREEN, 5->YELLOW, 9->YELLOW, 10->RED",
        basel_zone_from_count(4) == ZONE_GREEN
        and basel_zone_from_count(5) == ZONE_YELLOW
        and basel_zone_from_count(9) == ZONE_YELLOW
        and basel_zone_from_count(10) == ZONE_RED,
    )
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Basel backtesting harness for VaR models.")
    parser.add_argument("--test-days", type=int, default=None,
                        help="limit to the first N returns (default: full sample)")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--basel-window", type=int, default=BASEL_WINDOW)
    parser.add_argument("--refresh", action="store_true", help="ignore the cached panel")
    args = parser.parse_args(argv)

    print("=" * 78)
    print("Basel backtesting harness - Halbleib & Pohlmeier (2012)")
    print("=" * 78)

    print("\n[self-tests on known cases]")
    tests_ok = _self_tests(args.alpha)

    print("\n[panel]")
    panel, from_cache = build_panel(
        test_days=args.test_days, window=args.window, alpha=args.alpha, refresh=args.refresh
    )
    dates = pd.to_datetime(panel[DATE_COLUMN])
    print(f"  out-of-sample S    : {len(panel)}")
    print(f"  period             : {dates.iloc[0].date()} -> {dates.iloc[-1].date()}")
    print(f"  alpha              : {args.alpha}  (expected violations "
          f"{len(panel) * args.alpha:.1f})")
    print(f"  from cache         : {from_cache}")

    table = backtest_summary(panel, alpha=args.alpha, basel_window=args.basel_window)

    print("\n[backtest summary]")
    print("  Kupiec LR_uc ~ chi2(1) | Christoffersen LR_ind ~ chi2(1) | LR_cc ~ chi2(2)")
    print("  p < 0.05 marked * ; Basel zone uses the most recent "
          f"{args.basel_window}-day window")
    head = (
        f"  {'model':<10}{'N':>5}{'rate':>9}{'LR_uc':>9}{'p_uc':>10}"
        f"{'LR_ind':>9}{'p_ind':>10}{'LR_cc':>9}{'p_cc':>10}{'Basel':>8}{'N250':>6}"
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for _, r in table.iterrows():
        def mark(p: float) -> str:
            return f"{p:.4f}*" if p < 0.05 else f"{p:.4f} "
        flag = "" if r["ind_identified"] else "u"
        print(
            f"  {r['model']:<10}{int(r['n_violations']):>5}{r['hit_rate']:>8.3%}"
            f"{r['lr_uc']:>9.3f}{mark(r['p_uc']):>10}"
            f"{r['lr_ind']:>9.3f}{mark(r['p_ind']) + flag:>10}"
            f"{r['lr_cc']:>9.3f}{mark(r['p_cc']):>10}"
            f"{r['basel_zone']:>8}{int(r['basel_violations']):>6}"
        )
    print("\n  'u' = independence test not identified (no observation follows a violation)")

    print("\n[hit transition counts]")
    th = f"  {'model':<10}{'n00':>8}{'n01':>8}{'n10':>8}{'n11':>8}"
    print(th)
    print("  " + "-" * (len(th) - 2))
    for _, r in table.iterrows():
        print(
            f"  {r['model']:<10}{int(r['trans_n00']):>8}{int(r['trans_n01']):>8}"
            f"{int(r['trans_n10']):>8}{int(r['trans_n11']):>8}"
        )

    print("\n[Basel zone occupancy across all rolling 250-day windows]")
    zh = f"  {'model':<10}{'GREEN':>9}{'YELLOW':>9}{'RED':>9}{'windows':>9}"
    print(zh)
    print("  " + "-" * (len(zh) - 2))
    actual = panel[ACTUAL_COLUMN].to_numpy(dtype=float)
    for col in [c for c in panel.columns if c.startswith(VAR_COLUMN_PREFIX)]:
        zones = rolling_basel_zones(
            hit_sequence(actual, panel[col].to_numpy(dtype=float)), args.basel_window
        )
        valid = zones[zones != ZONE_INSUFFICIENT]
        counts = valid.value_counts()
        print(
            f"  {col[len(VAR_COLUMN_PREFIX):]:<10}"
            f"{counts.get(ZONE_GREEN, 0):>9}{counts.get(ZONE_YELLOW, 0):>9}"
            f"{counts.get(ZONE_RED, 0):>9}{len(valid):>9}"
        )

    print("\n[caveat]")
    print("  CQOM and CCOM weights are fitted on this same period, so their rows are an")
    print("  IN-SAMPLE assessment (the paper's Tables 1-2). The paper's out-of-sample")
    print("  assessment (Table 3) re-estimates the weights at each forecast date; that is")
    print("  not what is reported here. HS / GARCH-N / GARCH-t are genuine out-of-sample.")

    print("\n[sanity checks]")
    ok = tests_ok
    ok &= _check("summary table has one row per VaR series", len(table) == 5)
    ok &= _check(
        "all p-values in [0, 1]",
        bool(
            table[["p_uc", "p_ind", "p_cc"]].ge(0).all().all()
            and table[["p_uc", "p_ind", "p_cc"]].le(1).all().all()
        ),
    )
    ok &= _check(
        "all LR statistics finite and non-negative",
        bool(
            np.isfinite(table[["lr_uc", "lr_ind", "lr_cc"]].to_numpy()).all()
            and (table[["lr_uc", "lr_ind", "lr_cc"]].to_numpy() >= 0).all()
        ),
    )
    valid_zones = [ZONE_GREEN, ZONE_YELLOW, ZONE_RED]
    ok &= _check(
        "every Basel zone is a valid label where the table applies",
        bool(
            table.loc[table["basel_applicable"], "basel_zone"].isin(valid_zones).all()
            and (table.loc[~table["basel_applicable"], "basel_zone"] == ZONE_INSUFFICIENT).all()
        ),
    )
    ok &= _check(
        "violation counts agree with an independent recomputation",
        all(
            int((panel[ACTUAL_COLUMN] < panel[f"{VAR_COLUMN_PREFIX}{r['model']}"]).sum())
            == int(r["n_violations"])
            for _, r in table.iterrows()
        ),
    )
    ok &= _check(
        "transition counts sum to S - 1",
        bool(
            (
                table[["trans_n00", "trans_n01", "trans_n10", "trans_n11"]].sum(axis=1)
                == table["n_obs"] - 1
            ).all()
        ),
    )

    print("\n" + ("All checks passed." if ok else "SOME CHECKS FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
