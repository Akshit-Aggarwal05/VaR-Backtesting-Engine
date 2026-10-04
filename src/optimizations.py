"""Optimal VaR combination methods (CCOM and CQOM).

Implements Section 2.2 of Halbleib & Pohlmeier (2012), "Improving the value at
risk forecasts: Theory and evidence from the financial crisis", Journal of
Economic Dynamics & Control 36, 1212-1228.

Both methods combine stand-alone VaR forecasts linearly (eq. 2.4 / 2.6):

    VaR_comb_{T+s}(lambda) = lambda_0 + sum_j lambda_j * VaR^j_{T+s}

CQOM -- Conditional Quantile Optimization Method (Section 2.2.2, eq. 2.8).
    Models the conditional p-quantile of returns as a linear function of the
    stand-alone forecasts and solves the Koenker & Bassett (1978) check-function
    problem, i.e. a quantile regression of r on the VaR matrix at tau = p.

CCOM -- Conditional Coverage Optimization Method (Section 2.2.1, eq. 2.7).
    Method-of-moments on the hit sequence H_t = 1(r_t < VaR_comb_t), using the
    two restrictions implied by its martingale property:

        psi_1 = (1/S)     * sum_{s=1..S} [H_s - p]              -> unconditional coverage
        psi_2 = (1/(S-1)) * sum_{s=2..S} [H_s - p] * H_{s-1}    -> independence of hits

    and minimizing psi' psi.

Two implementation points that decide whether CCOM works at all:

  1. The raw objective is a step function of the weights, so its gradient is
     zero almost everywhere and ``scipy.optimize.minimize`` would return the
     starting guess unchanged.  The paper's own fix is applied: H is replaced by
     the logistic smoother (p. 1215)

         F(lambda, X, h) = 1 / (1 + exp[(r_{T+s} - VaR_comb_{T+s}) / h])

     with bandwidth h -> 0 and S*h -> infinity as S grows.

  2. The paper notes gradient methods "may encounter difficulties in finding the
     global, rather than the local minimum" and switches to simulated annealing.
     We stay with ``minimize`` as specified, but run it from several
     deterministic starting points and keep the best -- the same concern,
     addressed within the required optimizer.

Reported hit statistics are always recomputed from the *discrete* indicator at
the optimum, never from the smoothed surrogate, so the diagnostics are honest.

Deviations from the paper, both configurable (see ``CombinationSpec``):
  * ``include_intercept`` defaults to False per the current task spec; the paper
    keeps lambda_0 free to "correct for the biases in the VaR's forecasts".
  * ``sum_to_one`` defaults to False; the paper restricts the VaR loadings to
    sum to one (eq. 2.6) for interpretability, while imposing no bounds.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.regression.quantile_regression import QuantReg

try:
    from data_loader import DATE_COLUMN, RETURN_COLUMN, load_returns
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
    from data_loader import DATE_COLUMN, RETURN_COLUMN, load_returns
    from var_models import (
        ACTUAL_COLUMN,
        ALPHA,
        VAR_COLUMN_PREFIX,
        WINDOW,
        default_models,
        rolling_var_forecasts,
    )


class CombinationError(RuntimeError):
    """Raised when a combination cannot be estimated."""


# --- Weight parameterization ---------------------------------------------------
@dataclass(frozen=True)
class CombinationSpec:
    """How the loading vector is parameterized.

    ``include_intercept`` adds the paper's lambda_0.  ``sum_to_one`` imposes
    eq. 2.6's restriction that the VaR loadings sum to one, leaving the last
    loading implied.  Neither imposes sign or boundary constraints, matching the
    paper's choice to "allow the combined estimator to take values beyond the
    values of the stand-alone estimators".
    """

    n_models: int
    include_intercept: bool = False
    sum_to_one: bool = False

    @property
    def n_free(self) -> int:
        n = self.n_models - 1 if self.sum_to_one else self.n_models
        return n + (1 if self.include_intercept else 0)

    def unpack(self, params: np.ndarray) -> tuple[float, np.ndarray]:
        """Map free parameters to (intercept, full weight vector)."""
        params = np.asarray(params, dtype=float).ravel()
        offset = 0
        intercept = 0.0
        if self.include_intercept:
            intercept = float(params[0])
            offset = 1
        tail = params[offset:]
        if self.sum_to_one:
            weights = np.empty(self.n_models, dtype=float)
            weights[:-1] = tail
            weights[-1] = 1.0 - tail.sum()
        else:
            weights = tail.astype(float)
        return intercept, weights

    def pack(self, intercept: float, weights: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`unpack`, for building starting values."""
        weights = np.asarray(weights, dtype=float)
        tail = weights[:-1] if self.sum_to_one else weights
        return np.concatenate(([intercept], tail)) if self.include_intercept else tail.copy()


def combine(X: np.ndarray, intercept: float, weights: np.ndarray) -> np.ndarray:
    """Linear VaR combination: lambda_0 + sum_j lambda_j VaR^j."""
    return float(intercept) + X @ np.asarray(weights, dtype=float)


# --- Hit sequence --------------------------------------------------------------
def hit_sequence(actual: np.ndarray, var: np.ndarray) -> np.ndarray:
    """H_t = 1(r_t < VaR_t), the binary exceedance indicator."""
    return (np.asarray(actual) < np.asarray(var)).astype(float)


def hit_diagnostics(hits: np.ndarray, alpha: float = ALPHA) -> dict[str, float]:
    """Discrete moment diagnostics for a hit sequence."""
    hits = np.asarray(hits, dtype=float)
    S = hits.size
    psi1 = float(np.mean(hits - alpha))
    psi2 = (
        float(np.sum((hits[1:] - alpha) * hits[:-1]) / (S - 1)) if S > 1 else np.nan
    )
    # Plain lag-1 autocorrelation, reported alongside the paper's moment.
    if S > 2 and hits.std() > 0:
        autocorr = float(np.corrcoef(hits[1:], hits[:-1])[0, 1])
    else:
        autocorr = np.nan
    return {
        "n_obs": float(S),
        "n_hits": float(hits.sum()),
        "hit_rate": float(hits.mean()),
        "target": float(alpha),
        "psi1": psi1,
        "psi2": psi2,
        "objective": psi1**2 + (psi2**2 if np.isfinite(psi2) else 0.0),
        "lag1_autocorr": autocorr,
    }


# --- Result container ----------------------------------------------------------
@dataclass
class CombinationResult:
    """Fitted combination: weights, the combined series, and diagnostics."""

    name: str
    spec: CombinationSpec
    model_names: list[str]
    intercept: float
    weights: np.ndarray
    var: pd.Series                        # combined VaR, indexed by date
    objective: float
    converged: bool
    alpha: float = ALPHA
    diagnostics: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def weight_table(self) -> pd.Series:
        idx = (["intercept"] if self.spec.include_intercept else []) + list(self.model_names)
        vals = ([self.intercept] if self.spec.include_intercept else []) + list(self.weights)
        return pd.Series(vals, index=idx, name=self.name)

    @property
    def weight_sum(self) -> float:
        return float(np.sum(self.weights))


# --- Input preparation ---------------------------------------------------------
def prepare_panel(var_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
    """Drop rows with any missing VaR/return and return the clean panel.

    The rolling engine emits no forecast until the estimation window is full, so
    the only NaNs here come from failed GARCH fits; either way they are dropped
    and counted.
    """
    cols = [c for c in var_frame.columns if c.startswith(VAR_COLUMN_PREFIX)]
    if not cols:
        raise CombinationError("no VaR columns found in the input frame")
    if ACTUAL_COLUMN not in var_frame.columns:
        raise CombinationError(f"missing '{ACTUAL_COLUMN}' column")

    needed = [ACTUAL_COLUMN, *cols]
    before = len(var_frame)
    clean = var_frame.dropna(subset=needed).reset_index(drop=True)
    model_names = [c[len(VAR_COLUMN_PREFIX):] for c in cols]
    return clean, model_names, before - len(clean)


def _design(panel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    cols = [c for c in panel.columns if c.startswith(VAR_COLUMN_PREFIX)]
    X = panel[cols].to_numpy(dtype=float)
    y = panel[ACTUAL_COLUMN].to_numpy(dtype=float)
    dates = pd.to_datetime(panel[DATE_COLUMN]) if DATE_COLUMN in panel else pd.Series(panel.index)
    return X, y, dates


# --- CQOM ----------------------------------------------------------------------
def fit_cqom(
    var_frame: pd.DataFrame,
    alpha: float = ALPHA,
    include_intercept: bool = False,
    sum_to_one: bool = False,
) -> CombinationResult:
    """Conditional Quantile Optimization Method (eq. 2.8).

    Quantile regression of the realized return on the stand-alone VaR forecasts
    at tau = alpha.  Under ``sum_to_one`` the restriction is absorbed exactly by
    regressing (r - VaR_k) on the differences (VaR_j - VaR_k), so the result is
    still a single unconstrained quantile regression.
    """
    panel, model_names, _ = prepare_panel(var_frame)
    X, y, dates = _design(panel)
    spec = CombinationSpec(X.shape[1], include_intercept, sum_to_one)

    if len(panel) <= spec.n_free:
        raise CombinationError(
            f"CQOM needs more observations ({len(panel)}) than parameters ({spec.n_free})"
        )

    if sum_to_one:
        design = X[:, :-1] - X[:, [-1]]
        target = y - X[:, -1]
    else:
        design = X
        target = y
    if include_intercept:
        design = np.column_stack([np.ones(len(design)), design])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = QuantReg(target, design).fit(q=alpha)

    params = np.asarray(fit.params, dtype=float)
    intercept, weights = spec.unpack(params)
    combined = combine(X, intercept, weights)
    diag = hit_diagnostics(hit_sequence(y, combined), alpha)

    return CombinationResult(
        name="CQOM",
        spec=spec,
        model_names=model_names,
        intercept=intercept,
        weights=weights,
        var=pd.Series(combined, index=dates.to_numpy(), name="VaR_CQOM"),
        objective=float(_check_loss(y, combined, alpha)),
        converged=bool(np.all(np.isfinite(params))),
        alpha=alpha,
        diagnostics=diag,
        notes="quantile regression (Koenker-Bassett check loss)",
    )


def _check_loss(y: np.ndarray, fitted: np.ndarray, alpha: float) -> float:
    """Koenker-Bassett asymmetric loss, the quantity CQOM minimizes."""
    resid = y - fitted
    return float(np.sum(np.where(resid >= 0, alpha * resid, (alpha - 1.0) * resid)))


# --- CCOM ----------------------------------------------------------------------
def _ccom_objective(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    spec: CombinationSpec,
    bandwidth: float,
) -> float:
    """psi' psi from eq. 2.7, with H replaced by the paper's logistic smoother."""
    intercept, weights = spec.unpack(params)
    combined = combine(X, intercept, weights)
    if not np.all(np.isfinite(combined)):
        return 1e6

    if bandwidth > 0.0:
        z = np.clip((y - combined) / bandwidth, -500.0, 500.0)
        H = 1.0 / (1.0 + np.exp(z))
    else:
        H = (y < combined).astype(float)

    S = H.size
    psi1 = np.sum(H - alpha) / S
    psi2 = np.sum((H[1:] - alpha) * H[:-1]) / (S - 1) if S > 1 else 0.0
    return float(psi1**2 + psi2**2)


def default_bandwidth(y: np.ndarray) -> float:
    """h = sd(r) * S^(-1/3): satisfies h -> 0 and S*h -> infinity."""
    S = max(int(np.size(y)), 2)
    scale = float(np.std(y, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return scale * S ** (-1.0 / 3.0)


def fit_ccom(
    var_frame: pd.DataFrame,
    alpha: float = ALPHA,
    include_intercept: bool = False,
    sum_to_one: bool = False,
    bandwidth: float | None = None,
    method: str = "Nelder-Mead",
    extra_starts: list[np.ndarray] | None = None,
    maxiter: int = 20000,
    bandwidth_path: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125),
) -> CombinationResult:
    """Conditional Coverage Optimization Method (eq. 2.7).

    Minimizes psi' psi over the loading vector with ``scipy.optimize.minimize``,
    from several deterministic starting points to guard against the local-minimum
    problem the paper flags.

    Two refinements are needed because the smoothed surrogate is not the
    estimator:

    * **Bandwidth continuation.** The optimizer is run along a decreasing
      sequence of bandwidths, warm-starting each leg, following the paper's
      h_T -> 0 asymptotics.  A single coarse h leaves a wide flat basin in which
      many weight vectors look equally optimal.
    * **Selection on the discrete objective.** Eq. 2.7 defines the estimator as
      the argmin of psi' psi built from the *true* indicator; smoothing is only a
      numerical device.  Every candidate visited -- including each starting point,
      so that the single-model vectors e_j are always in the feasible set -- is
      therefore scored on the discrete objective, with the finest-bandwidth
      smoothed value as the tie-break.  This also makes CCOM no worse than the
      best stand-alone forecast by construction, since e_j is always feasible
      (it satisfies the sum-to-one restriction too).
    """
    panel, model_names, _ = prepare_panel(var_frame)
    X, y, dates = _design(panel)
    k = X.shape[1]
    spec = CombinationSpec(k, include_intercept, sum_to_one)

    if len(panel) <= spec.n_free:
        raise CombinationError(
            f"CCOM needs more observations ({len(panel)}) than parameters ({spec.n_free})"
        )

    h = default_bandwidth(y) if bandwidth is None else float(bandwidth)

    # Deterministic starts: equal weights, each model alone, and the CQOM solution.
    starts: list[np.ndarray] = [spec.pack(0.0, np.full(k, 1.0 / k))]
    for j in range(k):
        w = np.zeros(k)
        w[j] = 1.0
        starts.append(spec.pack(0.0, w))
    try:
        cq = fit_cqom(var_frame, alpha, include_intercept, sum_to_one)
        starts.append(spec.pack(cq.intercept, cq.weights))
    except CombinationError:
        pass
    if extra_starts:
        starts.extend(np.asarray(s, dtype=float) for s in extra_starts)

    options = (
        {"maxiter": maxiter, "xatol": 1e-10, "fatol": 1e-14}
        if method == "Nelder-Mead"
        else {"maxiter": maxiter}
    )
    h_fine = h * min(bandwidth_path)

    def discrete_objective(params: np.ndarray) -> float:
        intercept_, weights_ = spec.unpack(params)
        combined_ = combine(X, intercept_, weights_)
        if not np.all(np.isfinite(combined_)):
            return np.inf
        return hit_diagnostics(hit_sequence(y, combined_), alpha)["objective"]

    candidates: list[np.ndarray] = []
    any_success = False
    for x0 in starts:
        x0 = np.asarray(x0, dtype=float).ravel()
        if x0.size != spec.n_free:
            continue
        candidates.append(x0)          # the start itself is feasible
        current = x0
        for multiplier in bandwidth_path:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    _ccom_objective,
                    current,
                    args=(X, y, alpha, spec, h * multiplier),
                    method=method,
                    options=options,
                )
            any_success |= bool(res.success)
            if np.all(np.isfinite(res.x)):
                current = res.x
                candidates.append(np.asarray(res.x, dtype=float).copy())

    scored = [
        (discrete_objective(c), _ccom_objective(c, X, y, alpha, spec, h_fine), c)
        for c in candidates
    ]
    scored = [s for s in scored if np.isfinite(s[0])]
    if not scored:
        raise CombinationError("CCOM optimization produced no finite solution")
    best_discrete, best_smoothed, best_params = min(scored, key=lambda s: (s[0], s[1]))

    intercept, weights = spec.unpack(best_params)
    combined = combine(X, intercept, weights)
    # Diagnostics from the DISCRETE indicator, not the smoothed surrogate.
    diag = hit_diagnostics(hit_sequence(y, combined), alpha)
    diag["smoothed_objective"] = float(best_smoothed)
    diag["bandwidth"] = h
    diag["bandwidth_fine"] = h_fine
    diag["n_starts"] = float(len(starts))
    diag["n_candidates"] = float(len(scored))

    return CombinationResult(
        name="CCOM",
        spec=spec,
        model_names=model_names,
        intercept=intercept,
        weights=weights,
        var=pd.Series(combined, index=dates.to_numpy(), name="VaR_CCOM"),
        objective=float(diag["objective"]),
        converged=any_success,
        alpha=alpha,
        diagnostics=diag,
        notes=(
            f"method of moments, {method}, logistic smoothing "
            f"h={h:.3e}->{h_fine:.3e}, selected on discrete psi'psi"
        ),
    )


# --- Convenience ---------------------------------------------------------------
def fit_combinations(
    var_frame: pd.DataFrame,
    alpha: float = ALPHA,
    include_intercept: bool = False,
    sum_to_one: bool = False,
) -> dict[str, CombinationResult]:
    """Fit both combination methods on the same panel."""
    kwargs = dict(alpha=alpha, include_intercept=include_intercept, sum_to_one=sum_to_one)
    return {
        "CQOM": fit_cqom(var_frame, **kwargs),
        "CCOM": fit_ccom(var_frame, **kwargs),
    }


def combined_frame(
    var_frame: pd.DataFrame, results: dict[str, CombinationResult]
) -> pd.DataFrame:
    """Append combined VaR columns to the stand-alone panel."""
    panel, _, _ = prepare_panel(var_frame)
    out = panel.copy()
    for name, res in results.items():
        out[f"{VAR_COLUMN_PREFIX}{name}"] = res.var.to_numpy()
    return out


# --- Verification block --------------------------------------------------------
def _check(label: str, ok: bool) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def _report(res: CombinationResult) -> None:
    print(f"\n  {res.name}  ({res.notes})")
    print(f"    converged         : {res.converged}")
    w = res.weight_table()
    for idx, val in w.items():
        print(f"    weight[{idx:<9}] : {val:+.6f}")
    print(f"    sum of VaR weights: {res.weight_sum:+.6f}")
    d = res.diagnostics
    print(
        f"    hit rate          : {d['hit_rate']:.4%} "
        f"({int(d['n_hits'])}/{int(d['n_obs'])})  target {d['target']:.2%}"
    )
    print(f"    psi1 (coverage)   : {d['psi1']:+.6e}")
    print(f"    psi2 (independence): {d['psi2']:+.6e}")
    print(f"    psi'psi (discrete): {d['objective']:.6e}")
    if "smoothed_objective" in d:
        print(f"    psi'psi (smoothed): {d['smoothed_objective']:.6e}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit CQOM and CCOM VaR combinations.")
    # 800 returns -> 550 forecasts, comparable to the paper's S = 510 / 653.
    parser.add_argument("--test-days", type=int, default=800)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args(argv)

    print("=" * 72)
    print("Optimal VaR combinations: CCOM & CQOM - Halbleib & Pohlmeier (2012)")
    print("=" * 72)

    returns = load_returns()
    slice_df = returns.iloc[: args.test_days].reset_index(drop=True)
    models = default_models()

    print("\n[setup]")
    print(f"  test slice        : first {len(slice_df)} returns "
          f"({slice_df[DATE_COLUMN].iloc[0].date()} -> {slice_df[DATE_COLUMN].iloc[-1].date()})")
    print(f"  rolling window    : {args.window}")
    print(f"  alpha             : {args.alpha}")
    print(f"  stand-alone models: {', '.join(m.name for m in models)}")
    print(f"  expected S        : {len(slice_df) - args.window}")

    print("\n[stage 1] rolling stand-alone VaR forecasts")
    rolling = rolling_var_forecasts(
        slice_df, models=models, window=args.window, alpha=args.alpha, progress_every=150
    )
    panel, model_names, dropped = prepare_panel(rolling.var_frame)
    print(f"  forecasts         : {len(rolling.var_frame)} in {rolling.elapsed_seconds:.1f}s")
    print(f"  rows dropped (NaN): {dropped}")
    print(f"  usable S          : {len(panel)}")

    print("\n[stage 2] combinations as specified (no intercept, free weights)")
    results = fit_combinations(panel, alpha=args.alpha)
    for res in results.values():
        _report(res)

    print("\n[stage 3] paper-faithful variant (eq. 2.6: intercept + weights sum to 1)")
    paper = fit_combinations(panel, alpha=args.alpha, include_intercept=True, sum_to_one=True)
    for res in paper.values():
        _report(res)

    print("\n[stand-alone benchmark hit rates, same panel]")
    for col in [c for c in panel.columns if c.startswith(VAR_COLUMN_PREFIX)]:
        d = hit_diagnostics(
            hit_sequence(panel[ACTUAL_COLUMN].to_numpy(), panel[col].to_numpy()), args.alpha
        )
        print(f"  {col:<16} hit rate {d['hit_rate']:.4%}  "
              f"({int(d['n_hits'])}/{int(d['n_obs'])})  psi'psi {d['objective']:.3e}")

    out = combined_frame(panel, results)
    var_cols = [c for c in out.columns if c.startswith(VAR_COLUMN_PREFIX)]

    print("\n[combined VaR - last 5 rows]")
    disp = out[[DATE_COLUMN, ACTUAL_COLUMN, *var_cols]].copy()
    disp[DATE_COLUMN] = disp[DATE_COLUMN].dt.strftime("%Y-%m-%d")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(disp.tail(5).to_string(index=False, float_format=lambda v: f"{v: .6f}"))

    print("\n[same rows, percent]")
    pct = disp.copy()
    for c in [ACTUAL_COLUMN, *var_cols]:
        pct[c] = (pct[c] * 100).map(lambda v: f"{v:+.3f}%")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pct.tail(5).to_string(index=False))

    print("\n[small-slice degeneracy check: S = 50, as in Step 2]")
    small = panel.iloc[:50].reset_index(drop=True)
    try:
        s_cq = fit_cqom(small, alpha=args.alpha)
        s_diag = s_cq.diagnostics
        print(f"  CQOM on S=50 -> weights {np.round(s_cq.weights, 4)}, "
              f"hit rate {s_diag['hit_rate']:.2%} ({int(s_diag['n_hits'])} hits)")
        print("  NOTE: at alpha=0.01 a 50-obs panel has <1 expected violation, so the")
        print("        q=0.01 regression is pinned by a single point. Not interpretable.")
    except CombinationError as exc:
        print(f"  CQOM on S=50 failed as expected: {exc}")

    print("\n[sanity checks]")
    ok = True
    cq, cc = results["CQOM"], results["CCOM"]
    ok &= _check("CQOM converged", cq.converged)
    ok &= _check("CCOM converged", cc.converged)
    ok &= _check("CQOM weights all finite", bool(np.all(np.isfinite(cq.weights))))
    ok &= _check("CCOM weights all finite", bool(np.all(np.isfinite(cc.weights))))
    ok &= _check(
        "combined VaR series have no NaN",
        bool(out[var_cols].notna().all().all()),
    )
    ok &= _check(
        "all combined VaR values negative",
        bool((out[[f"{VAR_COLUMN_PREFIX}CQOM", f"{VAR_COLUMN_PREFIX}CCOM"]] < 0).all().all()),
    )
    ok &= _check(
        "combined VaR economically plausible (> -50%)",
        bool((out[var_cols] > -0.5).all().all()),
    )
    ok &= _check(
        "CQOM attains lower check loss than every stand-alone forecast",
        all(
            cq.objective <= _check_loss(
                panel[ACTUAL_COLUMN].to_numpy(), panel[c].to_numpy(), args.alpha
            ) + 1e-12
            for c in [col for col in panel.columns if col.startswith(VAR_COLUMN_PREFIX)]
        ),
    )
    ok &= _check(
        "CCOM attains psi'psi no worse than every stand-alone forecast",
        all(
            cc.objective <= hit_diagnostics(
                hit_sequence(panel[ACTUAL_COLUMN].to_numpy(), panel[c].to_numpy()), args.alpha
            )["objective"] + 1e-12
            for c in [col for col in panel.columns if col.startswith(VAR_COLUMN_PREFIX)]
        ),
    )
    ok &= _check(
        "paper-faithful CCOM/CQOM weights sum to 1 under eq. 2.6",
        bool(
            np.isclose(paper["CQOM"].weight_sum, 1.0)
            and np.isclose(paper["CCOM"].weight_sum, 1.0)
        ),
    )
    ok &= _check(
        "hit_sequence matches an independent recomputation",
        bool(
            np.array_equal(
                hit_sequence(panel[ACTUAL_COLUMN].to_numpy(), cq.var.to_numpy()),
                (panel[ACTUAL_COLUMN].to_numpy() < cq.var.to_numpy()).astype(float),
            )
        ),
    )

    print("\n" + ("All checks passed." if ok else "SOME CHECKS FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
