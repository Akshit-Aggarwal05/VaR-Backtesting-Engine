"""Stand-alone VaR models for the backtesting engine.

Implements the three benchmark ("stand-alone") VaR approaches from Halbleib &
Pohlmeier (2012), "Improving the value at risk forecasts: Theory and evidence
from the financial crisis", Journal of Economic Dynamics & Control 36,
1212-1228.

Methodology, per the paper:

  * VaR definition (eq. 2.2, p. 1214) -- a *location-scale* forecast:

        VaR_{t+1|t}(p) = mu_{t+1|t} + Q_p(Z) * sigma_{t+1|t}

    where Q_p(Z) is the p-th quantile of the standardized innovations z_t.
    Returned values are therefore negative for a long position.

  * Parametric specification (Section 3.1, p. 1216): conditional mean is an
    "ARMA(1,0) model with intercept" (i.e. AR(1) + constant); conditional
    variance is GARCH(1,1).  Innovations are Normal (ND) or Student-t (SD),
    with the degrees of freedom estimated jointly by ML.

  * Historical Simulation (Section 3.1, p. 1216): "estimating VaR simply by the
    sample quantile of a rolling window of historical data", with windows
    "mostly set to be between 250 and 750 observations".

  * Evaluation level is p = 0.01, "in line with the Basel II requirements".

Two implementation details that materially affect the numbers:

  1. ``Q_p(Z)`` must be the quantile of the *standardized* (unit-variance)
     innovation, not of the raw Student-t.  For nu = 8 these differ by ~15%
     (-2.508 vs -2.896).  We use ``arch``'s own ``distribution.ppf``, which
     applies the sqrt((nu-2)/nu) scaling internally.

  2. GARCH quasi-ML is poorly conditioned on raw log-returns (~1e-2), so fits
     are performed on percentage returns (100 * r) and the resulting VaR is
     divided by 100.  VaR is location-scale equivariant, so this is exact, not
     an approximation.

No-lookahead contract: the forecast recorded against date d_t is built *only*
from returns strictly before d_t.  See ``rolling_var_forecasts``.
"""

from __future__ import annotations

import sys
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

try:  # support both `python src/var_models.py` and `import src.var_models`
    from data_loader import DATE_COLUMN, RETURN_COLUMN, load_returns
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_loader import DATE_COLUMN, RETURN_COLUMN, load_returns

# --- Paper replication defaults ------------------------------------------------
ALPHA = 0.01          # VaR probability level p, per Basel II
WINDOW = 250          # rolling estimation window (paper uses 250/500/750/1000)
GARCH_SCALE = 100.0   # fit on percentage returns for numerical conditioning

VAR_COLUMN_PREFIX = "VaR_"
ACTUAL_COLUMN = "actual_return"


class VaRModelError(RuntimeError):
    """Raised when a VaR model cannot produce a forecast at all."""


@dataclass
class VaRForecast:
    """A single one-step-ahead VaR forecast plus its decomposition."""

    var: float
    mu: float = np.nan
    sigma: float = np.nan
    quantile: float = np.nan          # Q_p(Z), the standardized innovation quantile
    converged: bool = True
    extra: dict[str, float] = field(default_factory=dict)


# --- Model interface -----------------------------------------------------------
class VaRModel(ABC):
    """Common interface so the backtester can treat every model identically."""

    name: str = "base"

    @abstractmethod
    def forecast(self, window: np.ndarray, alpha: float = ALPHA) -> VaRForecast:
        """One-step-ahead VaR from an estimation ``window`` of past returns."""

    @property
    def column(self) -> str:
        return f"{VAR_COLUMN_PREFIX}{self.name}"

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(name={self.name!r})"


class HistoricalSimulation(VaRModel):
    """Non-parametric VaR: the empirical alpha-quantile of the rolling window.

    Carries no conditional mean or volatility model -- as the paper notes, it
    "ignores the conditional dependencies among returns as well as the relevant
    information on extreme past events that situate outside the sampling
    window".
    """

    name = "HS"

    def __init__(self, interpolation: str = "linear") -> None:
        self.interpolation = interpolation

    def forecast(self, window: np.ndarray, alpha: float = ALPHA) -> VaRForecast:
        var = float(np.quantile(window, alpha, method=self.interpolation))
        return VaRForecast(
            var=var,
            mu=float(np.mean(window)),
            sigma=float(np.std(window, ddof=1)),
            quantile=np.nan,            # not a location-scale model
            converged=True,
            extra={"n_obs": float(window.size)},
        )


class GARCHVaR(VaRModel):
    """AR(1)-GARCH(1,1) VaR with Normal or Student-t innovations.

    Matches the paper's parametric specification: ARMA(1,0) conditional mean
    with intercept, GARCH(1,1) conditional variance, and the quantile taken
    from the assumed standardized innovation distribution.
    """

    def __init__(
        self,
        dist: str = "normal",
        name: str | None = None,
        mean: str = "AR",
        lags: int = 1,
        scale: float = GARCH_SCALE,
    ) -> None:
        self.dist = dist
        self.mean = mean
        self.lags = lags
        self.scale = scale
        self.name = name or {"normal": "GARCH-N", "t": "GARCH-t"}.get(dist, f"GARCH-{dist}")

    def _fit(self, window: np.ndarray):
        model = arch_model(
            window * self.scale,       # percentage returns: better conditioned
            mean=self.mean,
            lags=self.lags,
            vol="GARCH",
            p=1,
            q=1,
            dist=self.dist,
            rescale=False,             # we control scaling explicitly
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.fit(disp="off", show_warning=False, update_freq=0)

    def forecast(self, window: np.ndarray, alpha: float = ALPHA) -> VaRForecast:
        try:
            res = self._fit(window)
            fc = res.forecast(horizon=1, reindex=False)
            mu_s = float(fc.mean.iloc[-1, 0])
            var_s = float(fc.variance.iloc[-1, 0])
        except Exception:  # optimizer blow-up on a pathological window
            return VaRForecast(var=np.nan, converged=False)

        if not np.isfinite(mu_s) or not np.isfinite(var_s) or var_s <= 0.0:
            return VaRForecast(var=np.nan, converged=False)

        sigma_s = float(np.sqrt(var_s))

        # Q_p(Z) from the *standardized* innovation distribution.
        n_dist = res.model.distribution.num_params
        dist_params = np.asarray(res.params, dtype=float)[-n_dist:] if n_dist else None
        q = float(np.asarray(res.model.distribution.ppf(alpha, dist_params)).ravel()[0])

        # Undo the scaling: VaR is location-scale equivariant.
        var_value = (mu_s + sigma_s * q) / self.scale

        extra: dict[str, float] = {}
        for key in ("nu", "alpha[1]", "beta[1]", "omega"):
            if key in res.params.index:
                extra[key] = float(res.params[key])
        if {"alpha[1]", "beta[1]"} <= set(res.params.index):
            extra["persistence"] = extra["alpha[1]"] + extra["beta[1]"]

        return VaRForecast(
            var=var_value,
            mu=mu_s / self.scale,
            sigma=sigma_s / self.scale,
            quantile=q,
            converged=bool(res.convergence_flag == 0),
            extra=extra,
        )


def default_models() -> list[VaRModel]:
    """The three stand-alone models required for this stage."""
    return [
        HistoricalSimulation(),
        GARCHVaR(dist="normal"),
        GARCHVaR(dist="t"),
    ]


# --- Rolling engine ------------------------------------------------------------
@dataclass
class RollingVaRResult:
    """Output of a rolling VaR run."""

    var_frame: pd.DataFrame        # date, actual_return, one VaR column per model
    diagnostics: pd.DataFrame      # per-model mu/sigma/quantile/params per date
    window: int
    alpha: float
    elapsed_seconds: float = 0.0

    @property
    def model_columns(self) -> list[str]:
        return [c for c in self.var_frame.columns if c.startswith(VAR_COLUMN_PREFIX)]

    def failure_counts(self) -> dict[str, int]:
        return {c: int(self.var_frame[c].isna().sum()) for c in self.model_columns}

    def hit_rates(self) -> dict[str, float]:
        """Realized violation rate -- a first sanity check, not a formal test.

        Formal unconditional/conditional coverage testing is a later stage.
        """
        actual = self.var_frame[ACTUAL_COLUMN]
        rates: dict[str, float] = {}
        for col in self.model_columns:
            valid = self.var_frame[col].notna()
            if int(valid.sum()) == 0:
                rates[col] = np.nan
                continue
            breaches = (actual[valid] < self.var_frame[col][valid]).sum()
            rates[col] = float(breaches) / float(valid.sum())
        return rates


def rolling_var_forecasts(
    returns: pd.DataFrame | pd.Series,
    models: list[VaRModel] | None = None,
    window: int = WINDOW,
    alpha: float = ALPHA,
    progress_every: int = 0,
) -> RollingVaRResult:
    """Roll a fixed-width window through ``returns`` and forecast VaR each day.

    No-lookahead contract: for each target date ``d_t`` the estimation sample is
    ``r[t - window : t]`` -- it ends at ``t - 1`` and never includes ``r_t``
    itself.  The realized return ``r_t`` is carried alongside purely so the
    backtesting stage can compare forecast against outcome.

    With ``n`` returns this yields exactly ``n - window`` forecasts.
    """
    models = models or default_models()

    if isinstance(returns, pd.DataFrame):
        if RETURN_COLUMN not in returns.columns:
            raise VaRModelError(f"expected a '{RETURN_COLUMN}' column")
        dates = pd.to_datetime(returns[DATE_COLUMN]).to_numpy()
        values = returns[RETURN_COLUMN].to_numpy(dtype=float)
    else:
        dates = pd.to_datetime(returns.index).to_numpy()
        values = returns.to_numpy(dtype=float)

    n = values.size
    if window < 2:
        raise VaRModelError(f"window must be >= 2, got {window}")
    if n <= window:
        raise VaRModelError(
            f"need more than {window} returns to produce a forecast, got {n}"
        )
    if not np.isfinite(values).all():
        raise VaRModelError("returns contain NaN/inf; clean the data first")

    rows: list[dict[str, object]] = []
    diag_rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for t in range(window, n):
        train = values[t - window : t]          # r_{t-window} .. r_{t-1}
        row: dict[str, object] = {
            DATE_COLUMN: dates[t],
            ACTUAL_COLUMN: float(values[t]),
        }
        for model in models:
            fc = model.forecast(train, alpha=alpha)
            row[model.column] = fc.var
            diag_rows.append(
                {
                    DATE_COLUMN: dates[t],
                    "model": model.name,
                    "var": fc.var,
                    "mu": fc.mu,
                    "sigma": fc.sigma,
                    "quantile": fc.quantile,
                    "converged": fc.converged,
                    **fc.extra,
                }
            )
        rows.append(row)

        if progress_every and (len(rows) % progress_every == 0 or t == n - 1):
            done, total = len(rows), n - window
            print(
                f"  .. {done}/{total} forecasts "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )

    var_frame = pd.DataFrame(rows)
    var_frame[DATE_COLUMN] = pd.to_datetime(var_frame[DATE_COLUMN])
    return RollingVaRResult(
        var_frame=var_frame,
        diagnostics=pd.DataFrame(diag_rows),
        window=window,
        alpha=alpha,
        elapsed_seconds=time.perf_counter() - started,
    )


# --- Verification block --------------------------------------------------------
def _check(label: str, ok: bool) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def main() -> int:
    test_days = 300

    print("=" * 70)
    print("Stand-alone VaR models - Halbleib & Pohlmeier (2012)")
    print("=" * 70)

    returns = load_returns()
    slice_df = returns.iloc[:test_days].reset_index(drop=True)
    models = default_models()

    print(f"\n[setup]")
    print(f"  full sample          : {len(returns)} returns")
    print(f"  test slice           : first {len(slice_df)} returns "
          f"({slice_df[DATE_COLUMN].iloc[0].date()} -> "
          f"{slice_df[DATE_COLUMN].iloc[-1].date()})")
    print(f"  rolling window       : {WINDOW}")
    print(f"  VaR level alpha      : {ALPHA}")
    print(f"  expected forecasts   : {len(slice_df) - WINDOW}")
    print(f"  models               : {', '.join(m.name for m in models)}")

    print("\n[fitting]")
    result = rolling_var_forecasts(
        slice_df, models=models, window=WINDOW, alpha=ALPHA, progress_every=25
    )
    var_frame = result.var_frame
    cols = result.model_columns

    print(f"\n[forecasts] {len(var_frame)} rows in {result.elapsed_seconds:.1f}s")
    print("\nLast 5 rows of VaR forecasts (1-day, 99%):")
    display = var_frame.copy()
    display[DATE_COLUMN] = display[DATE_COLUMN].dt.strftime("%Y-%m-%d")
    with pd.option_context("display.float_format", lambda v: f"{v: .6f}"):
        print(display.tail(5).to_string(index=False))

    print("\nSame rows in percent:")
    pct = display.copy()
    for c in [ACTUAL_COLUMN, *cols]:
        pct[c] = (pct[c] * 100).map(lambda v: f"{v:+.3f}%")
    print(pct.tail(5).to_string(index=False))

    print("\n[per-model summary]")
    hit = result.hit_rates()
    fails = result.failure_counts()
    header = f"  {'model':<16}{'mean VaR':>12}{'min':>12}{'max':>12}{'hits':>9}{'n/a':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in cols:
        s = var_frame[c]
        print(
            f"  {c:<16}{s.mean():>12.5f}{s.min():>12.5f}{s.max():>12.5f}"
            f"{hit[c]:>8.2%}{fails[c]:>6d}"
        )

    print("\n[estimated GARCH parameters, last window]")
    diag = result.diagnostics
    last_date = diag[DATE_COLUMN].max()
    for _, r in diag[diag[DATE_COLUMN] == last_date].iterrows():
        if not np.isfinite(r.get("persistence", np.nan)):
            continue
        nu = r.get("nu", np.nan)
        nu_txt = f", nu={nu:.2f}" if np.isfinite(nu) else ""
        print(
            f"  {r['model']:<10} sigma={r['sigma']:.5f}  Q_p(Z)={r['quantile']:+.4f}  "
            f"alpha+beta={r['persistence']:.4f}{nu_txt}"
        )

    print("\n[sanity checks]")
    ok = True
    ok &= _check(
        f"produced exactly {len(slice_df) - WINDOW} forecasts",
        len(var_frame) == len(slice_df) - WINDOW,
    )
    ok &= _check("at least 50 forecasts", len(var_frame) >= 50)
    ok &= _check("no failed fits", sum(fails.values()) == 0)
    ok &= _check(
        "all VaR values negative (long-position loss quantile)",
        bool((var_frame[cols] < 0).all().all()),
    )
    ok &= _check(
        "all VaR values economically plausible (> -50%)",
        bool((var_frame[cols] > -0.5).all().all()),
    )
    # Fatter tails must give a more conservative quantile at the same sigma.
    q_n = diag[(diag["model"] == "GARCH-N")]["quantile"].dropna()
    q_t = diag[(diag["model"] == "GARCH-t")]["quantile"].dropna()
    ok &= _check(
        "Student-t quantile is more extreme than Normal (fat tails)",
        bool((q_t.to_numpy() < q_n.to_numpy()).all()),
    )
    # No-lookahead: the first forecast must only use the first `window` returns.
    first_hs = float(var_frame[HistoricalSimulation().column].iloc[0])
    expected_hs = float(np.quantile(
        slice_df[RETURN_COLUMN].to_numpy()[:WINDOW], ALPHA
    ))
    ok &= _check(
        "HS forecast reproducible from pre-window data only (no lookahead)",
        np.isclose(first_hs, expected_hs),
    )
    ok &= _check(
        "forecast dates align with returns one step ahead of each window",
        var_frame[DATE_COLUMN].iloc[0] == slice_df[DATE_COLUMN].iloc[WINDOW],
    )

    print("\n" + ("All checks passed." if ok else "SOME CHECKS FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
