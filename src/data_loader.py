"""Data pipeline for the VaR backtesting engine.

Replicates the data stage of Halbleib & Pohlmeier (2012), "Improving the value
at risk forecasts: Theory and evidence from the financial crisis", Journal of
Economic Dynamics & Control 36, 1212-1228.

Methodology notes taken from the paper (Section 3.1, p. 1216):
  * The authors use "daily log-returns computed from closing dividend and split
    adjusted prices".  We therefore request dividend/split-adjusted closes from
    yfinance (``auto_adjust=True``), not the raw close.
  * Returns are continuously compounded:

        r_t = ln(P_t / P_{t-1})

  * The paper evaluates VaR at p = 0.01 on rolling/recursive windows of up to
    1000 observations, so the series must carry enough pre-crisis history to
    fill the longest estimation window before the first evaluation date.

Scale convention: this module stores *raw* log-returns (e.g. -0.0712), matching
the formula above.  Downstream GARCH/FIGARCH estimation is better conditioned
on percentage returns (100 * r_t); scale at the model layer rather than here, so
that a single canonical return series is kept on disk.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# --- Paper replication defaults ------------------------------------------------
TICKER = "^GSPC"            # S&P 500 index
START_DATE = "2000-01-01"
END_DATE = "2010-12-31"     # inclusive

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "sp500_returns.csv"

TRADING_DAYS_PER_YEAR = 252
# Calendar days of extra history fetched before START_DATE so that the first
# requested trading day has a valid predecessor price (and hence a return).
PAD_CALENDAR_DAYS = 10
# |r_t| above this is not a real move in a broad equity index; it signals a bad
# price (unadjusted split, bad tick).  The worst S&P 500 day in 2000-2010 is
# about -9.5% (2008-10-15), so this threshold only catches data errors.
MAX_PLAUSIBLE_ABS_RETURN = 0.5

DATE_COLUMN = "date"
RETURN_COLUMN = "log_return"


class DataLoaderError(RuntimeError):
    """Raised when the pipeline cannot produce a usable return series."""


@dataclass
class CleaningReport:
    """Bookkeeping for every row dropped or flagged while cleaning."""

    raw_price_rows: int = 0
    dropped_duplicate_dates: int = 0
    dropped_missing_prices: int = 0
    dropped_nonpositive_prices: int = 0
    dropped_first_observation: int = 0
    dropped_outside_window: int = 0
    zero_returns: int = 0
    implausible_returns: int = 0
    final_rows: int = 0

    def lines(self) -> list[str]:
        return [
            f"raw price rows fetched        : {self.raw_price_rows}",
            f"dropped duplicate dates       : {self.dropped_duplicate_dates}",
            f"dropped missing prices        : {self.dropped_missing_prices}",
            f"dropped non-positive prices   : {self.dropped_nonpositive_prices}",
            f"dropped first obs (no P_t-1)  : {self.dropped_first_observation}",
            f"dropped outside date window   : {self.dropped_outside_window}",
            f"zero returns retained (flag)  : {self.zero_returns}",
            f"implausible returns retained  : {self.implausible_returns}",
            f"final return observations     : {self.final_rows}",
        ]


# --- Fetch ---------------------------------------------------------------------
def _download_raw(
    ticker: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    max_retries: int = 3,
    retry_wait_seconds: float = 2.0,
) -> pd.DataFrame:
    """Call ``yf.download`` with retries, returning the raw frame."""
    kwargs = {
        "start": start.strftime("%Y-%m-%d"),
        "end": end_exclusive.strftime("%Y-%m-%d"),
        "auto_adjust": True,   # dividend- and split-adjusted closes, per the paper
        "progress": False,
        "threads": False,
    }
    # Newer yfinance returns MultiIndex columns even for a single ticker.
    if "multi_level_index" in inspect.signature(yf.download).parameters:
        kwargs["multi_level_index"] = False

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        raw = None
        try:
            raw = yf.download(ticker, **kwargs)
        except Exception as exc:  # network/parsing failures from the provider
            last_error = exc
        if raw is not None and not raw.empty:
            return raw
        if attempt < max_retries:
            time.sleep(retry_wait_seconds)

    detail = f" Last error: {last_error!r}" if last_error else ""
    raise DataLoaderError(
        f"yfinance returned no data for {ticker} between {kwargs['start']} and "
        f"{kwargs['end']} after {max_retries} attempts.{detail}"
    )


def _extract_close(raw: pd.DataFrame, ticker: str) -> pd.Series:
    """Pull the adjusted close column out of whatever shape yfinance returned."""
    frame = raw
    if isinstance(frame.columns, pd.MultiIndex):
        for name in ("Close", "Adj Close"):
            for level in range(frame.columns.nlevels):
                if name in frame.columns.get_level_values(level):
                    frame = frame.xs(name, axis=1, level=level, drop_level=True)
                    break
            else:
                continue
            break
        else:
            raise DataLoaderError(
                f"No close column in yfinance response; columns={list(raw.columns)}"
            )
        if isinstance(frame, pd.DataFrame):
            if ticker in frame.columns:
                frame = frame[ticker]
            elif frame.shape[1] == 1:
                frame = frame.iloc[:, 0]
            else:
                raise DataLoaderError(
                    f"Ambiguous close columns for {ticker}: {list(frame.columns)}"
                )
        return frame.astype("float64").rename("close")

    for name in ("Close", "Adj Close"):
        if name in frame.columns:
            return frame[name].astype("float64").rename("close")
    raise DataLoaderError(
        f"No close column in yfinance response; columns={list(frame.columns)}"
    )


def _normalize_index(prices: pd.Series) -> pd.Series:
    """Coerce the index to tz-naive, time-stripped daily timestamps."""
    index = pd.to_datetime(prices.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    out = prices.copy()
    out.index = index.normalize()
    out.index.name = DATE_COLUMN
    return out


def fetch_prices(
    ticker: str = TICKER,
    start: str = START_DATE,
    end: str = END_DATE,
    pad_calendar_days: int = PAD_CALENDAR_DAYS,
) -> pd.Series:
    """Fetch adjusted closing prices for ``ticker`` over ``[start, end]``.

    ``end`` is treated as inclusive (yfinance's own ``end`` is exclusive, so one
    day is added internally).  ``pad_calendar_days`` of history is fetched
    before ``start`` so the first in-window trading day still has a predecessor
    price, and therefore a return.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise DataLoaderError(f"start ({start}) is after end ({end}).")

    fetch_start = start_ts - pd.Timedelta(days=max(pad_calendar_days, 0))
    raw = _download_raw(ticker, fetch_start, end_ts + pd.Timedelta(days=1))
    prices = _normalize_index(_extract_close(raw, ticker))
    return prices.sort_index()


# --- Transform & clean ---------------------------------------------------------
def build_returns(
    prices: pd.Series,
    start: str = START_DATE,
    end: str = END_DATE,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Turn a price series into a cleaned daily log-return frame."""
    report = CleaningReport(raw_price_rows=int(prices.shape[0]))

    before = prices.shape[0]
    prices = prices[~prices.index.duplicated(keep="first")]
    report.dropped_duplicate_dates = before - prices.shape[0]

    before = prices.shape[0]
    prices = prices.dropna()
    report.dropped_missing_prices = before - prices.shape[0]

    before = prices.shape[0]
    prices = prices[prices > 0]  # ln() undefined otherwise
    report.dropped_nonpositive_prices = before - prices.shape[0]

    if prices.shape[0] < 2:
        raise DataLoaderError(
            f"Only {prices.shape[0]} usable price(s) after cleaning; "
            "cannot compute returns."
        )

    returns = np.log(prices / prices.shift(1)).rename(RETURN_COLUMN)
    before = returns.shape[0]
    returns = returns.dropna()
    report.dropped_first_observation = before - returns.shape[0]

    before = returns.shape[0]
    returns = returns.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    report.dropped_outside_window = before - returns.shape[0]

    if returns.empty:
        raise DataLoaderError(
            f"No returns remain inside the requested window {start}..{end}."
        )

    # Flag (do not drop) values that look suspicious but could be genuine.
    report.zero_returns = int((returns == 0.0).sum())
    report.implausible_returns = int((returns.abs() > MAX_PLAUSIBLE_ABS_RETURN).sum())
    report.final_rows = int(returns.shape[0])

    frame = returns.to_frame().reset_index()
    frame.columns = [DATE_COLUMN, RETURN_COLUMN]
    return frame, report


def build_dataset(
    ticker: str = TICKER,
    start: str = START_DATE,
    end: str = END_DATE,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Fetch, clean and return the log-return dataset."""
    prices = fetch_prices(ticker=ticker, start=start, end=end)
    return build_returns(prices, start=start, end=end)


# --- Persistence ---------------------------------------------------------------
def save_returns(frame: pd.DataFrame, path: Path = OUTPUT_PATH) -> Path:
    """Write the return frame to CSV with ISO dates and full float precision."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    return path


def load_returns(path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Read a previously saved return file back into a frame (for later steps)."""
    path = Path(path)
    if not path.exists():
        raise DataLoaderError(
            f"{path} not found - run `python src/data_loader.py` first."
        )
    frame = pd.read_csv(path, parse_dates=[DATE_COLUMN])
    missing = {DATE_COLUMN, RETURN_COLUMN} - set(frame.columns)
    if missing:
        raise DataLoaderError(f"{path} is missing column(s): {sorted(missing)}")
    return frame.sort_values(DATE_COLUMN).reset_index(drop=True)


# --- Summary statistics --------------------------------------------------------
def summarize(frame: pd.DataFrame) -> dict[str, object]:
    """Compute the summary statistics printed by the verification block."""
    returns = frame[RETURN_COLUMN].astype("float64")
    dates = pd.to_datetime(frame[DATE_COLUMN])
    daily_std = float(returns.std(ddof=1))
    return {
        "rows": int(returns.shape[0]),
        "start": dates.min().date().isoformat(),
        "end": dates.max().date().isoformat(),
        "mean_daily": float(returns.mean()),
        "std_daily": daily_std,
        "annualized_mean": float(returns.mean()) * TRADING_DAYS_PER_YEAR,
        "annualized_vol": daily_std * np.sqrt(TRADING_DAYS_PER_YEAR),
        "skewness": float(returns.skew()),
        "excess_kurtosis": float(returns.kurt()),  # pandas kurt() is excess
        "min": float(returns.min()),
        "min_date": dates.loc[returns.idxmin()].date().isoformat(),
        "max": float(returns.max()),
        "max_date": dates.loc[returns.idxmax()].date().isoformat(),
        "empirical_q01": float(returns.quantile(0.01)),
        "nan_count": int(returns.isna().sum()),
    }


def _print_summary(stats: dict[str, object], report: CleaningReport, path: Path) -> None:
    print("=" * 68)
    print("S&P 500 (^GSPC) daily log-returns - Halbleib & Pohlmeier (2012)")
    print("=" * 68)

    print("\n[cleaning]")
    for line in report.lines():
        print(f"  {line}")

    print("\n[summary statistics]")
    print(f"  observations               : {stats['rows']}")
    print(f"  date range                 : {stats['start']} -> {stats['end']}")
    print(f"  mean daily return          : {stats['mean_daily']:+.6f}")
    print(f"  daily volatility (sd)      : {stats['std_daily']:.6f}")
    print(f"  annualized mean (252d)     : {stats['annualized_mean']:+.4%}")
    print(f"  annualized volatility      : {stats['annualized_vol']:.4%}")
    print(f"  skewness                   : {stats['skewness']:+.4f}")
    print(f"  excess kurtosis            : {stats['excess_kurtosis']:+.4f}")
    print(f"  min return                 : {stats['min']:+.4%} on {stats['min_date']}")
    print(f"  max return                 : {stats['max']:+.4%} on {stats['max_date']}")
    print(f"  empirical 1% quantile      : {stats['empirical_q01']:+.4%}")
    print(f"  remaining NaNs             : {stats['nan_count']}")

    print("\n[output]")
    print(f"  written to                 : {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and clean daily S&P 500 log-returns for VaR backtesting."
    )
    parser.add_argument("--ticker", default=TICKER)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE, help="inclusive end date")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args(argv)

    try:
        frame, report = build_dataset(ticker=args.ticker, start=args.start, end=args.end)
    except DataLoaderError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    path = save_returns(frame, Path(args.output))
    _print_summary(summarize(frame), report, path)

    print("\n[head]")
    print(frame.head(3).to_string(index=False))
    print("\n[tail]")
    print(frame.tail(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
