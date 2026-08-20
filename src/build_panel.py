"""
build_panel.py — Align the raw CBU / FRED / BOP series onto one monthly panel.

Frequencies coming in:
  CBU FX          monthly (already sampled at month-start)
  FRED RUB/USD    monthly
  FRED Brent      daily  -> monthly mean
  CBU BOP         quarterly flows (analytical xlsx, already discrete;
                  PDF/DOCX fallback is YTD-differenced in data_pull)
                  -> upsample to monthly here

Upsampling the remittance series is not a free lunch. The default is a step
function (hold the quarterly flow constant across its three months) because
that copies information the data actually contains. A cubic spline looks
smoother and will change ADF/KPSS statistics downstream; it is available
behind a flag, not as the default, for exactly that reason.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cbu_bop import ANALYTICAL_XLSX_NAME
from data_pull import FRED_SERIES, RAW_DIR

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Official USD/UZS jumped from ~4,000 to ~8,100 at the Sept 2017 unification
# and has since traded roughly 8,000–13,000. Anything outside this band is
# more likely a unit error than a real observation.
USD_UZS_RANGE = (1_000.0, 25_000.0)
# Secondary income credit has run roughly $1–15bn a quarter in the 2020s.
# Monthly values equal the quarterly flow under the default step upsample.
REMITTANCES_MN_USD_RANGE = (0.0, 25_000.0)


def upsample_quarterly_to_monthly(
    quarterly: pd.Series,
    method: str = "step",
) -> pd.Series:
    """Expand a quarterly series to month-start frequency.

    `method="step"` (default) holds each quarter's value constant across
    its three months. That is the conservative choice for unit-root tests:
    it does not invent intra-quarter dynamics the CBU did not publish.

    `method="spline"` fits a cubic spline through the quarter-start knots.
    Months strictly between knots are interpolated; the two months after
    the last knot (Nov–Dec of a Q4 observation, for example) cannot be
    interpolated without extrapolation, so they are ffilled from the last
    knot and flagged on stdout. Requires at least four non-NaN quarters.
    """
    if method not in {"step", "spline"}:
        raise ValueError(f"method must be 'step' or 'spline', got {method!r}")
    if quarterly.empty:
        raise ValueError("upsample_quarterly_to_monthly() got an empty series.")

    s = quarterly.copy()
    s = pd.to_numeric(s, errors="coerce")
    if isinstance(s.index, pd.PeriodIndex):
        s.index = s.index.to_timestamp(how="start")
    s.index = pd.to_datetime(s.index).to_period("Q").to_timestamp(how="start")
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]

    start = s.index.min()
    # Last quarter-start covers that quarter's remaining two months.
    end = s.index.max() + pd.offsets.MonthBegin(2)
    monthly_idx = pd.date_range(start=start, end=end, freq="MS")

    if method == "step":
        out = s.reindex(monthly_idx).ffill()
        print(
            f"[panel] remittances upsampled {s.index.min().date()}–"
            f"{end.date()} by step (quarterly value held for 3 months)"
        )
        return out.rename(quarterly.name)

    known = s.dropna()
    if len(known) < 4:
        raise ValueError(
            f"cubic spline needs at least 4 non-NaN quarters, got {len(known)}. "
            "Pass method='step' or add more CBU releases. Not falling back "
            "silently — a 3-point 'spline' would just be making numbers up."
        )

    aligned = s.reindex(monthly_idx)
    interpolated = aligned.interpolate(method="cubic", limit_area="inside")
    trailing = interpolated.isna() & (interpolated.index > known.index.max())
    if trailing.any():
        print(
            f"[panel] WARNING: spline cannot interpolate "
            f"{int(trailing.sum())} month(s) after the last quarter-start "
            f"knot ({known.index.max().date()}) without extrapolating. "
            "Holding the last knot value (same as step) for those months "
            "rather than inventing a cubic tail."
        )
        interpolated = interpolated.ffill()
    leading = interpolated.isna() & (interpolated.index < known.index.min())
    if leading.any():
        print(
            f"[panel] WARNING: {int(leading.sum())} month(s) before the first "
            "knot left as NaN — spline will not back-extrapolate."
        )
    print(
        f"[panel] remittances upsampled {s.index.min().date()}–{end.date()} "
        "by cubic spline (intra-quarter values are interpolated, not observed)"
    )
    return interpolated.rename(quarterly.name)


def align_to_common_monthly(*series: pd.Series) -> pd.DataFrame:
    """Inner-join several monthly series onto one complete MS DatetimeIndex.

    Window is the intersection of each series' observed (non-NaN) range —
    earliest date where ALL sources have data through the latest such date.
    Interior NaNs are kept and returned; they are not dropped or filled.
    """
    if not series:
        raise ValueError("align_to_common_monthly() needs at least one series.")

    cleaned: list[pd.Series] = []
    bounds: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for s in series:
        x = s.copy()
        x.index = pd.to_datetime(x.index).to_period("M").to_timestamp(how="start")
        x = x.sort_index()
        x = x[~x.index.duplicated(keep="last")]
        observed = x.dropna()
        if observed.empty:
            raise ValueError(f"series {x.name!r} is entirely NaN; cannot align.")
        bounds.append((str(x.name), observed.index.min(), observed.index.max()))
        cleaned.append(x)

    start = max(b[1] for b in bounds)
    end = min(b[2] for b in bounds)
    if start > end:
        detail = ", ".join(f"{n}: {a.date()}–{b.date()}" for n, a, b in bounds)
        raise ValueError(f"series have no overlapping dates ({detail})")

    idx = pd.date_range(start=start, end=end, freq="MS")
    print(f"[panel] common window {start.date()} to {end.date()} ({len(idx)} months)")
    for name, a, b in bounds:
        print(f"[panel]   {name}: {a.date()} to {b.date()}")

    return pd.concat([s.reindex(idx) for s in cleaned], axis=1)


def sanity_check(panel: pd.DataFrame) -> None:
    """Print (do not silently fix) anything that would corrupt later tests."""
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise TypeError("panel index must be a DatetimeIndex")

    expected = pd.date_range(panel.index.min(), panel.index.max(), freq="MS")
    missing = expected.difference(panel.index)
    if len(panel.index) != len(expected) or not panel.index.equals(expected):
        print(
            f"[panel] WARNING: index is not a complete monthly MS range "
            f"{expected.min().date()}–{expected.max().date()}. "
            f"Missing {len(missing)} month(s): "
            f"{[d.date().isoformat() for d in missing[:12]]}"
            f"{'...' if len(missing) > 12 else ''}"
        )
    else:
        print(f"[panel] index OK: {len(panel)} consecutive month-starts, no gaps")

    nan_total = int(panel.isna().sum().sum())
    if nan_total:
        nan_rows = panel.index[panel.isna().any(axis=1)]
        print(
            f"[panel] WARNING: {nan_total} NaN(s) after merge across "
            f"{len(nan_rows)} month(s). NOT dropping them. Dates: "
            f"{[d.date().isoformat() for d in nan_rows[:20]]}"
            f"{'...' if len(nan_rows) > 20 else ''}"
        )
        print(f"[panel] NaNs by column:\n{panel.isna().sum().to_string()}")
    else:
        print("[panel] no NaNs after merge")

    _flag_range(panel, "usd_uzs", USD_UZS_RANGE, "USD/UZS")
    if "remittances_mn_usd" in panel.columns:
        _flag_range(
            panel,
            "remittances_mn_usd",
            REMITTANCES_MN_USD_RANGE,
            "remittances (mn USD)",
        )


def _flag_range(
    panel: pd.DataFrame,
    column: str,
    bounds: tuple[float, float],
    label: str,
) -> None:
    if column not in panel.columns:
        print(f"[panel] WARNING: expected column {column!r} missing from panel")
        return
    lo, hi = bounds
    s = panel[column].dropna()
    if s.empty:
        print(f"[panel] WARNING: {label} is entirely NaN")
        return
    bad = s[(s < lo) | (s > hi)]
    print(
        f"[panel] {label}: min={s.min():.2f} max={s.max():.2f} "
        f"(plausible band {lo:g}–{hi:g})"
    )
    if not bad.empty:
        print(
            f"[panel] WARNING: {len(bad)} {label} value(s) outside "
            f"{lo:g}–{hi:g}. Not dropping. e.g. {bad.head(3).to_dict()}"
        )


def _load_cbu_fx() -> pd.DataFrame:
    matches = sorted(RAW_DIR.glob("cbu_fx_MS_*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No CBU FX cache in {RAW_DIR} (expected cbu_fx_MS_*.csv). "
            "Run pull_cbu_fx_series() first."
        )
    path = matches[-1]
    if len(matches) > 1:
        print(f"[panel] multiple CBU FX caches; using {path.name}")
    print(f"[panel] loading {path.name}")
    fx = pd.read_csv(path, index_col=0, parse_dates=True)
    rename = {}
    if "USD" in fx.columns:
        rename["USD"] = "usd_uzs"
    if "RUB" in fx.columns:
        rename["RUB"] = "rub_uzs"
    return fx.rename(columns=rename)


def _load_fred_monthly(series_id: str, name: str) -> pd.Series:
    path = RAW_DIR / f"fred_{series_id}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No FRED cache at {path}. Run pull_fred_series({series_id!r}) first."
        )
    print(f"[panel] loading {path.name} as {name}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    s.name = name
    return s


def _load_remittances() -> pd.Series:
    path = RAW_DIR / "cbu_bop_remittances.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No remittances cache at {path}. Drop {ANALYTICAL_XLSX_NAME} "
            "(or CBU PDF/DOCX releases) in data/raw/manual/ and run "
            "pull_cbu_bop_remittances() first — do not build the panel "
            "without this series."
        )
    print(f"[panel] loading {path.name}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "remittances_mn_usd" not in df.columns:
        raise ValueError(
            f"{path.name} has columns {list(df.columns)}; expected "
            "remittances_mn_usd (the quarterly flow, not a YTD cumulative)."
        )
    s = pd.to_numeric(df["remittances_mn_usd"], errors="coerce")
    s.name = "remittances_mn_usd"
    return s


def build_panel(upsample_method: str = "step") -> pd.DataFrame:
    """Load raw caches, upsample remittances, align, sanity-check, write CSV."""
    fx = _load_cbu_fx()
    usd_uzs = fx["usd_uzs"].rename("usd_uzs") if "usd_uzs" in fx.columns else None
    if usd_uzs is None:
        raise ValueError("CBU FX cache has no USD column.")
    rub_uzs = fx["rub_uzs"].rename("rub_uzs") if "rub_uzs" in fx.columns else None

    rub_usd = _load_fred_monthly(FRED_SERIES["rub_usd_monthly"], "rub_usd")
    brent_daily = _load_fred_monthly(FRED_SERIES["brent_crude_daily"], "brent")
    # Daily Brent has weekend/holiday NaNs already coerced from FRED's '.'.
    # Mean over observed days in the month — not interpolated, not ffilled.
    n_brent_nan = int(brent_daily.isna().sum())
    if n_brent_nan:
        print(
            f"[panel] Brent daily has {n_brent_nan} NaN(s) (FRED weekends/"
            "holidays or missing). Monthly mean skips them; not filling."
        )
    brent = brent_daily.resample("MS").mean().rename("brent")

    rem_q = _load_remittances()
    rem_m = upsample_quarterly_to_monthly(rem_q, method=upsample_method)

    pieces: list[pd.Series] = [usd_uzs, rub_usd, brent, rem_m]
    if rub_uzs is not None:
        pieces.insert(1, rub_uzs)

    panel = align_to_common_monthly(*pieces)
    sanity_check(panel)

    out_path = PROCESSED_DIR / "panel.csv"
    panel.to_csv(out_path)
    print(f"[panel] wrote {out_path} shape={panel.shape}")
    return panel


if __name__ == "__main__":
    build_panel(upsample_method="step")
