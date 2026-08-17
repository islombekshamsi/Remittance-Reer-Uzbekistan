"""
data_pull.py — Raw data ingestion for the remittances/REER project.

Two live sources implemented here, both hit directly over HTTP with no
API key required:
  - CBU exchange rate archive (cbu.uz)      -> daily USD/UZS, RUB/UZS
  - FRED public CSV export (fred.stlouisfed.org) -> RUB/USD, Brent crude

Two sources stubbed out below, because neither has a stable public API —
both require a manual download that you point these functions at:
  - CBU Balance of Payments (personal transfers + compensation of
    employees — this is your actual remittances figure, the FX archive
    above is NOT remittances, just the exchange rate)
  - stat.uz CPI

This script only pulls and caches RAW data to data/raw/. It does not
resample, align frequencies, or merge — that belongs in build_panel.py,
which is the harder problem (CBU BOP is quarterly, FRED is monthly, WB
is annual — you reconcile that there, not here).

IMPORTANT: I could not test the live CBU/FRED requests from the sandbox
this was written in (outbound network there is allowlisted to package
registries only, not cbu.uz or fred.stlouisfed.org). The CBU field names
below (`Rate`, `Nominal`) are based on the schema CBU has historically
published and that's widely used by third-party integrations, but
print(raw_response) once and confirm before you trust a full historical
pull. Same caution applies to the FRED series IDs — double check
fred.stlouisfed.org/series/<id> still resolves to what you expect.

Usage:
    pip install -r requirements.txt
    python src/data_pull.py
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MANUAL_DIR = RAW_DIR / "manual"
RAW_DIR.mkdir(parents=True, exist_ok=True)
MANUAL_DIR.mkdir(parents=True, exist_ok=True)

CBU_BASE = "https://cbu.uz/en/arkhiv-kursov-valyut/json"  # /{currency}/{date}/ ; drop currency for all currencies that day
REQUEST_DELAY_SECONDS = 0.5   # be polite — this is a small central bank's server, not a CDN
REQUEST_TIMEOUT = 15
MAX_WEEKEND_LOOKBACK_DAYS = 4  # CBU doesn't publish every calendar day; walk backward to the last published rate


# --------------------------------------------------------------------------
# CBU exchange rate archive
# --------------------------------------------------------------------------

def _fetch_cbu_day(day: date, currency: str) -> list[dict]:
    """Fetch CBU's official rate for one currency on one calendar day.
    Returns [] if nothing was published that day (weekends/holidays)."""
    url = f"{CBU_BASE}/{currency}/{day.isoformat()}/"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"[cbu] {day} {currency}: request failed ({e})")
        return []


def _rate_with_weekend_fallback(day: date, currency: str) -> float | None:
    """Walk backward up to MAX_WEEKEND_LOOKBACK_DAYS to find the most
    recent published rate if `day` itself has no data."""
    for offset in range(MAX_WEEKEND_LOOKBACK_DAYS + 1):
        try_day = day - pd.Timedelta(days=offset)
        data = _fetch_cbu_day(try_day, currency)
        if data:
            record = data[0]
            rate = float(record["Rate"])
            nominal = float(record.get("Nominal", 1))  # some currencies quote per 100/1000 units — USD/RUB are per-1, but don't assume
            return rate / nominal
    return None


def pull_cbu_fx_series(
    start: str,
    end: str,
    currencies: tuple[str, ...] = ("USD", "RUB"),
    freq: str = "MS",
) -> pd.DataFrame:
    """Build a CBU official-rate time series between `start` and `end`.

    `freq="MS"` (month start) samples once per month rather than every
    day — 15 years of daily data across 2 currencies is 10,000+ requests
    against a small government server. Monthly sampling is plenty for a
    monthly VECM. Pass freq="D" only if you specifically need daily
    granularity (e.g. GARCH/volatility work) and are prepared for a slow,
    heavily-cached pull.
    """
    cache_path = RAW_DIR / f"cbu_fx_{freq}_{start}_{end}.csv"
    if cache_path.exists():
        print(f"[cbu] using cached {cache_path.name}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    dates = pd.date_range(start=start, end=end, freq=freq)
    rows = []
    for d in dates:
        record = {"date": d}
        for ccy in currencies:
            record[ccy] = _rate_with_weekend_fallback(d.date(), ccy)
            time.sleep(REQUEST_DELAY_SECONDS)
        rows.append(record)
        print(f"[cbu] pulled {d.date()}")

    df = pd.DataFrame(rows).set_index("date")
    df.to_csv(cache_path)
    print(f"[cbu] wrote {cache_path}")
    return df


# --------------------------------------------------------------------------
# FRED — public fredgraph.csv export, no API key needed
# --------------------------------------------------------------------------

FRED_SERIES = {
    "rub_usd_monthly": "CCUSMA02RUM618N",  # OECD, monthly avg RUB per USD
    "brent_crude_daily": "DCOILBRENTEU",   # daily Brent spot, USD/barrel — resample to monthly yourself
}


def pull_fred_series(series_id: str, start: str | None = None, end: str | None = None) -> pd.Series:
    """Pull one FRED series via the public CSV export.

    Verify the series_id still resolves at fred.stlouisfed.org/series/<id>
    before trusting a pull — FRED renames/discontinues/replaces series
    more often than you'd expect, especially anything Russia-related
    post-2022.
    """
    cache_path = RAW_DIR / f"fred_{series_id}.csv"
    if cache_path.exists():
        print(f"[fred] using cached {cache_path.name}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        df = pd.read_csv(url, parse_dates=["DATE"], index_col="DATE")
        df.to_csv(cache_path)
        print(f"[fred] wrote {cache_path}")

    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")  # FRED marks missing values with "."
    if start:
        s = s[s.index >= start]
    if end:
        s = s[s.index <= end]
    return s


# --------------------------------------------------------------------------
# Manual sources — no stable public API for either. Download raw files into
# MANUAL_DIR yourself, then implement the parse below.
# --------------------------------------------------------------------------

def pull_cbu_bop_remittances() -> pd.DataFrame:
    """CBU Balance of Payments: personal transfers + compensation of
    employees. THIS is the actual remittances series — the FX archive
    above is just the exchange rate, not remittance flows.

    No API for this. CBU publishes it as quarterly BOP release tables
    under cbu.uz/en/statistics/.

    TODO:
      1. Download the latest BOP release (Excel) from cbu.uz/en/statistics/
      2. Save to data/raw/manual/cbu_bop_<release_date>.xlsx
      3. Locate the "Personal transfers" and "Compensation of employees"
         rows (label wording has shifted release to release — check both
         English and Russian/Uzbek versions if the English release lags)
      4. Parse and sum them here, indexed by quarter
    """
    raise NotImplementedError(
        f"Drop CBU BOP Excel release(s) in {MANUAL_DIR} and implement the parse. "
        "Table layout changes release to release, so this isn't safely "
        "automatable without eyeballing each file first."
    )


def pull_stat_uz_cpi() -> pd.DataFrame:
    """Uzbekistan CPI — ideally disaggregated tradable vs. non-tradable if
    you want to construct your own monthly REER rather than rely on World
    Bank's annual one. No API; stat.uz publishes PDF/Excel bulletins.

    TODO: same pattern as pull_cbu_bop_remittances() — download to
    data/raw/manual/, then parse here.
    """
    raise NotImplementedError(f"Drop stat.uz CPI bulletins in {MANUAL_DIR} and implement the parse.")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    fx = pull_cbu_fx_series("2010-01-01", "2025-12-01", currencies=("USD", "RUB"), freq="MS")
    print(fx.tail())

    rub = pull_fred_series(FRED_SERIES["rub_usd_monthly"], start="2010-01-01")
    print(rub.tail())

    brent = pull_fred_series(FRED_SERIES["brent_crude_daily"], start="2010-01-01")
    print(brent.resample("MS").mean().tail())
