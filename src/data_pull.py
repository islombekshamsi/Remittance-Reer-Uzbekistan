"""
data_pull.py — Raw data ingestion for the remittances/REER project.

Two live sources implemented here, both hit directly over HTTP with no
API key required:
  - CBU exchange rate archive (cbu.uz)      -> daily USD/UZS, RUB/UZS
  - FRED public CSV export (fred.stlouisfed.org) -> RUB/USD, Brent crude

CBU Balance of Payments remittances are parsed from manually downloaded
releases (no stable API). Drop files in data/raw/manual/ named like
eng_BOP_IIP_ED_2025Q3.pdf — those publications are year-to-date
cumulative, so pull_cbu_bop_remittances() differences them into discrete
quarterly flows before caching. stat.uz CPI is still a stub.

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

from cbu_bop import (
    extract_secondary_income_credit,
    select_preferred_releases,
    ytd_to_quarterly,
)

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
    """CBU BOP remittances proxy from manually downloaded quarterly releases.

    THIS is the remittances series — the FX archive above is just the
    exchange rate. There is no CBU API for BOP tables; download the English
    publications from cbu.uz/en/publications/balance-of-payments/ into
    data/raw/manual/. Expected names look like:

        eng_BOP_IIP_ED_2026Q1.pdf
        eng_BOP_IIP_ED_2025Q2.pdf   # H1 2025 cumulative
        eng_BOP_IIP_ED_2025Q3.pdf   # 9-month 2025 cumulative
        eng_BOP_IIP_9M_2023.docx    # older twins; prefer .docx over .pdf

    2024+ releases are PDF-only (parsed with pdfplumber). 2019–2023 have
    .docx twins — those are preferred because python-docx reads the Addenda 1
    table as a real table, whereas recent PDFs fragment it.

    v1 line item is "Secondary income, credit", not personal transfers +
    compensation of employees (those rows are not in Addenda 1). That is a
    known simplification — see the README.

    Each file is YTD cumulative, not a quarterly flow. This function
    differences them via ytd_to_quarterly() before writing the cache.
    """
    cache_path = RAW_DIR / "cbu_bop_remittances.csv"
    if cache_path.exists():
        print(f"[cbu-bop] using cached {cache_path.name}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    files = [
        p for p in MANUAL_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".pdf", ".docx"}
        and not p.name.startswith(".")
    ]
    if not files:
        raise FileNotFoundError(
            f"No CBU BOP releases in {MANUAL_DIR}. Download the English "
            "PDF/DOCX files from cbu.uz/en/publications/balance-of-payments/ "
            "(names like eng_BOP_IIP_ED_2025Q3.pdf) into that folder, then "
            "re-run pull_cbu_bop_remittances(). Do not skip this series."
        )

    chosen = select_preferred_releases(files)
    ytd_rows: list[dict] = []
    for (year, quarter), path in sorted(chosen.items()):
        value = extract_secondary_income_credit(path, reporting_year=year)
        period = pd.Period(year=year, quarter=quarter, freq="Q")
        ytd_rows.append(
            {
                "date": period.to_timestamp(how="start"),
                "ytd_mn_usd": value,
                "source_file": path.name,
            }
        )
        print(f"[cbu-bop] {period} YTD from {path.name}: {value}")

    ytd = pd.DataFrame(ytd_rows).set_index("date")["ytd_mn_usd"]
    flows = ytd_to_quarterly(ytd)
    flows.index = flows.index.to_timestamp(how="start")
    out = pd.DataFrame({"remittances_mn_usd": flows, "ytd_mn_usd": ytd})
    out.to_csv(cache_path)
    print(f"[cbu-bop] wrote {cache_path} ({len(flows)} quarters)")
    return out


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
