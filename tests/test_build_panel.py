"""Unit tests for monthly upsampling and panel alignment.

Synthetic series only — a gap or a misaligned frequency here is the kind of
bug unit-root tests will not catch for you later.
"""

from __future__ import annotations

import pandas as pd
import pytest

from build_panel import align_to_common_monthly, upsample_quarterly_to_monthly


def _quarterly() -> pd.Series:
    idx = pd.period_range("2020Q1", "2020Q4", freq="Q")
    return pd.Series([10.0, 20.0, 30.0, 40.0], index=idx, name="remittances_mn_usd")


def test_step_upsample_holds_quarterly_value_for_three_months():
    monthly = upsample_quarterly_to_monthly(_quarterly(), method="step")
    expected_idx = pd.date_range("2020-01-01", "2020-12-01", freq="MS")
    assert monthly.index.equals(expected_idx)
    # Q1 held through Jan–Mar, Q2 through Apr–Jun, etc.
    assert (monthly.loc["2020-01-01":"2020-03-01"] == 10.0).all()
    assert (monthly.loc["2020-04-01":"2020-06-01"] == 20.0).all()
    assert (monthly.loc["2020-07-01":"2020-09-01"] == 30.0).all()
    assert (monthly.loc["2020-10-01":"2020-12-01"] == 40.0).all()


def test_spline_upsample_passes_through_knots_and_moves_interior_months():
    monthly = upsample_quarterly_to_monthly(_quarterly(), method="spline")
    # Knots must be preserved — otherwise we are not interpolating this series.
    assert monthly.loc["2020-01-01"] == pytest.approx(10.0)
    assert monthly.loc["2020-04-01"] == pytest.approx(20.0)
    # Feb is between Q1 and Q2 knots; a step function would still be 10.
    assert monthly.loc["2020-02-01"] != pytest.approx(10.0)
    assert 10.0 < float(monthly.loc["2020-02-01"]) < 20.0


def test_align_inner_join_on_overlap_and_keeps_interior_nans():
    usd = pd.Series(
        [8000.0, 8100.0, 8200.0, 8300.0, 8400.0, 8500.0],
        index=pd.date_range("2020-01-01", periods=6, freq="MS"),
        name="usd_uzs",
    )
    rem = pd.Series(
        [100.0, 100.0, 100.0, 110.0, 110.0, 110.0],
        index=pd.date_range("2020-01-01", periods=6, freq="MS"),
        name="remittances_mn_usd",
    )
    brent = pd.Series(
        [50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
        index=pd.date_range("2020-02-01", periods=7, freq="MS"),
        name="brent",
    )
    rub = pd.Series(
        [70.0, 71.0, 72.0, 73.0, 74.0, 75.0],
        index=pd.date_range("2019-12-01", periods=6, freq="MS"),
        name="rub_usd",
    )
    # Punch a hole in FX inside the overlap; alignment must not drop it.
    usd.loc["2020-03-01"] = float("nan")

    panel = align_to_common_monthly(usd, rub, brent, rem)
    expected = pd.date_range("2020-02-01", "2020-05-01", freq="MS")
    assert panel.index.equals(expected)
    assert pd.isna(panel.loc["2020-03-01", "usd_uzs"])
    assert panel.loc["2020-03-01", "brent"] == pytest.approx(51.0)
    assert panel.shape == (4, 4)
