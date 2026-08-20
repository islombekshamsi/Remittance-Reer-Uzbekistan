"""Tests for CBU BOP YTD→quarterly differencing and release parsing.

No live PDFs here — a silent off-by-one in the difference is exactly the
bug that would look like a beautiful remittance series later.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cbu_bop import (
    CbuBopParseError,
    parse_cbu_number,
    parse_release_period,
    parse_secondary_income_from_text,
    ytd_to_quarterly,
)


def test_ytd_to_quarterly_differences_within_year_only():
    ytd = pd.Series(
        {
            pd.Period("2024Q1"): 100.0,
            pd.Period("2024Q2"): 250.0,  # H1
            pd.Period("2024Q3"): 400.0,  # 9 months
            pd.Period("2024Q4"): 500.0,  # full year
            pd.Period("2025Q1"): 80.0,   # new year — must NOT subtract 500
        }
    )
    flows = ytd_to_quarterly(ytd)
    assert flows[pd.Period("2024Q1")] == pytest.approx(100.0)
    assert flows[pd.Period("2024Q2")] == pytest.approx(150.0)
    assert flows[pd.Period("2024Q3")] == pytest.approx(150.0)
    assert flows[pd.Period("2024Q4")] == pytest.approx(100.0)
    assert flows[pd.Period("2025Q1")] == pytest.approx(80.0)


def test_ytd_to_quarterly_missing_predecessor_is_nan_not_ytd(capsys):
    """A YTD figure without the previous quarter must not be used as a flow.

    Q2 YTD without Q1 → Q2 flow is NaN (cannot recover Q2).
    Q3 YTD without Q2 YTD → Q3 flow is NaN. If Q2 YTD *is* present, Q3 can
    still be differenced even when the Q2 *flow* itself is unknown.
    """
    ytd = pd.Series(
        {
            pd.Period("2024Q2"): 250.0,  # no Q1
            pd.Period("2025Q1"): 80.0,
            pd.Period("2025Q3"): 300.0,  # no Q2
        }
    )
    flows = ytd_to_quarterly(ytd)
    assert pd.isna(flows[pd.Period("2024Q2")])
    assert flows[pd.Period("2025Q1")] == pytest.approx(80.0)
    assert pd.isna(flows[pd.Period("2025Q3")])
    logged = capsys.readouterr().out
    assert "WARNING" in logged
    assert "inventing" in logged


def test_parse_release_period_handles_cbu_filename_variants():
    assert parse_release_period("eng_BOP_IIP_ED_2026Q1.pdf") == (2026, 1)
    assert parse_release_period("eng_BOP_IIP_ED_2025Q2.pdf") == (2025, 2)
    assert parse_release_period("eng_BOP_IIP_ED_2025Q4-_2_.pdf") == (2025, 4)
    assert parse_release_period("eng_BOP_IIP_ED_2024H1.pdf") == (2024, 2)
    assert parse_release_period("eng_BOP_IIP_9M_2023.docx") == (2023, 3)
    assert parse_release_period("eng_BOP_-IIP_1Q_2023.docx") == (2023, 1)
    assert parse_release_period("en_BOP_-IIP_3Q2022.docx") == (2022, 3)
    assert parse_release_period("eng_BOP_IIP_ED_2023.pdf") == (2023, 4)
    with pytest.raises(CbuBopParseError):
        parse_release_period("BALANCE-OF-PAYMENTS_-INTERNATIONAL-INVESTMENT-POSITION.docx")


def test_parse_secondary_income_picks_reporting_year_column():
    q1_2026 = """
    Addenda 1. Balance of Payments for the Quarters I of 2024-2026
    (analytic presentation)
    (million USD)
    Quarter I Quarter I Quarter I
    Indicators
    of 2024 of 2025 of 2026
    A. Current account balance -2 104.5 -303.8 -5 793.3
    Goods, credit (exports)
    5 331.4 6 241.3 3 444.5
    Goods, debit (imports)
    8 063.4 7 669.4 10 195.3
    Services, credit (exports)
    1 203.2 1 839.1 2 173.1
    Services, debit (imports)
    2 369.0 3 126.9 3 712.2
    Balance on goods and services -3 897.8 -2 715.9 -8 289.9
    Primary income, credit 1 406.3 1 704.4 1 430.3
    Primary income, debit 1 061.6 1 295.7 1 473.4
    Secondary income, credit 1 690.1 2 346.0 2 906.9
    Secondary income, debit 241.5 342.5 367.1
    """
    nine_month = """
    Addenda 1. Balance of Payments for 9 months of 2023-2025
    (million USD)
    Indicators
    Secondary income, credit 6 957,2 8 738,7 10 859,4
    """
    assert parse_secondary_income_from_text(q1_2026, 2026) == pytest.approx(2906.9)
    assert parse_secondary_income_from_text(q1_2026, 2024) == pytest.approx(1690.1)
    assert parse_secondary_income_from_text(nine_month, 2025) == pytest.approx(10859.4)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1 690.1", 1690.1),
        ("6 957,2", 6957.2),
        ("-5 793.3", -5793.3),
        ("15 143,1", 15143.1),
        ("0,3", 0.3),
    ],
)
def test_parse_cbu_number(raw: str, expected: float):
    assert parse_cbu_number(raw) == pytest.approx(expected)
