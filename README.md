# Remittances and the Real Exchange Rate in Uzbekistan

Research project testing whether Uzbekistan's remittance inflows — nearly
$19B in 2025, among the highest remittance-to-GDP ratios in the world —
are driving real exchange rate appreciation (the "Dutch disease" channel),
using the country's 2017 currency unification as a natural experiment.

## Research questions

1. Do remittances and the real exchange rate move together over the long
   run, or is any correlation just coincidence?
2. Did that relationship change after September 2017, when Uzbekistan
   unified its exchange rate regime and prices started reflecting real
   market conditions for the first time?
3. Does the currency respond differently to a remittance surge than to a
   remittance drop (asymmetric response)?
4. Does the effect scale with dependence? Tajikistan (~45% of GDP) and
   Kyrgyzstan (~24%) are far more remittance-dependent than Uzbekistan
   (~14%) — a dose-response test of whether the mechanism is real.

## Status

Data collection stage — see [Issues](./issues) for progress.

## Remittances proxy (read this before trusting the series)

CBU's Addenda 1 table does **not** publish "Personal transfers" or
"Compensation of employees" as separate lines — only the aggregates
"Secondary income, credit" and "Primary income, credit." v1 of this
project uses **Secondary income, credit** as the remittances proxy.
That overstates true remittances (it includes current transfers that
are not household remittances: general-government transfers, NGO
grants, etc.).

TODO(v2): switch to the IMF BOP database (data.imf.org) disaggregated
series — personal transfers + compensation of employees — once that
pull is implemented.

The preferred source is `data/raw/manual/BOP_Analytical_Uzbekistan.xlsx`
(BPM6 analytic presentation). Those figures are **already discrete
quarterly flows** — `pull_cbu_bop_remittances()` does not difference them.
Individual CBU PDF/DOCX releases are year-to-date and only used as a
fallback (via `ytd_to_quarterly()`).

```
python src/data_pull.py          # FX + FRED (cached under data/raw/)
PYTHONPATH=src python -c "from data_pull import pull_cbu_bop_remittances; pull_cbu_bop_remittances()"
python src/build_panel.py        # writes data/processed/panel.csv
```
