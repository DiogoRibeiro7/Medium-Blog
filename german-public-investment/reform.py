"""Early evidence on the March 2025 reform: quarterly public investment to mid-2026.

    python reform.py

Gross fixed capital formation of general government (P.51G), quarterly, not seasonally adjusted,
as a rolling four-quarter sum over rolling four-quarter GDP, for Germany and the peers, 2015 to the
latest quarter; first-half comparisons 2024, 2025 and 2026; and the documented figures on the
reform and on investment needs (``documented.csv``), set against the shortfalls computed in
``investment.py``.

Writes ``results/reform_quarterly.csv`` and ``results/reform_summary.txt``.
"""

from __future__ import annotations

import pandas as pd

from common import HERE, PEERS, RESULTS, gdp, quarterly, write


def documented() -> pd.DataFrame:
    return pd.read_csv(HERE / "documented.csv").set_index("item")


def run() -> str:
    q = quarterly()
    rows = []
    for c, f in q.groupby("geo"):
        f = f.sort_values("quarter").set_index("quarter")
        r = 100 * f["P51G"].rolling(4).sum() / f["gdp"].rolling(4).sum()
        for k, v in r.dropna().items():
            rows.append(dict(geo=c, quarter=k, gfcf_4q=float(v)))
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "reform_quarterly.csv", index=False)
    W = R.pivot_table(index="quarter", columns="geo", values="gfcf_4q")
    last = W["DE"].dropna().index.max()
    lines = [f"Gross public investment, rolling four quarters, % of GDP; latest quarter {last}"]
    for k in ("2019-Q4", "2023-Q4", "2024-Q4", "2025-Q2", "2025-Q4", last):
        if k in W.index:
            lines.append(f"  {k}: Germany {W.loc[k, 'DE']:.2f}; peers " + ", ".join(f"{p} {W.loc[k, p]:.2f}" for p in PEERS) + f"; peer mean {W.loc[k, list(PEERS)].mean():.2f}")
    piv = q.pivot_table(index="quarter", columns="geo", values="P51G")
    h = {yr: piv.loc[[f"{yr}-Q1", f"{yr}-Q2"]].sum() for yr in (2024, 2025, 2026)}
    lines.append("")
    lines.append("First-half gross public investment, EUR bn: Germany " + ", ".join(f"{yr} {h[yr]['DE'] / 1000:.1f}" for yr in h))
    lines.append(f"  growth, H1 2025 over H1 2024: Germany {100 * (h[2025]['DE'] / h[2024]['DE'] - 1):+.1f}%; H1 2026 over H1 2025: Germany {100 * (h[2026]['DE'] / h[2025]['DE'] - 1):+.1f}%, peer mean {100 * (h[2026][list(PEERS)] / h[2025][list(PEERS)] - 1).mean():+.1f}%")
    lines.append(f"  increase H1 2026 over H1 2024: Germany EUR {(h[2026]['DE'] - h[2024]['DE']) / 1000:.1f} bn ({100 * (h[2026]['DE'] / h[2024]['DE'] - 1):+.1f}%)")

    doc = documented()
    Y = gdp()
    y25 = Y.loc[2025, "DE"] / 1000
    lines.append("")
    lines.append(f"German GDP 2025: EUR {y25:.0f} bn")
    svik = doc.loc["reform_svik", "value"]
    lines.append(f"SVIK: EUR {svik:.0f} bn over 12 years = EUR {svik / 12:.1f} bn a year = {100 * svik / 12 / y25:.2f}% of 2025 GDP")
    lines.append(f"SVIK 2025: planned EUR {doc.loc['svik_2025_planned', 'value']:.1f} bn, outflow about EUR {doc.loc['svik_2025_actual', 'value']:.0f} bn ({100 * doc.loc['svik_2025_actual', 'value'] / doc.loc['svik_2025_planned', 'value']:.0f}% of plan); ifo: borrowing EUR {doc.loc['ifo_svik_credit_2025', 'value']:.1f} bn against an increase in federal investment of EUR {doc.loc['ifo_federal_investment_increase_2025', 'value']:.1f} bn")
    lines.append(f"SVIK disbursed by July 2026: EUR {doc.loc['svik_disbursed_july_2026', 'value']:.1f} bn ({100 * doc.loc['svik_disbursed_july_2026', 'value'] / svik:.0f}% of the fund)")
    lines.append(f"Investment needs: KfW municipal backlog EUR {doc.loc['kfw_backlog_2026', 'value']:.1f} bn ({100 * doc.loc['kfw_backlog_2026', 'value'] / y25:.1f}% of 2025 GDP); IMK and IW additional needs EUR {doc.loc['iw_imk_needs', 'value']:.0f} bn over ten years ({100 * doc.loc['iw_imk_needs', 'value'] / 10 / y25:.2f}% of 2025 GDP a year)")
    return write("reform_summary.txt", lines)


if __name__ == "__main__":
    print(run())
