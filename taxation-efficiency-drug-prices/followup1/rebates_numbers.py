"""Follow-up 1 to the taxation-efficiency-drug-prices series: a test of the conjecture left open by Part 7.

Every number quoted in the article is printed by this script. Run, after fetch.py has put the documents in
data/ (the GitHub Actions workflow does this and commits them):

    python rebates_numbers.py > output.txt

The conjecture (Part 7). Reference pricing turns each country's list price into something other payers use, so
small countries that are both referenced and referencing should bargain through confidential returns rather
than through list prices, and should do more of it as the share of spending priced by reference grows. Two
predictions follow and are tested here:

  P1  The returns to the SNS as a share of its spending on medicines rose over the last decade, faster in
      hospitals than in pharmacies.
  P2  The gap between Portugal's list prices and its net prices is larger than in the countries it references
      (Spain, France, Italy and Belgium).

Sections
1. Data and coverage.
2. Portugal: what the published pieces say about the returns, 2013-2025.
3. France: rebates (remises) 2016-2024, by setting, and the safeguard clause.
4. Belgium: compensations under article 81/111 conventions and levies, 2015-2024, by setting.
5. Italy and Spain: what is published, and why it does not give a comparable gap.
6. Test of P1: the trend in each country.
7. Test of P2: the gap between list and net in 2024.
8. Net prices: Portugal's net price against France's and Belgium's, with the uncertainty of its inputs.
9. What the referenced countries' rebates do to Portugal's price ceiling, and who pays the list price.

Inputs taken from earlier parts are read from part3/figure_data.json, part7/output.txt and part7/data.
"""

from __future__ import annotations

import gzip
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROOT = HERE.parent
P7 = ROOT / "part7" / "data"

SEED = 20261113
REPS = 20000

# Statutory figures that are not in a data file. Each is quoted with its source and used as stated.
ES_DEDUCTION = 0.075        # Spain, Real Decreto-ley 8/2010 (art. 8-9), as amended by Real Decreto-ley 9/2011:
ES_ORPHAN = 0.04            # 7.5% deduction on SNS invoices of medicines outside the reference price system,
ES_TEN_YEARS = 0.15         # 4% for orphan medicines, 15% after ten years of public funding with no generic
IT_RIPIANO = 0.50           # Italy, Law 145/2018, art. 1, c. 580: companies repay 50% of the overrun of the
                            # ceiling on direct (hospital) purchases


def section(title: str) -> None:
    print()
    print("=" * 94)
    print(title)
    print("=" * 94)


def text(name: str, where: Path = DATA) -> str:
    return re.sub(r"\s+", " ", gzip.open(where / f"{name}.txt.gz", "rt", encoding="utf-8").read())


def find(pattern: str, s: str, what: str) -> re.Match:
    m = re.search(pattern, s)
    if not m:
        raise SystemExit(f"{what}: pattern not found ({pattern})")
    return m


def num(s: str) -> float:
    """Portuguese, French or Italian number: '1.792,6' -> 1792.6, '7,9' -> 7.9, '41 070' -> 41070."""
    s = s.strip().strip("[]%").replace(" ", "")
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    return float(s)


def row(label: str, s: str, n: int, what: str) -> list[float]:
    """The n numbers that follow a row label in a flattened table."""
    m = find(re.escape(label) + r"((?: \[?[\d\.]+(?:,\d+)?%?\]?){" + str(n) + "})", s, what)
    return [num(x) for x in m.group(1).split()]


def sns_bill() -> pd.DataFrame:
    hosp = pd.read_csv(P7 / "sns_despesa_com_medicamentos_nos_hospitais_do_sns.csv.gz")
    amb = pd.read_csv(P7 / "sns_despesa_com_medicamentos_no_ambulatorio_sns.csv.gz")
    for d in (hosp, amb):
        d["year"] = d["tempo"].str[:4].astype(int)
    months = hosp.groupby("year")["tempo"].nunique()
    full = [int(y) for y, n in months.items() if n == 12]
    h = hosp.groupby("year")["encargos_sns_hospitalar"].sum() / 1e6
    a = amb.groupby("year")[["encargos_sns_ambulatorio", "encargos_utentes_ambulatorio"]].sum() / 1e6
    bill = pd.DataFrame({"hospital": h, "pharmacy": a["encargos_sns_ambulatorio"],
                         "patients": a["encargos_utentes_ambulatorio"]}).loc[full]
    bill["sns"] = bill["hospital"] + bill["pharmacy"]
    return bill


def corners(*ranges):
    return itertools.product(*ranges)


def main() -> None:
    rng = np.random.default_rng(SEED)
    figure: dict = {}

    # ------------------------------------------------------------------ 1
    section("1. Data and coverage")
    log = (DATA / "fetch_log.txt").read_text(encoding="utf-8")
    docs = re.findall(r"\[(\w+)\] wrote", log)
    print(f"Documents fetched and checked by fetch.py: {len(docs)}; failed: "
          f"{find(r'failed: (.*)', log, 'fetch log').group(1)}")
    for prefix, label in [("pt_", "Portugal"), ("fr_", "France"), ("be_", "Belgium"), ("it_", "Italy")]:
        print(f"  {label}: {', '.join(d for d in docs if d.startswith(prefix))}")
    bill = sns_bill()
    print(f"SNS spending on medicines (Part 7 data, SNS transparency portal): complete years {bill.index.min()}-"
          f"{bill.index.max()}")
    rules = {r["rule"]: r["vs_fr"] for r in json.loads((ROOT / "part3" / "figure_data.json").read_text())["rules"]}
    list_rel = {"FR": rules["France alone"], "ES": rules["Spain alone"], "IT": rules["Italy alone"],
                "BE": rules["Belgium alone"], "PT": rules["average of the four (in force)"]}
    p7 = (ROOT / "part7" / "output.txt").read_text(encoding="utf-8")
    pt_pharm = float(find(r"over the French price [\d.]+ \(presentations priced by all four countries\), ([\d.]+) "
                          r"\(all pharmacy medicines priced\)", p7, "Part 7 ceiling").group(1))
    print("Part 3: list prices relative to France, presentations priced in all four reference countries: "
          + ", ".join(f"{k} {v:.3f}" for k, v in list_rel.items())
          + f"; Portugal's ceiling over France for all pharmacy medicines priced {pt_pharm:.3f}")
    size = {p["geo"]: p for p in json.loads((ROOT / "part7" / "figure_data.json").read_text())["size"]["points"]}
    print(f"Population (Part 7, Eurostat 2025): Portugal {size['PT']['pop'] / 1e6:.2f} million, Belgium "
          f"{size['BE']['pop'] / 1e6:.2f} million, France {size['FR']['pop'] / 1e6:.1f} million; AIC per head "
          f"(EU=100): PT {size['PT']['aic']:.0f}, BE {size['BE']['aic']:.0f}, FR {size['FR']['aic']:.0f}")

    # ------------------------------------------------------------------ 2
    section("2. Portugal: what the published pieces say about the returns")
    print("SNS spending on medicines, EUR million (hospital SNS charges + pharmacy SNS charges):")
    for y in range(2013, bill.index.max() + 1):
        print(f"  {y}  hospital {bill.loc[y, 'hospital']:8,.1f}  pharmacy {bill.loc[y, 'pharmacy']:8,.1f}  "
              f"total {bill.loc[y, 'sns']:8,.1f}")

    print("\n(a) The State-industry agreements: the contribution each text sets")
    agr = {}
    for y, pat in [(2013, r"contribution in the amount of (\d+) million euros for the reduction of public "
                          r"expenditure on medicines in the hospital market"),
                   (2014, r"contribution in the amount of (\d+) million euro"),
                   (2015, r"contribution in the amount of (\d+) million euro"),
                   (2016, r"financial contribution amounting to (\d+) million euro"),
                   (2017, r"minimum amount of (\d+) million euros, which will be quarterly updated")]:
        agr[y] = float(find(pat, text(f"pt_agreement_{y}"), f"agreement {y}").group(1))
    find(r"replicado anualmente, entre 2019 e 2023", text("pt_agreement_2024"), "agreement 2024")
    find(r"aplicado para o ano de 2024 nos mesmos termos e condições que vigoraram no ano de 2023",
         text("pt_agreement_2024"), "agreement 2024 terms")
    a25 = text("pt_agreement_2025")
    find(r"taxas previstas no Regime de Contrib.{0,3}ção Extraordinária", a25, "agreement 2025 rates")
    m = find(r"não deve exceder os (\d+)% \(sete por cento\)", a25, "agreement 2025 growth limit")
    growth_cap = float(m.group(1)) / 100
    m = find(r"montante correspondente a (\d+)% do valor do desvio face à Base", a25, "agreement 2025 penalty")
    penalty = float(m.group(1)) / 100
    for y, v in agr.items():
        base = bill.loc[y, "hospital"] if y == 2013 else bill.loc[y, "sns"]
        what = "SNS hospital spending" if y == 2013 else "SNS spending"
        print(f"  {y}: EUR {v:.0f} million{' (hospital market only)' if y == 2013 else ''}"
              f"{' (minimum, updated each quarter with SNS spending)' if y == 2017 else ''} = {v / base:.1%} of {what}")
    print("  2016-2018 agreement replicated each year 2019-2023 and applied in 2024 on the same terms (2024 text)")
    print(f"  2025: contribution at the rates of the extraordinary contribution (CEIF) on SNS spending; if net spending "
          f"grows more than {growth_cap:.0%} a year, industry adds {penalty:.0%} of the excess")

    print("\n(b) The extraordinary contribution on the pharmaceutical industry (CEIF), revenue, EUR million (CFP):")
    cfp20 = find(r"representou (\d+) M€ em 2018, cerca de ([\d,]+) M€ em 2019 e de ([\d,]+) M€ no ano de 2020",
                 text("pt_cfp_sns_2020"), "CFP 2020")
    ceif = {2018: num(cfp20.group(1)), 2019: num(cfp20.group(2)), 2020: num(cfp20.group(3))}
    m = find(r"farmac.utica \(([\d,]+) M€, com um aumento de ([\d,]+) M€", text("pt_cfp_sns_2022"), "CFP 2022")
    ceif[2022] = num(m.group(1))
    ceif[2021] = round(ceif[2022] - num(m.group(2)), 1)
    for y, name in [(2023, "pt_cfp_sns_2023"), (2024, "pt_cfp_sns_2024"), (2025, "pt_cfp_sns_2025")]:
        ceif[y] = num(find(r"farmac.utica \(([\d,]+) M€", text(name), name).group(1))
    print("  " + "; ".join(f"{y} {v:.1f} ({v / bill.loc[y, 'sns']:.2%} of SNS spending)" for y, v in sorted(ceif.items())))

    print("\n(c) Totals of the returns")
    ap24 = text("pt_apifarma_newsletter_2024_09")
    m = find(r"Entre (\d{4}) e (\d{4}) as devoluções da Indústria Farmacêutica ao Estado aumentaram de (\d+)% "
             r"para (\d+)%", ap24, "APIFARMA study")
    y0, y1, s0, s1 = int(m.group(1)), int(m.group(2)), float(m.group(3)) / 100, float(m.group(4)) / 100
    proj = float(find(r"possa chegar, em (\d{4}), aos (\d+)% do mercado SNS hospitalar", ap24, "projection").group(2)) / 100
    print(f"  APIFARMA study (September 2024): returns rose from {s0:.0%} in {y0} to {s1:.0%} in {y1}; projected "
          f"{proj:.0%} of the SNS hospital market in 2026. The newsletter does not state the denominator of the "
          f"{s0:.0%} and {s1:.0%}:")
    apifarma_read = {}
    for col, what in [("hospital", "of SNS hospital spending"), ("sns", "of all SNS spending")]:
        e0, e1 = s0 * bill.loc[y0, col], s1 * bill.loc[y1, col]
        apifarma_read[col] = (e0, e1)
        print(f"    read as a share {what}: EUR {e0:,.0f} million in {y0}, EUR {e1:,.0f} million in {y1} "
              f"({e0 / bill.loc[y0, 'sns']:.1%} and {e1 / bill.loc[y1, 'sns']:.1%} of all SNS spending)")
    print(f"    the {proj:.0%} projection on 2025 hospital spending would be EUR {proj * bill.loc[2025, 'hospital']:,.0f} "
          f"million")

    eco = text("pt_eco_2025_05_29")
    m = find(r"ficou próxima dos ([\d\.]+) milhões de euros em 2024, um aumento de ([\d,]+)%", eco, "ECO gross")
    g24, gg = num(m.group(1)), num(m.group(2)) / 100
    m = find(r"o aumento da despesa com medicamentos em 2024 foi de (\d+)% face a 2023, com um total de ([\d,]+) mil "
             r"milhões", eco, "ECO net")
    gn, n24 = float(m.group(1)) / 100, num(m.group(2)) * 1000
    h24 = num(find(r"fixando-se em ([\d\.]+,\d) milhões de euros", eco, "ECO hospital").group(1))
    find(r"quer no âmbito dos contratos de financiamento quer no âmbito do acordo com a indústria", eco, "ECO scope")
    print(f"  Infarmed, report on 2024 (ECO, 29 May 2025): SNS spending EUR {g24:,.0f} million (+{gg:.1%}; hospitals "
          f"and primary care {h24:,.1f}; portal {bill.loc[2024, 'sns']:,.1f}); net of all returns under the financing "
          f"contracts and the agreement EUR {n24:,.0f} million (rounded in the text), +{gn:.0%} on 2023")
    r24 = g24 - n24
    g23, n23 = g24 / (1 + gg), n24 / (1 + gn)
    r23 = g23 - n23
    print(f"    implied returns: 2024 EUR {r24:,.0f} million ({r24 / g24:.1%} of spending); 2023 EUR {r23:,.0f} million "
          f"({r23 / g23:.1%}); 2023 spending implied {g23:,.1f} (portal {bill.loc[2023, 'sns']:,.1f})")
    # Rounding: '3.960' to the unit of 10, '11,5%' to a tenth, '3,5 mil milhões' to a hundred million, '9%' to a unit.
    G, GG, N, GN = (g24 - 5, g24 + 5), (gg - 0.0005, gg + 0.0005), (n24 - 50, n24 + 50), (gn - 0.005, gn + 0.005)
    sh24 = [1 - n / g for g, n in corners(G, N)]
    sh23 = [1 - (n / (1 + b)) / (g / (1 + a)) for g, a, n, b in corners(G, GG, N, GN)]
    rise = [(1 - n / g) - (1 - (n / (1 + b)) / (g / (1 + a))) for g, a, n, b in corners(G, GG, N, GN)]
    r24_lo, r24_hi = min(g - n for g, n in corners(G, N)), max(g - n for g, n in corners(G, N))
    print(f"    with the rounding of the four published figures: 2024 returns EUR {r24_lo:,.0f}-{r24_hi:,.0f} million "
          f"({min(sh24):.1%}-{max(sh24):.1%}); 2023 share {min(sh23):.1%}-{max(sh23):.1%}; rise 2023-2024 "
          f"{min(rise) * 100:+.1f} to {max(rise) * 100:+.1f} percentage points (positive in every case)")
    conf24, conf23 = r24 - 200, r23 - 200
    print(f"    less the agreement's EUR 200 million (its 2016-2017 amount, if unchanged): financing contracts EUR "
          f"{conf23:,.0f} million in 2023 ({conf23 / g23:.1%}) and EUR {conf24:,.0f} million in 2024 ({conf24 / g24:.1%})")

    ap25 = text("pt_apifarma_newsletter_2025_09")
    a600 = float(find(r"na ordem dos (\d+) milh.es de euros, em 2024", ap25, "APIFARMA 2024").group(1))
    print(f"  APIFARMA newsletter (September 2025), study for CIP: returns of about EUR {a600:.0f} million in 2024 = "
          f"{a600 / bill.loc[2024, 'sns']:.1%} of SNS spending")
    inf = text("infarmed_note_2025", P7)
    m = find(r"anual estimado de (\d+)-(\d+) milh\w+, apenas cerca de (\d+)% ficam refletidos", inf, "Infarmed 2025")
    r25_lo, r25_hi, reflected = float(m.group(1)), float(m.group(2)), float(m.group(3)) / 100
    s25 = bill.loc[2025, "sns"]
    print(f"  Infarmed note (November 2025): returns in 2025 EUR {r25_lo:.0f}-{r25_hi:.0f} million = "
          f"{r25_lo / s25:.1%}-{r25_hi / s25:.1%} of SNS spending (EUR {s25:,.1f} million); about {reflected:.0%} of "
          f"them already reflected in the in-year spending reports")
    print(f"    if the {reflected:.0%} were already netted out of the portal's annual figure, as a share of spending "
          f"before any return: {r25_lo / (s25 + reflected * r25_lo):.1%}-{r25_hi / (s25 + reflected * r25_hi):.1%}")
    pt24_lo, pt24_hi = r24_lo / bill.loc[2024, "sns"], a600 / bill.loc[2024, "sns"]
    print(f"  Band used for 2024 below, from the lowest Infarmed reading to the APIFARMA figure: EUR {r24_lo:,.0f}-"
          f"{a600:,.0f} million = {pt24_lo:.1%}-{pt24_hi:.1%} of SNS spending; with the CEIF "
          f"{(r24_lo + ceif[2024]) / bill.loc[2024, 'sns']:.1%}-{(a600 + ceif[2024]) / bill.loc[2024, 'sns']:.1%}")
    print("  No source splits the returns between hospitals and pharmacies. Infarmed's note says they come mostly "
          "from the difference between the maximum price and the negotiated price; ECO's report names pharmacy "
          "medicines (diabetes, COPD, edoxaban, sacubitril/valsartan) under financing contracts with confidential prices"
          if re.search(r"preços confidenciais e limites de encargos", eco) and
          re.search(r"diferencial entre o preço máximo e o preço negociado", inf) else "check text")
    pt_series = {
        "agreement": {y: agr[y] / (bill.loc[y, "hospital"] if y == 2013 else bill.loc[y, "sns"]) for y in agr},
        "apifarma": {y0: s0, y1: s1},
        "infarmed": {2023: [min(sh23), r23 / g23, max(sh23)], 2024: [min(sh24), r24 / g24, max(sh24)],
                     2025: [r25_lo / s25, (r25_lo + r25_hi) / 2 / s25, r25_hi / s25]},
        "apifarma_cip_2024": a600 / bill.loc[2024, "sns"],
    }

    # ------------------------------------------------------------------ 3
    section("3. France: rebates negotiated by the CEPS")
    dd = text("fr_drees_dd139")
    m = find(r"passant de (\d+,\d) à (\d+,\d) milliards d'euros entre 2016 et 2024", dd, "DREES rebates")
    fr_r16, fr_r24 = num(m.group(1)), num(m.group(2))
    m = find(r"progressant modérément de (\d+,\d) à (\d+,\d) milliards d'euros", dd, "DREES gross 2016-2020")
    fr_g16, fr_g20 = num(m.group(1)), num(m.group(2))
    m = find(r"les dépenses nettes ont diminué de (\d+,\d) milliard d.euros sur cette période, avant de progresser de "
             r"(\d+,\d) milliards entre 2020 et 2024 pour atteindre (\d+,\d) milliards", dd, "DREES net")
    fr_dn1, fr_dn2, fr_n24r = num(m.group(1)), num(m.group(2)), num(m.group(3))
    m = find(r"estimée à ([\d ]+) millions d'euros en brutes, ([\d ]+) après déduction des remises produits et "
             r"d.accès dérogatoire et à ([\d ]+) millions d'euros après déduction de la clause de sauvegarde", dd,
             "DREES 2024 exact")
    fr_g24, fr_n24, fr_sn24 = (num(m.group(i)) for i in (1, 2, 3))
    m = find(r"dont (\d,\d) milliards au titre des remises produits et (\d,\d) milliard au titre des remises liées",
             dd, "DREES rebate types")
    fr_prod, fr_early = num(m.group(1)), num(m.group(2))
    fr_n20 = fr_n24r - fr_dn2
    fr_r20 = fr_g20 - fr_n20
    print(f"DREES (Dossiers n.º 139, July 2026), reimbursable medicines in pharmacies and outside hospital tariffs:")
    print(f"  2016: gross EUR {fr_g16:.1f} bn, rebates {fr_r16:.1f} bn = {fr_r16 / fr_g16:.1%}")
    print(f"  2020: gross EUR {fr_g20:.1f} bn, net {fr_n20:.1f} bn (2024 net less the 2020-2024 rise), rebates "
          f"{fr_r20:.1f} bn = {fr_r20 / fr_g20:.1%}; check of 2016 net: {fr_n20 + fr_dn1:.1f} against "
          f"{fr_g16 - fr_r16:.1f}")
    print(f"  2024: gross EUR {fr_g24:,.0f} million, net of rebates {fr_n24:,.0f} ({1 - fr_n24 / fr_g24:.1%}; product "
          f"rebates {fr_prod:.1f} bn, early access {fr_early:.1f} bn), net also of the safeguard clause {fr_sn24:,.0f} "
          f"({1 - fr_sn24 / fr_g24:.1%})")
    find(r"prix facial, dont le niveau sert de référence", dd, "DREES reference sentence")
    print("  DREES on why: firms may prefer confidential rebates to a lower list price, because the list price is the "
          "reference in negotiations on other markets (sentence found in the text)")
    m = find(r"prescrit en ville est estimé à ([\d ]+) millions d'euros en brut et à ([\d ]+) millions d'euros en net",
             dd, "DREES ville")
    fr_city_g, fr_city_n = num(m.group(1)), num(m.group(2))
    m = find(r"liste en sus affichent la plus forte croissance en valeur brute \(\+(\d,\d) milliards\), mais seulement "
             r"\+(\d,\d) milliard en net", dd, "DREES liste en sus")
    les_g, les_n = num(m.group(1)), num(m.group(2))
    f23 = text("fr_drees_es2026_fiche23")
    m = find(r"s.élève à (\d+,\d) milliards d.euros, et (\d,\d) milliards après déduction", f23, "fiche 23")
    h_g, h_n = num(m.group(1)), num(m.group(2))
    dm = num(find(r"La dépense attribuée aux dispositifs médicaux atteint (\d,\d) milliards", f23, "devices").group(1))
    fr_hosp_lo, fr_hosp_hi = (h_g - h_n) / h_g, (h_g - h_n) / (h_g - dm)
    print(f"  by setting, 2024: prescribed and dispensed in the community EUR {fr_city_g:,.0f} million gross, "
          f"{fr_city_n:,.0f} net ({1 - fr_city_n / fr_city_g:.1%}); expensive hospital products (liste en sus and early "
          f"access) EUR {h_g:.1f} bn gross, {h_n:.1f} net, rebates {h_g - h_n:.1f} bn = {fr_hosp_lo:.1%} of the total "
          f"and {fr_hosp_hi:.1%} of the medicines in it (devices {dm:.1f} bn carry no rebate in this count)")
    print(f"  liste en sus 2016-2024: +{les_g:.1f} bn gross, +{les_n:.1f} bn net, so rebates on it rose by "
          f"{les_g - les_n:.1f} bn, {(les_g - les_n) / (fr_r24 - fr_r16):.0%} of the rise in all rebates")
    igf = text("fr_igf_2026")
    m = find(r"remises ayant atteint (\d,\d) Mds€ en 2023 \(pour une dépense de médicaments remboursables de (\d+,\d) "
             r"Mds € la même année\), contre (\d,\d) Mds€ en 2019", igf, "IGF rebates")
    igf_r23, igf_g23, igf_r19 = num(m.group(1)), num(m.group(2)), num(m.group(3))
    sg = num(find(r"pour un montant versé à l.Assurance maladie de plus de (\d,\d) Mds€ en 2023 et 2024", igf,
                  "IGF safeguard").group(1))
    m = find(r"la part des baisses de prix dans la dépense nette étant passée d.environ (\d,\d) % à (\d,\d) % entre les "
             r"périodes 2012-2019 et 2020-2023", igf, "IGF price cuts")
    cut0, cut1 = num(m.group(1)) / 100, num(m.group(2)) / 100
    print(f"IGAS-IGF mission (July 2025): rebates EUR {igf_r19:.1f} bn in 2019 and {igf_r23:.1f} bn in 2023, on reimbursable "
          f"spending of {igf_g23:.1f} bn ({igf_r23 / igf_g23:.1%}); safeguard clause over EUR {sg:.1f} bn in 2023 and "
          f"2024; yearly list-price cuts {cut0:.1%} of net spending in 2012-2019, {cut1:.1%} in 2020-2023")
    slower = ((1 - cut1) / (1 - cut0)) ** 4 - 1
    print(f"  if the slower pace of list-price cuts ran for the four years 2020-2023, French list prices would end "
          f"{slower:+.1%} above the path of 2012-2019 (rough: the share of savings in net spending read as a rate of "
          f"price decline)")
    find(r"Le maintien de prix faciaux élevés s.accompagne par ailleurs de coûts de distribution croissants", igf,
         "IGF distribution")
    print("  IGF: keeping list prices high also raises distribution costs, since pharmacy and wholesale margins are set "
          "on the list price (sentence found in the text)")
    fr_series = {2016: fr_r16 / fr_g16, 2020: fr_r20 / fr_g20, 2024: 1 - fr_n24 / fr_g24}
    fr_broad24 = 1 - fr_sn24 / fr_g24

    # ------------------------------------------------------------------ 4
    section("4. Belgium: compensations under article 81/111 conventions, and levies")
    be = text("be_morse_2024")
    gross = row("Dépenses comptabilisées (docN) (1)", be, 6, "MORSE table 3 gross")
    comp = row("Compensations art. 81/111 (2)", be, 6, "MORSE table 3 compensations")
    levy = row("Prélèvements (4)", be, 6, "MORSE table 3 levies")
    real = row("Charge budgétaire réelle des dépenses", be, 6, "MORSE table 3 real")
    years = list(range(2019, 2025))
    print("MORSE 2024, table 3, EUR million: gross | compensations | levies | real cost | compensations % gross | "
          "compensations and levies % gross")
    be_series, be_broad = {}, {}
    for y, g, c, lv, r in zip(years, gross, comp, levy, real):
        check = g - c - lv - r
        be_series[y], be_broad[y] = c / g, (c + lv) / g
        print(f"  {y}  {g:8,.1f}  {c:8,.1f}  {lv:6,.1f}  {r:8,.1f}  {c / g:6.1%}  {(c + lv) / g:6.1%}"
              f"{'' if abs(check) < 0.2 else f'  (check {check:+.1f})'}")
    print(f"  2024 compensations provisional; 2019-2024: gross x{gross[-1] / gross[0]:.2f}, compensations "
          f"x{comp[-1] / comp[0]:.2f}, levies x{levy[-1] / levy[0]:.2f}")
    cum_t = row("Cumulative Turnover", be, 10, "MORSE cumulative turnover")
    cum_c = row("Cumulative Compensation", be, 10, "MORSE cumulative compensation")
    cum_p = row("Cumulative Comp Percentage", be, 10, "MORSE cumulative percentage")
    t_y = np.diff([0.0] + cum_t)
    c_y = np.diff([0.0] + cum_c)
    print("Conventions, by year (differences of the cumulative table 50, ex-factory prices), EUR million: turnover | "
          "compensation | rate")
    conv_rate = {}
    for i, y in enumerate(range(2015, 2025)):
        conv_rate[y] = c_y[i] / t_y[i]
        print(f"  {y}  {t_y[i]:8,.1f}  {c_y[i]:8,.1f}  {c_y[i] / t_y[i]:6.1%}{'  provisional' if y == 2024 else ''}")
    print(f"  cumulative rate 2015-2024 {cum_p[-1]:.1f}% (report: probably between 41.7% and 45.4%; check "
          f"{cum_c[-1] / cum_t[-1]:.1%}); compensation x{c_y[-1] / c_y[0]:.0f} and turnover x{t_y[-1] / t_y[0]:.1f} "
          f"between 2015 and 2024")
    old = text("be_morse_2022_2023")
    old_t = row("Cumulative Turnover", old, 11, "MORSE 2022-2023 turnover")
    old_c = row("Cumulative Compensation", old, 11, "MORSE 2022-2023 compensation")
    o22 = (old_c[9] - old_c[8]) / (old_t[9] - old_t[8])
    print(f"  check against the previous report (data 2022-2023): 2022 rate {o22:.1%} (this report {conv_rate[2022]:.1%})")
    m = find(r"Le chiffre d.affaires déclaré est au moins (\d,\d) Md€, pour (\d,\d) Md€ de rétrocessions", be,
             "MORSE summary")
    print(f"  executive summary 2024: declared turnover at least EUR {num(m.group(1)):.1f} bn, EUR {num(m.group(2)):.1f} "
          f"bn returned")
    m = find(r"Spécialités pharmaceutiques sous convention ([\d\.]+,\d) ([\d,]+)% ([\d\.]+,\d) ([\d,]+)% "
             r"([\d\.]+,\d) ([\d,]+)%", be, "MORSE table 2")
    conv_ph, conv_h, conv_all = num(m.group(2)) / 100, num(m.group(4)) / 100, num(m.group(6)) / 100
    m = find(r"en 2020, la part des dépenses consacrées aux médicaments sous convention était d.environ (\d+) % du "
             r"total, soit (\d+) % pour les officines ouvertes et (\d+) % pour les pharmacies hospitalières", be,
             "MORSE 2020 shares")
    c20_all, c20_ph, c20_h = (float(m.group(i)) / 100 for i in (1, 2, 3))
    print(f"  share of gross spending on medicines under a convention: 2020 pharmacies {c20_ph:.0%}, hospitals "
          f"{c20_h:.0%}, all {c20_all:.0%}; 2024 pharmacies {conv_ph:.1%}, hospitals {conv_h:.1%}, all {conv_all:.1%}")
    print(f"  change 2020-2024: pharmacies {100 * (conv_ph - c20_ph):+.1f} points (x{conv_ph / c20_ph:.2f}), hospitals "
          f"{100 * (conv_h - c20_h):+.1f} points (x{conv_h / c20_h:.2f})")

    # ------------------------------------------------------------------ 5
    section("5. Italy and Spain")
    its = text("it_aifa_sintesi_2024")
    conv_it = num(find(r"convenzionata, al netto dei payback vigenti e degli sconti versati dalle farmacie ma al lordo "
                       r"dei ticket regionali, è stata pari a ([\d\.]+,\d) mln", its, "AIFA convenzionata").group(1))
    ad_it = num(find(r"acquisti diretti al netto dei gas medicinali è pari a ([\d\.]+,\d) mln", its, "AIFA AD").group(1))
    over = num(find(r"disavanzo di \+ ([\d\.]+,\d) mln", its, "AIFA overrun").group(1))
    cn = num(find(r"accordi negoziali \(impatto Gennaio - Dicembre 2024 pari a ([\d,]+) mln", its, "AIFA notes").group(1))
    pb183 = num(find(r"Payback dell.1,83 per cento .{0,800}?pari a (\d+,\d),? milioni", its, "AIFA 1.83").group(1))
    print(f"Italy, AIFA monitoring January-December 2024: pharmacy channel (convenzionata) EUR {conv_it:,.1f} million, "
          f"direct purchases {ad_it:,.1f} million, both net of paybacks")
    print(f"  payback notes under negotiated agreements EUR {cn:.1f} million ({cn / ad_it:.1%} of direct purchases); "
          f"1.83% payback on the pharmacy channel {pb183:.1f} million")
    print(f"  overrun of the direct-purchase ceiling EUR {over:,.1f} million; companies repay {IT_RIPIANO:.0%} by law, EUR "
          f"{IT_RIPIANO * over:,.0f} million = {IT_RIPIANO * over / ad_it:.1%} of direct purchases, "
          f"{IT_RIPIANO * over / (ad_it + conv_it):.1%} of the two channels")
    print("  Confidential discounts in Italy are mostly applied on the invoice to the hospital, so the published spending "
          "is already net of them and no aggregate list-to-net gap is published")
    print(f"Spain: by law (RDL 8/2010, as amended by RDL 9/2011) a deduction of {ES_DEDUCTION:.1%} on SNS invoices of "
          f"medicines outside the reference price system, {ES_ORPHAN:.0%} for orphan medicines and {ES_TEN_YEARS:.0%} "
          f"after ten years of funding with no generic or biosimilar; confidential financed prices are not aggregated")

    # ------------------------------------------------------------------ 6
    section("6. Test of P1: did the share returned rise?")

    def pp_year(a: dict, y0: int, y1: int) -> float:
        return 100 * (a[y1] - a[y0]) / (y1 - y0)

    print(f"France (DREES perimeter): {fr_series[2016]:.1%} (2016), {fr_series[2020]:.1%} (2020), {fr_series[2024]:.1%} "
          f"(2024): {pp_year(fr_series, 2016, 2020):+.1f} points a year 2016-2020, {pp_year(fr_series, 2020, 2024):+.1f} "
          f"2020-2024")
    print(f"Belgium (compensations): {be_series[2019]:.1%} (2019) to {be_series[2024]:.1%} (2024), "
          f"{pp_year(be_series, 2019, 2024):+.1f} points a year; with levies {be_broad[2019]:.1%} to {be_broad[2024]:.1%}, "
          f"{pp_year(be_broad, 2019, 2024):+.1f}")
    print(f"Belgium, rate on each euro sold under a convention: {conv_rate[2015]:.1%} (2015), {conv_rate[2019]:.1%} "
          f"(2019), {conv_rate[2023]:.1%} (2023), {conv_rate[2024]:.1%} (2024, provisional)")
    inf_mid = {y: v[1] for y, v in pt_series["infarmed"].items()}
    print(f"Portugal, Infarmed figures: {inf_mid[2023]:.1%} (2023), {inf_mid[2024]:.1%} (2024), {inf_mid[2025]:.1%} "
          f"(2025, middle of the range): {pp_year(inf_mid, 2023, 2025):+.1f} points a year 2023-2025")
    print(f"Portugal, APIFARMA study: {s0:.0%} ({y0}) to {s1:.0%} ({y1}), {100 * (s1 - s0) / (y1 - y0):+.1f} points a year "
          f"on its own (unstated) base")
    print(f"Portugal, agreement alone (lower bound of the total, contracts not published): {agr[2015] / bill.loc[2015, 'sns']:.1%} "
          f"(2015), {agr[2016] / bill.loc[2016, 'sns']:.1%} (2016); from 2016 to 2025 the total could have risen by at most "
          f"{100 * (r25_hi / s25 - agr[2016] / bill.loc[2016, 'sns']):.1f} points ({100 * (r25_hi / s25 - agr[2016] / bill.loc[2016, 'sns']) / 9:.1f} a year)")
    print("Hospitals against pharmacies: Portugal publishes no split. France 2024: community "
          f"{1 - fr_city_n / fr_city_g:.1%}, expensive hospital products {fr_hosp_lo:.0%}-{fr_hosp_hi:.0%}. Belgium "
          f"2020-2024: share under convention {100 * (conv_ph - c20_ph):+.0f} points in pharmacies, "
          f"{100 * (conv_h - c20_h):+.0f} in hospitals")

    # ------------------------------------------------------------------ 7
    section("7. Test of P2: the gap between list and net, 2024")
    pt_mid = (pt24_lo + pt24_hi) / 2
    pt_b_lo = (r24_lo + ceif[2024]) / bill.loc[2024, "sns"]
    pt_b_hi = (a600 + ceif[2024]) / bill.loc[2024, "sns"]
    fr_n, be_n = 1 - fr_n24 / fr_g24, be_series[2024]
    print("Share of gross public spending returned by industry, 2024:")
    print(f"  rebates and contract returns only:    Portugal {pt24_lo:.1%}-{pt24_hi:.1%} | France {fr_n:.1%} | "
          f"Belgium {be_n:.1%}")
    print(f"  with levies, CEIF and safeguard:       Portugal {pt_b_lo:.1%}-{pt_b_hi:.1%} | France {fr_broad24:.1%} | "
          f"Belgium {be_broad[2024]:.1%}")
    print(f"  Portugal's share over France's: {pt24_lo / fr_n:.2f}-{pt24_hi / fr_n:.2f} (narrow), "
          f"{pt_b_lo / fr_broad24:.2f}-{pt_b_hi / fr_broad24:.2f} (broad); over Belgium's: {pt24_lo / be_n:.2f}-"
          f"{pt24_hi / be_n:.2f} (narrow), {pt_b_lo / be_broad[2024]:.2f}-{pt_b_hi / be_broad[2024]:.2f} (broad)")
    print(f"  Portugal 2025 (Infarmed): {r25_lo / s25:.1%}-{r25_hi / s25:.1%}")
    print(f"  Spain: statutory deductions {ES_ORPHAN:.0%}-{ES_TEN_YEARS:.0%} on most branded medicines; Italy: ceiling "
          f"repayment {IT_RIPIANO * over / ad_it:.1%} of hospital purchases, discounts on the invoice not published")
    print(f"  protected medicines only: Portugal, if all returns came from them (Part 7) 15.4%-24.0%; France, expensive "
          f"hospital products {fr_hosp_lo:.0%}-{fr_hosp_hi:.0%}; Belgium, convention turnover {conv_rate[2023]:.0%} "
          f"(2023)-{conv_rate[2024]:.0%} (2024)")

    # ------------------------------------------------------------------ 8
    section("8. Net prices: Portugal against France and Belgium")
    L_fr = (list_rel["PT"], pt_pharm)
    L_be = list_rel["BE"]
    print(f"Portugal's net price over a reference country's net price = (Portugal's list / its list) x (1 - Portugal's "
          f"return rate) / (1 - its return rate)")
    for label, rfr, rbe, ppt in [("narrow", fr_n, be_n, (pt24_lo, pt24_hi)),
                                 ("broad", fr_broad24, be_broad[2024], (pt_b_lo, pt_b_hi))]:
        a = rng.uniform(*L_fr, REPS)
        r = rng.uniform(*ppt, REPS)
        vs_fr = a * (1 - r) / (1 - rfr)
        vs_be = (a / L_be) * (1 - r) / (1 - rbe)
        med, lo, hi = np.median(vs_fr), np.percentile(vs_fr, 5), np.percentile(vs_fr, 95)
        bmed, blo, bhi = np.median(vs_be), np.percentile(vs_be, 5), np.percentile(vs_be, 95)
        print(f"  {label}: Portugal list/France list uniform {L_fr[0]:.3f}-{L_fr[1]:.3f}, Portugal's rate uniform "
              f"{ppt[0]:.1%}-{ppt[1]:.1%}, France {rfr:.1%}, Belgium {rbe:.1%} ({REPS} draws)")
        print(f"    Portugal net / France net {med:.3f} [{lo:.3f}, {hi:.3f}]; share of draws above 1: "
              f"{np.mean(vs_fr > 1):.1%}")
        print(f"    Portugal net / Belgium net {bmed:.3f} [{blo:.3f}, {bhi:.3f}]; share of draws above 1: "
              f"{np.mean(vs_be > 1):.1%}")
        be_fr = L_be * (1 - rbe) / (1 - rfr)
        print(f"    Belgium net / France net {be_fr:.3f}")
        even_fr = [1 - (1 - rfr) / x for x in L_fr]
        even_be = [1 - (1 - rbe) * L_be / x for x in L_fr]
        print(f"    return rate Portugal would need to match France's net price {min(even_fr):.1%}-{max(even_fr):.1%}; "
              f"Belgium's {min(even_be):.1%}-{max(even_be):.1%}")
        if label == "narrow":
            figure["net"] = {"list": {"FR": 1.0, "BE": L_be, "PT": list_rel["PT"]},
                             "net": {"FR": 1 - rfr, "BE": L_be * (1 - rbe),
                                     "PT": [list_rel["PT"] * (1 - ppt[1]), list_rel["PT"] * (1 - pt_mid),
                                            list_rel["PT"] * (1 - ppt[0])]},
                             "net_broad": {"FR": 1 - fr_broad24, "BE": L_be * (1 - be_broad[2024]),
                                           "PT": [list_rel["PT"] * (1 - pt_b_hi), list_rel["PT"] * (1 - pt_b_lo)]},
                             "ratio_fr": [float(med), float(lo), float(hi)], "ratio_be": [float(bmed), float(blo),
                                                                                         float(bhi)]}
    print(f"  Portugal at the middle of its 2024 band ({pt_mid:.1%}) and the all-four list ratio: net / France net "
          f"{list_rel['PT'] * (1 - pt_mid) / (1 - fr_n):.3f}, net / Belgium net "
          f"{list_rel['PT'] / L_be * (1 - pt_mid) / (1 - be_n):.3f}")
    floor16 = agr[2016] / bill.loc[2016, "sns"]
    floor19 = 200 / bill.loc[2019, "sns"]
    early_fr = [x * (1 - floor16) / (1 - fr_series[2016]) for x in L_fr]
    early_be = [x / L_be * (1 - floor19) / (1 - be_series[2019]) for x in L_fr]
    print(f"Earlier years, holding the list ratios of Part 3 fixed and taking the agreement alone as Portugal's return "
          f"rate (a floor, so the ratios are upper bounds):")
    print(f"  2016: France {fr_series[2016]:.1%}, Portugal at least {floor16:.1%}: Portugal net / France net at most "
          f"{min(early_fr):.3f}-{max(early_fr):.3f} (2024: {figure['net']['ratio_fr'][0]:.3f})")
    print(f"  2019: Belgium {be_series[2019]:.1%}, Portugal at least {floor19:.1%} (EUR 200 million): Portugal net / "
          f"Belgium net at most {min(early_be):.3f}-{max(early_be):.3f} (2024: {figure['net']['ratio_be'][0]:.3f})")

    # ------------------------------------------------------------------ 9
    section("9. What the referenced countries' rebates do to Portugal's ceiling, and who pays the list price")
    base = list_rel["PT"]
    lists = {k: list_rel[k] for k in ("ES", "FR", "IT", "BE")}
    print(f"Portugal's ceiling is the average of four list prices: ({' + '.join(f'{v:.3f}' for v in lists.values())})/4 "
          f"= {np.mean(list(lists.values())):.3f} of France's list price")
    cases = [("France and Belgium at their net prices (rebates only)", {"FR": fr_n, "BE": be_n}),
             ("France and Belgium at their net prices (with safeguard and levies)", {"FR": fr_broad24, "BE": be_broad[2024]}),
             ("as above, and Spain's 7.5% deduction and Italy's 1.83% payback", {"FR": fr_broad24, "BE": be_broad[2024],
                                                                               "ES": ES_DEDUCTION, "IT": 0.0183})]
    cf = []
    for label, cut in cases:
        new = np.mean([lists[k] * (1 - cut.get(k, 0.0)) for k in lists])
        cf.append({"case": label, "ceiling": float(new), "change": float(new / base - 1)})
        print(f"  {label}: ceiling {new:.3f}, {new / base - 1:+.1%}")
    print(f"  against Portugal's own returns in 2024: {pt24_lo:.1%}-{pt24_hi:.1%} of SNS spending")
    w = {k: v / sum(lists.values()) for k, v in lists.items()}
    print("  weight of each country in the ceiling (its list over the sum): "
          + ", ".join(f"{k} {v:.1%}" for k, v in w.items()))
    figure["ceiling"] = cf
    p25, sp25 = bill.loc[2025, "patients"], bill.loc[2025, "pharmacy"]
    print(f"Patients' co-payments at the pharmacy are a share of the list price; returns go to the SNS only. "
          f"2025: patients EUR {p25:,.1f} million, SNS pharmacy charges {sp25:,.1f} million")
    for r in (0.0, 0.05, 0.10, 0.15):
        print(f"  if returns were {r:.0%} of SNS pharmacy charges: patients' share of the net cost "
              f"{p25 / (p25 + sp25 * (1 - r)):.1%}")
    m = find(r"rembourse ([\d,]+) % des dépenses de médicaments remboursables en ville et dans la liste en sus\. Ce taux "
             r"est de ([\d,]+) % en déduisant les remises", dd, "DREES coverage")
    print(f"  France 2024 (DREES): the public insurer pays {num(m.group(1)):.1f}% of gross spending and "
          f"{num(m.group(2)):.1f}% of spending net of rebates")

    figure["series"] = {"PT": pt_series, "FR": {str(k): v for k, v in fr_series.items()},
                        "FR_igf": {"2023": igf_r23 / igf_g23}, "BE": {str(k): v for k, v in be_series.items()},
                        "BE_broad": {str(k): v for k, v in be_broad.items()},
                        "BE_conv_rate": {str(k): v for k, v in conv_rate.items()}}
    (HERE / "figure_data.json").write_text(json.dumps(figure, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
