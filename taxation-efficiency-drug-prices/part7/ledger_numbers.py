"""Part 7 of the taxation-efficiency-drug-prices series: a ledger of what Portugal's public buyer of
medicines saves, and where the saving comes from.

Every number quoted in the article is printed by this script. Run, after part7/fetch.py has put the
public files in part7/data (the GitHub Actions workflow does this and commits them):

    python ledger_numbers.py > output.txt

The efficiency argument for a tax-financed single buyer says that it pays less for medicines than many
buyers would. Parts 2 to 6 measured the pieces of that claim one at a time. This script puts them in one
ledger for Portugal, in euros a year, and splits the price advantage into the part that Portugal's prices
borrow from other countries through its reference pricing rule and the part that the SNS obtains itself.

Sections
1. Data and coverage.
2. The SNS bill for medicines, 2015-2025, and the industry's returns to the SNS.
3. Where Portugal's price advantage over the United States comes from: other countries' prices (through
   the reference rule), the rule's own premium, and the SNS's own returns, under four US benchmarks.
4. Whether buyer size lowers prices across Europe (EURIPID price levels against population and income).
5. The time account: Portugal's extra wait, the part of it beyond the legal deadlines, and what a delay
   saves the budget and costs patients.
6. Leakage and the contribution to innovation, from Parts 5 and 6.
7. What else a single public payer buys: administration and insurance (OECD health accounts).
8. The ledger: each account in euros a year and per resident, with the uncertainty of its inputs.

Inputs taken from earlier parts are read from their committed figure_data.json and output.txt files.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROOT = HERE.parent

SEED = 20261113
REPS = 5000

# Published figures that are not in a data file. Each is quoted with its source and used as stated.
RAND_GROSS, RAND_NET = 4.22, 3.08          # US brand-name originator prices over 33 OECD countries, 2022
                                           # (Mulcahy, Schwam and Lovejoy 2024), before and after US rebates
LEGAL_INFARMED, LEGAL_NEGOTIATION = 75, 42  # days: Portaria 195-A/2015 (30 working days for negotiation)
HOSPITAL_PROTECTED = (0.6, 0.9)            # assumed share of hospital spending on medicines with no
                                           # generic or biosimilar competition (no public figure)


def section(title: str) -> None:
    print()
    print("=" * 94)
    print(title)
    print("=" * 94)


def text(name: str) -> str:
    return re.sub(r"\s+", " ", gzip.open(DATA / f"{name}.txt.gz", "rt", encoding="utf-8").read())


def fig(part: int) -> dict:
    return json.loads((ROOT / f"part{part}" / "figure_data.json").read_text(encoding="utf-8"))


def out_text(part: int) -> str:
    return (ROOT / f"part{part}" / "output.txt").read_text(encoding="utf-8")


def find(pattern: str, s: str, what: str) -> re.Match:
    m = re.search(pattern, s)
    if not m:
        raise SystemExit(f"{what}: pattern not found ({pattern})")
    return m


def pct(lo, hi) -> str:
    return f"{lo:.1%} to {hi:.1%}"


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def summary(a) -> tuple[float, float, float]:
    a = np.asarray(a, float)
    return float(np.median(a)), float(np.percentile(a, 5)), float(np.percentile(a, 95))


def main() -> None:
    rng = np.random.default_rng(SEED)
    figure: dict = {}

    # ------------------------------------------------------------------ 1
    section("1. Data and coverage")
    hosp = pd.read_csv(DATA / "sns_despesa_com_medicamentos_nos_hospitais_do_sns.csv.gz")
    amb = pd.read_csv(DATA / "sns_despesa_com_medicamentos_no_ambulatorio_sns.csv.gz")
    for d in (hosp, amb):
        d["year"] = d["tempo"].str[:4].astype(int)
    months = hosp.groupby("year")["tempo"].nunique()
    full = [int(y) for y, n in months.items() if n == 12]
    print(f"SNS transparency portal: hospital SNS charges and pharmacy charges by region and month, "
          f"{hosp['tempo'].min()} to {hosp['tempo'].max()}; complete years {full[0]}-{full[-1]}; "
          f"{months.iloc[-1]} months of {months.index[-1]}")
    inf = text("infarmed_note_2025")
    m = find(r"(anual estimado de 500-600 milh\w+, apenas cerca de 40% ficam refletidos)", inf, "Infarmed note")
    print(f"Infarmed note of 15 November 2025: industry returns to the SNS, '{m.group(1)} ...'")
    returns_lo, returns_hi = 500.0, 600.0
    ap = text("apifarma_boletim_2025_12")
    m = find(r"Quota no M\. Total \(%\) Valor unitário M\. Concorrencial (\d+,\d)% (\d+,\d)%", ap, "APIFARMA shares")
    conc_value, conc_volume = (float(m.group(i).replace(",", ".")) / 100 for i in (1, 2))
    mv = find(r"M\. Valor \(M€\) [\d,]+ [\d,]+% ([\d\.]+,\d) ", ap, "APIFARMA market value")
    pharmacy_market = float(mv.group(1).replace(".", "").replace(",", "."))
    mc = find(r"homogéneos, totalizou, em 2025, vendas de ([\d\.]+,\d) M€", ap, "APIFARMA competitive market")
    conc_eur = float(mc.group(1).replace(".", "").replace(",", "."))
    print(f"APIFARMA, December 2025: pharmacy market 2025 at ex-factory prices EUR {pharmacy_market:,.1f} million; "
          f"molecules with generic competition (homogeneous groups) EUR {conc_eur:,.1f} million, {conc_value:.1%} of "
          f"value (check {conc_eur / pharmacy_market:.1%}) and {conc_volume:.1%} of packs")
    agr = text("apifarma_agreement_2025")
    find(r"Despesa L.quida Medicamentos = Despesa Total", agr, "agreement")
    print("Agreement State-industry 2025: net spending = total SNS spending minus the extraordinary contribution and "
          "the amounts paid under financing contracts (formula found in the text)")
    sha = pd.read_csv(DATA / "oecd_sha.csv.gz")
    print(f"OECD health accounts (SHA): {sha['REF_AREA'].nunique()} areas, {sha['TIME_PERIOD'].min()}-"
          f"{sha['TIME_PERIOD'].max()}")

    # ------------------------------------------------------------------ 2
    section("2. The SNS bill for medicines")
    h = hosp.groupby("year")["encargos_sns_hospitalar"].sum() / 1e6
    a = amb.groupby("year")[["encargos_sns_ambulatorio", "encargos_utentes_ambulatorio", "valor_pvp_ambulatorio",
                              "encargos_sns_genericos"]].sum() / 1e6
    bill = pd.DataFrame({"hospital": h, "pharmacy_sns": a["encargos_sns_ambulatorio"],
                         "patients": a["encargos_utentes_ambulatorio"], "retail": a["valor_pvp_ambulatorio"]})
    bill["sns"] = bill["hospital"] + bill["pharmacy_sns"]
    bill = bill.loc[[y for y in full if y >= 2015]]
    print("EUR million: hospital SNS charges | pharmacy SNS charges | SNS total | patients' charges at the pharmacy")
    for y, r in bill.iterrows():
        print(f"  {y}  {r.hospital:8,.1f}  {r.pharmacy_sns:8,.1f}  {r.sns:8,.1f}  {r.patients:7,.1f}")
    last = full[-1]
    b = bill.loc[last]
    prev = bill.loc[last - 1]
    g10 = (b.sns / bill.loc[last - 10].sns) ** 0.1 - 1
    print(f"{last}: SNS EUR {b.sns:,.1f} million (+{b.sns / prev.sns - 1:.1%}; +{b.sns - prev.sns:,.1f}); hospital "
          f"{b.hospital:,.1f} (+{b.hospital - prev.hospital:,.1f}), pharmacy {b.pharmacy_sns:,.1f} "
          f"(+{b.pharmacy_sns - prev.pharmacy_sns:,.1f}); average growth {last - 10}-{last} {g10:.1%} a year")
    print(f"  patients paid {b.patients / b.retail:.1%} of the retail value of SNS-reimbursed pharmacy medicines "
          f"(EUR {b.patients:,.1f} million)")
    r_lo, r_hi = returns_lo / b.sns, returns_hi / b.sns
    print(f"Industry returns, Infarmed estimate for {last}: EUR {returns_lo:.0f}-{returns_hi:.0f} million = "
          f"{pct(r_lo, r_hi)} of SNS charges; net SNS spending EUR {b.sns - returns_hi:,.0f}-{b.sns - returns_lo:,.0f} million")
    prot_pharm = 1 - conc_value
    print(f"Share of pharmacy sales value with no generic competition (APIFARMA): {prot_pharm:.1%}")
    s_prot = {hp: b.hospital * hp + b.pharmacy_sns * prot_pharm for hp in (HOSPITAL_PROTECTED[0], 0.75, HOSPITAL_PROTECTED[1])}
    print("SNS spending on medicines without generic competition (hospital share assumed, pharmacy share from APIFARMA):")
    for hp, v in s_prot.items():
        print(f"  hospital share {hp:.2f}: EUR {v:,.0f} million ({v / b.sns:.1%} of SNS charges); "
              f"returns as a share of it if all returns came from these medicines {returns_lo / v:.1%}-{returns_hi / v:.1%}")
    figure["bill"] = {"years": [int(y) for y in bill.index], "hospital": bill.hospital.round(1).tolist(),
                      "pharmacy": bill.pharmacy_sns.round(1).tolist(), "patients": bill.patients.round(1).tolist()}

    # ------------------------------------------------------------------ 3
    section("3. Where Portugal's price advantage over the United States comes from")
    f2 = fig(2)
    us_fr_gross = f2["baskets"]["single-source"]["F"]
    us_fr_net = f2["rebates"]["grid"]["RAND 2024"][0]   # US net of RAND's rebate share, French list prices
    rules = fig(3)["rules"]
    c_all = next(r for r in rules if r["rule"].startswith("average of the four"))["vs_fr"]
    m = find(r"pharmacy medicines:.*?Portuguese ceiling \(average\)\s+([\d.]+)", out_text(3).replace("\n", " "), "Part 3")
    c_pharm = float(m.group(1))
    print(f"Part 2: US over French prices, single-source oral medicines, Fisher index {us_fr_gross:.3f} (US gross); "
          f"{us_fr_net:.3f} with US rebates of 37.2% (RAND 2024) and French list prices")
    print(f"Part 3: Portugal's reference ceiling over the French price {c_all:.3f} (presentations priced by all four "
          f"countries), {c_pharm:.3f} (all pharmacy medicines priced)")
    print(f"RAND 2024: US over 33 OECD countries, brand-name originators, {RAND_GROSS:.2f} gross, {RAND_NET:.2f} net")
    benchmarks = [
        ("Part 2, US gross / France", us_fr_gross, c_all),
        ("Part 2, US net / France", us_fr_net, c_all),
        ("RAND, US gross / 33 countries", RAND_GROSS, 1.0),
        ("RAND, US net / 33 countries", RAND_NET, 1.0),
    ]
    print("Log decomposition of the US price over Portugal's net price, ln(US/PT net) = ln(US/reference) "
          "- ln(PT list/reference) - ln(1 - own return rate):")
    rows = []
    for lab, k_ref, c in benchmarks:
        for r_own in (r_lo, r_hi):
            total = np.log(k_ref) - np.log(c) - np.log(1 - r_own)
            borrowed = np.log(k_ref) / total
            premium = -np.log(c) / total
            own = -np.log(1 - r_own) / total
            print(f"  {lab:31s} own returns {r_own:.1%}: US/PT net {np.exp(total):5.2f}; reference countries' prices "
                  f"{borrowed:6.1%}, Portugal's formula premium {premium:+6.1%}, SNS's own returns {own:5.1%}")
            rows.append({"benchmark": lab, "r": r_own, "borrowed": borrowed, "premium": premium, "own": own,
                         "us_pt_net": float(np.exp(total))})
    own_shares = [x["own"] for x in rows]
    print(f"  across the benchmarks, the SNS's own returns explain {min(own_shares):.1%} to {max(own_shares):.1%} "
          f"of the log gap; other countries' prices {min(x['borrowed'] for x in rows):.1%} to "
          f"{max(x['borrowed'] for x in rows):.1%}")
    r_gen = returns_hi / s_prot[HOSPITAL_PROTECTED[0]]
    gen = []
    for lab, k_ref, c in benchmarks:
        total = np.log(k_ref) - np.log(c) - np.log(1 - r_gen)
        gen.append(-np.log(1 - r_gen) / total)
    print(f"  most generous case, all of the upper estimate of returns on the smallest protected base (return rate "
          f"{r_gen:.1%}): own returns {min(gen):.1%} to {max(gen):.1%} of the log gap")
    hp = 0.75
    sp = s_prot[hp]
    print(f"In euros, for SNS spending on medicines without generic competition (hospital share {hp:.2f}: "
          f"EUR {sp:,.0f} million at Portuguese list prices):")
    for lab, k_ref, c in benchmarks:
        k = k_ref / c
        borrowed_eur = sp * (k - 1)
        own_eur = sp * r_lo, sp * r_hi
        print(f"  {lab:31s} at US prices EUR {sp * k:8,.0f} million; saving EUR {borrowed_eur + own_eur[0]:7,.0f}-"
              f"{borrowed_eur + own_eur[1]:7,.0f} million, of which own returns {own_eur[0]:4,.0f}-{own_eur[1]:4,.0f} "
              f"({own_eur[0] / (borrowed_eur + own_eur[0]):.1%}-{own_eur[1] / (borrowed_eur + own_eur[1]):.1%})")
    figure["decomposition"] = rows
    r_mid = (r_lo + r_hi) / 2
    print(f"If the reference countries' list prices rose and every SNS list price followed, the own return rate that "
          f"would keep Portugal's net spending where it is (now {r_mid:.1%}):")
    for up in (0.05, 0.10, 0.25):
        need = 1 - (1 - r_mid) / (1 + up)
        print(f"  reference prices +{up:.0%}: {need:.1%} of spending, EUR {need * b.sns:,.0f} million at {last} spending "
              f"(now EUR {r_mid * b.sns:,.0f} million)")

    # ------------------------------------------------------------------ 4
    section("4. Does buyer size lower prices? EURIPID price levels across Europe")
    p4 = ROOT / "part4" / "data"
    pli = pd.read_csv(p4 / "euripid_pli_2017.csv.gz")
    pop = pd.read_csv(p4 / "eurostat_population.csv.gz")
    pop = pop[pop["time"] == 2025][["geo", "value"]].rename(columns={"value": "pop"})
    aic = pd.read_csv(p4 / "eurostat_aic_pps.csv.gz")
    aic = aic[(aic["time"] == 2024) & (aic["na_item"] == "VI_PPS_EU27_2020_HAB")][["geo", "value"]].rename(columns={"value": "aic"})
    d = pli.merge(pop, on="geo").merge(aic, on="geo").dropna(subset=["pli_last", "pop", "aic"]).sort_values("geo")
    d = d.reset_index(drop=True)
    y = np.log(d["pli_last"].to_numpy())
    lp = np.log10(d["pop"].to_numpy())
    la = np.log(d["aic"].to_numpy())
    X = np.column_stack([np.ones(len(d)), lp, la])
    beta = ols(y, X)
    fitted = X @ beta
    r2 = 1 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    boots = []
    for _ in range(REPS):
        i = rng.integers(0, len(d), len(d))
        if np.unique(lp[i]).size > 2:
            boots.append(ols(y[i], X[i]))
    boots = np.array(boots)
    ci_pop = np.percentile(boots[:, 1], [2.5, 97.5])
    ci_aic = np.percentile(boots[:, 2], [2.5, 97.5])
    print(f"Countries with a EURIPID price level (2017, all medicines), population (2025) and AIC per head (2024): {len(d)}")
    print(f"  ln(price level) = {beta[0]:.3f} {beta[1]:+.3f} log10(population) {beta[2]:+.3f} ln(AIC per head); R2 {r2:.2f}")
    print(f"  population coefficient {beta[1]:+.3f} [{ci_pop[0]:+.3f}, {ci_pop[1]:+.3f}]: a market ten times larger, "
          f"same income, pays {np.exp(beta[1]) - 1:+.1%} [{np.exp(ci_pop[0]) - 1:+.1%}, {np.exp(ci_pop[1]) - 1:+.1%}]")
    print(f"  income coefficient {beta[2]:+.3f} [{ci_aic[0]:+.3f}, {ci_aic[1]:+.3f}]: 10% higher AIC per head, "
          f"{np.exp(beta[2] * np.log(1.1)) - 1:+.1%} on the price level")
    r_size = float(pd.Series(y).corr(pd.Series(lp)))
    r_inc = float(pd.Series(y).corr(pd.Series(la)))
    print(f"  simple correlations with ln(price level): log population {r_size:+.2f}, ln AIC {r_inc:+.2f}")
    ipt = int(d.index[d["geo"] == "PT"][0])
    print(f"  Portugal: price level {d.loc[ipt, 'pli_last']:.1f}, predicted {np.exp(fitted[ipt]):.1f} "
          f"(residual {np.exp(y[ipt] - fitted[ipt]) - 1:+.1%}); population {d.loc[ipt, 'pop'] / 1e6:.1f} million, "
          f"AIC {d.loc[ipt, 'aic']:.0f}")
    size_cf = {}
    for geo in ("ES", "FR", "UK"):
        j = int(d.index[d["geo"] == geo][0])
        delta = lp[j] - lp[ipt]
        eff = np.exp(beta[1] * delta) - 1
        effb = np.exp(boots[:, 1] * delta) - 1
        lo, hi = np.percentile(effb, [2.5, 97.5])
        size_cf[geo] = (delta, boots[:, 1])
        print(f"  Portugal with the population of {geo} ({d.loc[j, 'pop'] / 1e6:.1f} million), same income: price level "
              f"{eff:+.1%} [{lo:+.1%}, {hi:+.1%}]")
    fig_pts = []
    for i, r in d.iterrows():
        fig_pts.append({"geo": r["geo"], "pli": float(r["pli_last"]), "pop": float(r["pop"]), "aic": float(r["aic"]),
                        "part_pop": float(y[i] - beta[2] * la[i] - beta[0]),
                        "part_aic": float(y[i] - beta[1] * lp[i] - beta[0])})
    figure["size"] = {"beta": [float(x) for x in beta], "points": fig_pts}

    # ------------------------------------------------------------------ 5
    section("5. The time account")
    o4 = out_text(4)
    eu = float(find(r"time to availability (\d+) days", o4, "Part 4 EU").group(1))
    pt = float(find(r"Portugal: mean (\d+) days", o4, "Part 4 PT").group(1))
    f4 = fig(4)
    resid = {mm["model"]: mm["pt_resid"] for mm in f4["models"]}
    print(f"Part 4: Portugal {pt:.0f} days from EU authorisation to availability, EU27 {eu:.0f}: {pt - eu:.0f} days longer")
    print(f"  Portugal's excess over the models of Part 4: price + AIC {resid['price + AIC']:.0f} days; size + distance "
          f"+ AIC + price {resid['size + distance + AIC + price']:.0f}; with the cancer-medicine price level "
          f"{resid['size + distance + AIC + price L']:.0f}")
    tr = pd.DataFrame(f4["tracker"])
    over_inf = (tr["infarmed"] - LEGAL_INFARMED).clip(lower=0)
    over_neg = (tr["negotiation"] - LEGAL_NEGOTIATION).clip(lower=0)
    print(f"  approved processes in Infarmed's tracker: {len(tr)}; mean days beyond the legal deadlines: Infarmed "
          f"{over_inf.mean():.0f}, negotiation {over_neg.mean():.0f}, together {over_inf.mean() + over_neg.mean():.0f}; "
          f"median together {(over_inf + over_neg).median():.0f}")
    days_lo, days_hi = resid["size + distance + AIC + price L"], pt - eu
    growth_h = b.hospital - prev.hospital
    print(f"A delay defers spending on the delayed medicines. Scale: if the first-year spending of a year's new medicines "
          f"equalled the whole {last} increase in hospital spending (EUR {growth_h:,.1f} million, an upper end), a "
          f"delay of {days_lo:.0f}-{days_hi:.0f} days would defer EUR {growth_h * days_lo / 365:,.0f}-"
          f"{growth_h * days_hi / 365:,.0f} million a year")
    print("  For a medicine the SNS agreed to pay for, the price is at most what the SNS judged its health effect "
          "worth; the health forgone during the delay is worth at least the spending deferred.")

    # ------------------------------------------------------------------ 6
    section("6. Leakage and the contribution to innovation (Parts 5 and 6)")
    o5 = out_text(5)
    m = find(r"2021: (\d+) intentions notified, (\d+) prohibited \(\d+%\); ([\d,]+) packs, of which ([\d,]+) prohibited",
             o5, "Part 5")
    packs, banned = (int(m.group(i).replace(",", "")) for i in (3, 4))
    print(f"Part 5, 2021: export intentions for {packs:,} packs, {banned:,} prohibited, {packs - banned:,} allowed")
    m = find(r"the ten together: ([\d,]+) packs, ([\d.]+)% of the (\d+) million packs", o5, "Part 5 2024")
    print(f"Part 5, 2024: ten most exported listed medicines {m.group(1)} packs, {m.group(2)}% of SNS-reimbursed "
          f"pharmacy packs")
    o6 = out_text(6)
    pt_world = float(find(r"Portugal's market \(2024\) as a share of the world market \(2025\): ([\d.]+)%", o6,
                          "Part 6").group(1)) / 100
    world = float(find(r"world prescription market 2025 EUR ([\d,]+) million", o6, "Part 6 world").group(1).replace(",", ""))
    na_share = float(find(r"North America ([\d.]+)%", o6, "Part 6 NA").group(1)) / 100
    wb = pd.read_csv(ROOT / "part6" / "data" / "worldbank.csv.gz")
    gdp = wb[(wb["indicator"] == "NY.GDP.MKTP.CD") & (wb["year"] == 2024)].set_index("iso3")["value"]
    gdp_world = float(gdp["WLD"])
    gdp_pt = float(gdp["PRT"])
    gdp_na = float(gdp["USA"] + gdp["CAN"])
    print(f"Portugal's share of the world prescription market {pt_world:.2%}; of world GDP (2024, current dollars) "
          f"{gdp_pt / gdp_world:.2%}; ratio {pt_world / (gdp_pt / gdp_world):.2f}")
    pt_rest = pt_world / (1 - na_share)
    gdp_rest = gdp_pt / (gdp_world - gdp_na)
    print(f"  outside North America: Portugal {pt_rest:.2%} of the market and {gdp_rest:.2%} of GDP; ratio "
          f"{pt_rest / gdp_rest:.2f}")
    eur = pd.DataFrame(fig(6)["europe"])
    ptm = eur.loc[eur["iso2"] == "PT", "market_gdp"].iloc[0]
    print(f"Part 6: Portugal's pharmaceutical market {ptm:.2%} of GDP (rank {int((eur['market_gdp'] > ptm).sum()) + 1} of "
          f"{len(eur)} European countries); world market EUR {world:,.0f} million")

    # ------------------------------------------------------------------ 7
    section("7. What else a single public payer buys: administration and insurance (OECD health accounts)")
    x = sha[sha["UNIT_MEASURE"] == "XDC"]
    piv = x.pivot_table(index=["REF_AREA", "TIME_PERIOD"], columns=["FUNCTION", "FINANCING_SCHEME"], values="OBS_VALUE")
    names = sha.drop_duplicates("REF_AREA").set_index("REF_AREA")["Reference area"]
    rows7 = []
    for area, g in piv.groupby(level=0):
        g = g.droplevel(0)
        ok = g[[("_T", "_T"), ("HC7", "_T")]].dropna() if ("HC7", "_T") in g.columns else pd.DataFrame()
        if ok.empty:
            continue
        yr = int(ok.index.max())
        row = {"area": area, "name": names.get(area, area), "year": yr,
               "admin": g.loc[yr, ("HC7", "_T")] / g.loc[yr, ("_T", "_T")]}
        if ("HC51", "_T") in g.columns and pd.notna(g.loc[yr, ("HC51", "_T")]):
            row["pharma_public"] = g.loc[yr, ("HC51", "HF1")] / g.loc[yr, ("HC51", "_T")] if ("HC51", "HF1") in g.columns else np.nan
            row["pharma_oop"] = g.loc[yr, ("HC51", "HF3")] / g.loc[yr, ("HC51", "_T")] if ("HC51", "HF3") in g.columns else np.nan
            row["pharma_che"] = g.loc[yr, ("HC51", "_T")] / g.loc[yr, ("_T", "_T")]
        row["che"] = g.loc[yr, ("_T", "_T")]
        rows7.append(row)
    t7 = pd.DataFrame(rows7)
    countries = t7[t7["area"].str.fullmatch(r"[A-Z]{3}") & ~t7["area"].isin(["OEU", "OED", "EU2", "EA2", "E22"])]
    countries = countries[countries["name"].str.len() < 30]
    print(f"Countries with governance and administration (HC.7) and current health expenditure: {len(countries)}")
    sel = ["USA", "PRT", "ESP", "FRA", "ITA", "BEL", "DEU", "NLD", "GBR", "DNK", "SWE", "CHE"]
    print("  country (year): administration % of current health spending | pharmaceuticals % of it | share of "
          "pharmaceutical spending paid by government/compulsory schemes | by households out of pocket")
    for c in sel:
        r = countries[countries["area"] == c]
        if r.empty:
            continue
        r = r.iloc[0]
        print(f"    {r['name'][:16]:16s} ({r['year']}): {r['admin']:5.1%} | {r.get('pharma_che', np.nan):5.1%} | "
              f"{r.get('pharma_public', np.nan):5.1%} | {r.get('pharma_oop', np.nan):5.1%}")
    med_admin = countries["admin"].median()
    print(f"  median across {len(countries)} countries: administration {med_admin:.1%}; public share of pharmaceutical "
          f"spending {countries['pharma_public'].median():.1%} (n={countries['pharma_public'].notna().sum()}); "
          f"out of pocket {countries['pharma_oop'].median():.1%}")
    rpt = countries[countries["area"] == "PRT"].iloc[0]
    rus = countries[countries["area"] == "USA"].iloc[0]
    rank_oop = int((countries["pharma_oop"] > rpt["pharma_oop"]).sum()) + 1
    print(f"  Portugal: out-of-pocket share of pharmaceutical spending rank {rank_oop} of "
          f"{countries['pharma_oop'].notna().sum()} (1 = highest)")
    admin_gap = (rus["admin"] - rpt["admin"]) * rpt["che"]
    admin_gap_med = (med_admin - rpt["admin"]) * rpt["che"]
    print(f"  Portugal's current health spending {rpt['year']}: EUR {rpt['che']:,.0f} million; at the US administrative share "
          f"it would spend EUR {admin_gap:,.0f} million more on administration; at the median share EUR {admin_gap_med:,.0f} million more")
    figure["sha"] = countries[["area", "name", "year", "admin", "pharma_public", "pharma_oop"]].round(4).to_dict("records")

    # ------------------------------------------------------------------ 8
    section("8. The ledger: euros a year, Portugal, around 2025")
    pop_pt = float(pop.loc[pop["geo"] == "PT", "pop"].iloc[0])
    k_draw = rng.uniform(RAND_NET, us_fr_gross / c_all, REPS)
    hp_draw = rng.uniform(*HOSPITAL_PROTECTED, REPS)
    ret_draw = rng.uniform(returns_lo, returns_hi, REPS)
    days_draw = rng.uniform(days_lo, days_hi, REPS)
    delta_fr, bpop_boot = size_cf["FR"]
    bpop_draw = rng.choice(bpop_boot, REPS)
    sprot = b.hospital * hp_draw + b.pharmacy_sns * prot_pharm
    r_draw = ret_draw / b.sns
    borrowed = sprot * (k_draw - 1)
    own_prot = sprot * r_draw
    size_fr = sprot * (1 - np.exp(bpop_draw * delta_fr))
    delay = growth_h * days_draw / 365
    lines = [
        ("Saving against US prices on medicines without generic competition", borrowed + own_prot),
        ("  of which: other countries' prices, through the reference rule", borrowed),
        ("  of which: the SNS's own returns on those medicines", own_prot),
        ("Industry returns to the SNS, all medicines (Infarmed)", ret_draw),
        ("What France's size would be worth at Portugal's income (price level)", size_fr),
        ("Spending deferred by the extra wait (upper end) = health forgone (lower bound)", delay),
        ("Administration: Portugal's share against the US share of health spending", np.full(REPS, admin_gap)),
    ]
    print(f"Draws {REPS}: US over Portuguese list price uniform {RAND_NET:.2f}-{us_fr_gross / c_all:.2f}; hospital share "
          f"without competition uniform {HOSPITAL_PROTECTED[0]}-{HOSPITAL_PROTECTED[1]}; returns uniform "
          f"{returns_lo:.0f}-{returns_hi:.0f}; extra days uniform {days_lo:.0f}-{days_hi:.0f}; population coefficient "
          f"from the bootstrap of section 4. Population {pop_pt / 1e6:.2f} million.")
    print("  account: median [5th, 95th percentile], EUR million a year | EUR per resident")
    ledger = []
    for lab, v in lines:
        med, lo, hi = summary(v)
        print(f"  {lab}: {med:,.0f} [{lo:,.0f}, {hi:,.0f}] | {med * 1e6 / pop_pt:,.0f} [{lo * 1e6 / pop_pt:,.0f}, "
              f"{hi * 1e6 / pop_pt:,.0f}]")
        ledger.append({"label": lab, "med": med, "lo": lo, "hi": hi})
    own_share = own_prot / (borrowed + own_prot)
    med, lo, hi = summary(own_share)
    print(f"  share of the saving against US prices that is the SNS's own: {med:.1%} [{lo:.1%}, {hi:.1%}]")
    ratio = summary(ret_draw / size_fr)
    print(f"  industry returns over what France's size would be worth: {ratio[0]:.2f} [{ratio[1]:.2f}, {ratio[2]:.2f}]")
    print(f"  administration gap over industry returns: {admin_gap / returns_hi:.1f} to {admin_gap / returns_lo:.1f}")
    figure["ledger"] = ledger
    figure["pop_pt"] = pop_pt

    (HERE / "figure_data.json").write_text(json.dumps(figure, indent=1, sort_keys=True, default=float) + "\n",
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
