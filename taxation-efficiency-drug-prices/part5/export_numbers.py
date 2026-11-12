"""Part 5 of the taxation-efficiency-drug-prices series: medicines that leave Portugal through parallel
export, and the export controls that try to keep them.

Every number quoted in the article is printed by this script. Run, after part5/fetch.py has put the
public files in part5/data (the GitHub Actions workflow does this and commits them):

    python export_numbers.py > output.txt

The question is whether Portugal's low regulated prices make it a source of parallel exports, whether
those exports are what empties Portuguese pharmacies, and what the export controls Infarmed has used
since 2021 actually cover.

Sections
1. Data and coverage.
2. How much leaves, and to where: Infarmed's own figures on export intentions (2021, 2024, 2025) and
   Eurostat's trade in medicaments.
3. The European register of parallel distribution: where notices send medicines, which countries
   distributors name as sources when they name few, and how that lines up with price levels.
4. What Portugal restricts: the prior-notification lists and the suspended-export lists, 2021-2026,
   their size, their composition and how often the same medicines appear on both.
5. The monthly prohibitions of 2026, with Infarmed's indicators: export intentions and exports against
   the quantity the company sold, and Infarmed's prescription- and consumption-satisfaction ratios.
6. Why medicines run short in Portugal, by the reasons companies report.
7. The price of the exported brands: Part 3's rebuilt Portuguese ceilings for the brands on the list.
"""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PART4 = HERE.parent / "part4" / "data"
PART3 = HERE.parent / "part3"
SEED = 20261112
REPS = 5000

pd.set_option("display.width", 200)

# Names of member states as the EMA register writes them, with Eurostat codes.
MS = {
    "Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Croatia": "HR", "Cyprus": "CY", "Czech Republic": "CZ",
    "Czechia": "CZ", "Denmark": "DK", "The Netherlands": "NL", "Estonia": "EE", "Finland": "FI", "France": "FR", "Germany": "DE",
    "Greece": "EL", "Hungary": "HU", "Iceland": "IS", "Ireland": "IE", "Italy": "IT", "Latvia": "LV",
    "Liechtenstein": "LI", "Lithuania": "LT", "Luxembourg": "LU", "Malta": "MT", "Netherlands": "NL",
    "Norway": "NO", "Poland": "PL", "Portugal": "PT", "Romania": "RO", "Slovakia": "SK", "Slovenia": "SI",
    "Spain": "ES", "Sweden": "SE", "UK (Northern Ireland)": "XI", "United Kingdom": "UK",
}


def section(title: str) -> None:
    print()
    print("=" * 94)
    print(title)
    print("=" * 94)


def text(name: str) -> str:
    with gzip.open(DATA / f"{name}.txt.gz", "rt", encoding="utf-8") as f:
        return re.sub(r"\s+", " ", f.read())


def grab(t: str, pattern: str, n: int = 1):
    m = re.search(pattern, t, re.S)
    if not m:
        raise SystemExit(f"pattern not found: {pattern}")
    vals = tuple(int(v.replace(" ", "").replace(".", "")) for v in m.groups())
    return vals[0] if n == 1 else vals


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


def pearson(x, y) -> float:
    return float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])


def spearman(x, y) -> float:
    return pearson(pd.Series(x).rank(), pd.Series(y).rank())


def boot_corr(x, y, rng, fn=spearman) -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = []
    for _ in range(REPS):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            out.append(fn(x[i], y[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def generic_named(name: str, dci: str) -> bool:
    """A presentation sold under the name of its active substance (the usual form of a generic in
    Portugal, 'Quetiapina Zentiva'), rather than under a brand."""
    n, d = fold(name), fold(dci)
    if not n or not d:
        return False
    first = d.split()[0]
    return len(first) > 3 and n.startswith(first[: max(5, len(first) - 2)])


def ever_names(lists: pd.DataFrame, motive: str) -> set[str]:
    """Names of the presentations that appear on any prior-notification list with the given reason."""
    m = lists[(lists["type"].isin(["lnp", "prohibited"])) & lists["motive"].fillna("").str.startswith(motive)]
    return set(m["name"].dropna())


# ------------------------------------------------------------------------------------------ main
def main() -> None:
    rng = np.random.default_rng(SEED)
    figure: dict = {}

    # ------------------------------------------------------------------------------------ 1
    section("1. Data and coverage")
    docs = pd.read_csv(DATA / "pt_list_documents.csv.gz", dtype=str)
    rows = pd.read_csv(DATA / "pt_list_rows.csv.gz", dtype=str)
    pdn = pd.read_csv(DATA / "ema_pd_notices.csv.gz", dtype=str)
    com = pd.read_csv(DATA / "comext_pt_3004.csv.gz", keep_default_na=False)  # "NA" is Namibia
    print(f"Infarmed export-control documents found: {len(docs)} "
          f"({docs.groupby(['type', 'kind']).size().to_dict()})")
    ls = docs[docs["kind"] == "list"]
    bad = ls[ls["n_rows"].astype(int) == 0]
    print(f"  lists: {len(ls)} found, {len(ls) - len(bad)} read; the 2021 deliberation's two annexes read as the first lists; "
          f"deliberations kept as dates only")
    print("  could not be downloaded (connection refused by Infarmed's server): "
          + ", ".join(f"{t} {e}" for t, e in zip(bad["type"], bad["effective"])))
    print(f"  parsed rows (one per presentation and document): {len(rows)}")
    print(f"EMA register of parallel distribution notices: {len(pdn)} notices")
    print(f"Comext: Portugal's trade in medicaments (HS 3004), {com['year'].min()}-{com['year'].max()}")

    # ------------------------------------------------------------------------------------ 2
    section("2. How much leaves, and to where")
    r21, r24, r25 = text("pt_availability_2021"), text("pt_availability_2024"), text("pt_availability_2025")
    p21 = grab(r21, r"em 2021 foram proibidos (\d[\d ]*\d) pedidos de exportação")
    p24, a24 = grab(r24, r"Durante o ano de 2024, foram proibidos (\d[\d ]*\d) pedidos de intenção de exportação e autorizados (\d[\d ]*\d) pedidos", 2)
    p25, a25 = grab(r25, r"Durante o ano de 2025, foram proibidos (\d[\d ]*\d) pedidos de intenção de exportação e autorizados (\d[\d ]*\d) pedidos", 2)
    drop = grab(r25, r"decréscimo de (\d+)% no número de intenções de exportação")
    print("Requests to export medicines on the prior-notification list (Infarmed, SIEXP):")
    n21, k21, pk21 = grab(r21, r"foram notificadas (\d[\d ]*\d) intenções de exportação.*?representavam um total de "
                                r"(\d[\d ]*\d) embalagens,\s+das quais foram proibidas \d+% \((\d[\d ]*\d)\)", 3)
    print(f"  2021: {n21} intentions notified, {p21} prohibited ({p21 / n21:.0%}); {k21:,} packs, of which {pk21:,} "
          f"prohibited ({pk21 / k21:.0%})")
    print(f"  2024: {p24} prohibited, {a24} authorised ({p24 / (p24 + a24):.0%} prohibited)")
    print(f"  2025: {p25} prohibited, {a25} authorised ({p25 / (p25 + a25):.0%} prohibited); "
          f"Infarmed reports a {drop}% fall in export intentions, which it attributes possibly to its meetings and "
          f"thematic inspections with the sector")
    print(f"  total requests 2024 {p24 + a24}, 2025 {p25 + a25} ({(p25 + a25) / (p24 + a24) - 1:+.0%})")
    for year, t in ((2024, r24), (2025, r25)):
        seg = t[t.find("Top 10 dos medicamentos"):][:900]
        items = re.findall(r"(\d{7}) ([A-Za-zÀ-ÿ\-\. ]+?) (\d{1,3}(?: \d{3}(?!\d))+|\d{2,3})(?!\d)", seg)[:10]
        print(f"  top exported LNP medicines {year} (packs): " +
              "; ".join(f"{n.strip()} {int(q.replace(' ', '')):,}" for _, n, q in items))
        top10 = sum(int(q.replace(" ", "")) for _, _, q in items)
        packs = float(re.search(rf"{year}: retail value [\d.]+ bn euros, (\d+) million packs",
                                (HERE.parent / "part3" / "output.txt").read_text(encoding="utf-8")).group(1)) * 1e6
        print(f"    the ten together: {top10:,} packs, {top10 / packs:.1%} of the {packs / 1e6:.0f} million packs of SNS-reimbursed "
              f"medicines dispensed by community pharmacies that year (Part 3, SNS transparency portal)")
    # Destinations as printed in the reports (2021 as a table; 2024 and 2025 as charts).
    dest21 = dict((c, int(p)) for c, p in re.findall(r"(Alemanha|Reino Unido|França|Holanda|República Checa|Irlanda|Angola|Dinamarca|Hungria|Cabo Verde) (\d+)%",
                                                      r21[r21.find("Top 10 – países de destino"):]))
    print(f"  destinations 2021 (share of export intentions): {dest21}")
    # 2024 and 2025: pie charts, whose labels the text layout scatters. The shares below were read off
    # the charts; the check is that the percentages printed on each chart are exactly these.
    charts = {2024: (r24, {"Alemanha": 39, "Reino Unido": 12, "República Checa": 10, "Holanda": 8, "Angola": 7,
                           "Outros": 7, "Itália": 4, "França": 4, "Dinamarca": 4, "Polónia": 3, "Hungria": 2}),
              2025: (r25, {"Alemanha": 35, "Chéquia": 15, "Reino Unido": 14, "Dinamarca": 8, "Polónia": 7, "Outros": 21})}
    for year, (t, shares) in charts.items():
        seg = t[t.find("Países de destino de exportação"):]
        seg = seg[:re.search(r"Comissão Nacional|Articulação com", seg).start()]
        printed = sorted(int(x) for x in re.findall(r"(\d+)%", seg))
        if printed != sorted(shares.values()) or any(n not in seg for n in shares):
            raise SystemExit(f"destination chart {year} changed: {printed}")
        print(f"  destinations {year} (share of exports, Infarmed's chart): {shares}")
    exp = com[(com["flow"] == "export") & (com["indicator"] == "VALUE_IN_EUROS")]
    imp = com[(com["flow"] == "import") & (com["indicator"] == "VALUE_IN_EUROS")]
    tot = exp[exp["partner"] == "WORLD"].set_index("year")["value"]
    print("Eurostat Comext, Portugal's exports of medicaments in measured doses (HS 3004), EUR million:")
    print("  world: " + ", ".join(f"{y} {v / 1e6:,.0f}" for y, v in tot.items() if y in (2015, 2019, 2021, 2023, 2024, 2025)))
    timp = imp[imp["partner"] == "WORLD"].set_index("year")["value"]
    print("  imports, world: " + ", ".join(f"{y} {v / 1e6:,.0f}" for y, v in timp.items() if y in (2015, 2019, 2021, 2023, 2024, 2025)))
    last = exp[exp["year"] == 2025].set_index("partner")["value"]
    first = exp[exp["year"] == 2015].set_index("partner")["value"]
    top = last.drop([p for p in last.index if not re.fullmatch(r"[A-Z]{2}", p)], errors="ignore").sort_values(ascending=False).head(12)
    print("  largest destinations in 2025 (EUR million, share of world, 2015 value):")
    for p, v in top.items():
        print(f"    {p} {v / 1e6:7.1f}  {v / tot[2025]:5.1%}   2015 {first.get(p, np.nan) / 1e6:7.1f}")
    figure["comext"] = {p: [round(float(exp[(exp['partner'] == p) & (exp['year'] == y)]['value'].sum()) / 1e6, 1)
                            for y in range(2015, 2026)] for p in ["DE", "FR", "BE", "ES", "CZ", "NL", "DK", "UK", "GB"]}

    # ------------------------------------------------------------------------------------ 3
    section("3. The European register of parallel distribution notices")
    pdn["year"] = pd.to_datetime(pdn["date_notice"], format="%d/%m/%Y", errors="coerce").dt.year
    print(f"Notices: {len(pdn)}; status {pdn['status'].value_counts().to_dict()}")
    print(f"  years of notice: {int(pdn['year'].min())}-{int(pdn['year'].max())}; products {pdn['product'].nunique()}, "
          f"distributors {pdn['distributor'].nunique()}; notices per year 2021-2026: "
          + ", ".join(f"{int(y)} {n}" for y, n in pdn['year'].value_counts().sort_index().items() if y >= 2021))
    act = pdn[pdn["status"].eq("Active")].copy()
    norm = {"The Netherlands": "Netherlands", "Czechia": "Czech Republic"}
    for col in ("origins", "destinations"):
        act[col + "_list"] = act[col].fillna("").map(
            lambda s: sorted({norm.get(x.strip(), x.strip()) for x in s.split(",") if x.strip()}))
    dest = act.explode("destinations_list")["destinations_list"].value_counts()
    print(f"Active notices: {len(act)}. Destinations named (a notice can name several):")
    print("  " + ", ".join(f"{k} {v} ({v / len(act):.0%})" for k, v in dest.head(12).items()))
    print(f"  Portugal as a destination: {dest.get('Portugal', 0)} active notices ({dest.get('Portugal', 0) / len(act):.1%})")
    n_orig = act["origins_list"].map(len)
    print(f"Origins named per notice: median {n_orig.median():.0f}; naming 25 or more {(n_orig >= 25).mean():.0%}; "
          f"naming 10 or fewer {(n_orig.between(1, 10)).sum()} notices ({n_orig.between(1, 10).mean():.1%})")
    print(f"  notices naming Portugal among the possible origins: {act['origins_list'].map(lambda xs: 'Portugal' in xs).mean():.1%}")
    broad = act[n_orig >= 15]
    excl = {c: float((~broad["origins_list"].map(lambda xs, c=c: c in xs)).mean()) for c in MS if c in set(dest.index) | {"Portugal"}}
    excl = dict(sorted(excl.items(), key=lambda kv: kv[1]))
    print(f"  among the {len(broad)} notices naming 15 or more origins, share leaving each country out (least first):")
    print("   " + ", ".join(f"{k} {v:.1%}" for k, v in list(excl.items())[:10]) + " ... " +
          ", ".join(f"{k} {v:.1%}" for k, v in list(excl.items())[-6:]))
    sel = act[n_orig.between(1, 10)]
    orig_sel = sel.explode("origins_list")["origins_list"].value_counts()
    print(f"Selective notices (ten origins or fewer): {len(sel)}; how often each country is named:")
    print("  " + ", ".join(f"{k} {v}" for k, v in orig_sel.head(14).items()))
    # Cross-country: notices against the EURIPID price level of medicines.
    pop = pd.read_csv(PART4 / "eurostat_population.csv.gz")
    pop = pop[pop["time"] == 2025].set_index("geo")["value"]
    pli = pd.read_csv(PART4 / "euripid_pli_2017.csv.gz").set_index("geo")["pli_last"]
    tab = pd.DataFrame([{"geo": MS[c], "country": c, "dest": int(dest.get(c, 0)), "orig_sel": int(orig_sel.get(c, 0)),
                         "excl": float((~broad["origins_list"].map(lambda xs, c=c: c in xs)).mean())}
                        for c in sorted(set(MS) - {"Czechia", "United Kingdom", "UK (Northern Ireland)", "Liechtenstein"})])
    tab = tab.drop_duplicates("geo").set_index("geo")
    tab["pop_m"] = pop.reindex(tab.index) / 1e6
    tab["pli"] = pli.reindex(tab.index)
    tab["dest_pm"] = tab["dest"] / tab["pop_m"]
    tab["orig_pm"] = tab["orig_sel"] / tab["pop_m"]
    z = tab.dropna(subset=["pli", "pop_m"])
    for col, lab in (("dest_pm", "destination notices per million people"),
                     ("orig_pm", "mentions as an origin in selective notices, per million people"),
                     ("excl", "share of broad notices leaving the country out as an origin")):
        lo, hi = boot_corr(z["pli"], z[col], rng)
        print(f"  price level against {lab}: Spearman {spearman(z['pli'], z[col]):+.2f} [{lo:+.2f}, {hi:+.2f}] (n={len(z)})")
    print("  countries (price level 2017 | destination notices per million | selective-origin mentions per million | left out):")
    for g, r in tab.sort_values("orig_pm", ascending=False).iterrows():
        pl = "  n/a" if pd.isna(r["pli"]) else f"{r['pli']:5.0f}"
        print(f"    {r['country']:15s} {pl}  {r['dest_pm']:7.1f}  {r['orig_pm']:6.1f}  {r['excl']:5.1%}")
    figure["register"] = [{"geo": g, "country": r["country"], "pli": None if pd.isna(r["pli"]) else float(r["pli"]),
                           "dest_pm": round(float(r["dest_pm"]), 2), "orig_pm": round(float(r["orig_pm"]), 3),
                           "dest_share": round(float(r["dest"]) / len(act), 4), "left_out": round(float(r["excl"]), 4)}
                          for g, r in tab.dropna(subset=["pop_m"]).iterrows()]
    act["key"] = act["product"].map(fold)
    per_product = act.groupby("key").size()
    print(f"  active notices per product: median {per_product.median():.0f}, mean {per_product.mean():.1f}, products {len(per_product)}")

    # ------------------------------------------------------------------------------------ 4
    section("4. What Portugal restricts: prior-notification and suspended-export lists, 2021-2026")
    docs["n_rows"] = docs["n_rows"].astype(int)
    lists = rows[(rows["kind"] == "list") | (rows["type"] == "regulation")].copy()
    reg = lists["type"] == "regulation"
    # Deliberação 391/2021 entered into force on 1 May 2021, with its two lists as annexes.
    lists.loc[reg, "effective"] = "2021-05-01"
    lists.loc[reg & lists["annex"].eq("I"), "type"] = "lnp"
    lists.loc[reg & lists["annex"].eq("II"), "type"] = "suspended"
    lists["effective"] = pd.to_datetime(lists["effective"], errors="coerce")
    lists = lists.dropna(subset=["effective"])
    # One document per list type and effective date.
    keyed = lists.groupby(["type", "effective"])
    sizes = keyed["reg"].nunique().rename("n").reset_index()
    if "sheet" in lists:
        # Since 2026 the suspended list also bans whole classes (quetiapine, estradiol, isosorbide mononitrate),
        # each on its own sheet; n_named counts the presentations named one by one.
        rup = lists[~lists["sheet"].isin(["Quetiapina", "Estradiol", "Mononitrato de isossorbida"])]
        sizes = sizes.merge(rup.groupby(["type", "effective"])["reg"].nunique().rename("n_named").reset_index(),
                            on=["type", "effective"], how="left")
    for typ in ("lnp", "suspended", "prohibited"):
        s = sizes[sizes["type"] == typ].sort_values("effective")
        if len(s):
            print(f"{typ}: {len(s)} lists, {s['effective'].min():%Y-%m} to {s['effective'].max():%Y-%m}; presentations per list "
                  f"median {s['n'].median():.0f}, range {s['n'].min()}-{s['n'].max()}")
            print("   " + ", ".join(f"{d:%Y-%m} {n}" + (f"/{int(m)}" if pd.notna(m) and m != n else "")
                                    for d, n, m in zip(s["effective"], s["n"], s.get("n_named", s["n"]))))
    figure["sizes"] = [{"type": r["type"], "date": f"{r['effective']:%Y-%m-%d}", "n": int(r["n"]),
                        "n_named": int(r["n_named"]) if pd.notna(r.get("n_named")) else int(r["n"])}
                       for _, r in sizes.sort_values(["type", "effective"]).iterrows()]
    ever = {t: set(lists.loc[lists["type"] == t, "reg"]) for t in ("lnp", "suspended", "prohibited")}
    print(f"Distinct presentations ever listed: LNP {len(ever['lnp'])}, suspended {len(ever['suspended'])}, "
          f"prohibited (2026) {len(ever['prohibited'])}")
    both = ever["lnp"] & ever["suspended"]
    print(f"  on both the LNP and a suspended list at some point: {len(both)} "
          f"({len(both) / len(ever['lnp']):.0%} of LNP presentations, {len(both) / len(ever['suspended']):.0%} of suspended)")
    meta = lists.sort_values("effective").groupby("reg").agg(name=("name", "last"), dci=("dci", "last"), mah=("mah", "last"))
    meta["generic"] = [generic_named(n, d) for n, d in zip(meta["name"].fillna(""), meta["dci"].fillna(""))]
    for t in ("lnp", "suspended"):
        m = meta.loc[sorted(ever[t])]
        print(f"  {t}: sold under a substance name (generic-style) {m['generic'].mean():.0%}; most frequent substances: "
              + ", ".join(f"{k} {v}" for k, v in m["dci"].fillna("?").str.split(" \\(|,").str[0].value_counts().head(8).items()))
    # How long presentations stay on the suspended lists, counted in lists.
    sus = lists[lists["type"] == "suspended"]
    k = sus.groupby("reg")["effective"].nunique()
    n_lists = sus["effective"].nunique()
    print(f"  suspended lists observed: {n_lists}; a presentation appears on a median of {k.median():.0f} of them, "
          f"on one only for {(k == 1).mean():.0%}, on five or more for {(k >= 5).mean():.0%}")
    mot = lists[(lists["type"] == "lnp") & lists["motive"].notna()]
    if len(mot):
        print("  LNP reasons by list: " + "; ".join(
            f"{d:%Y-%m}: " + ", ".join(f"{k} {v}" for k, v in g["motive"].value_counts().items())
            + f" (export {g['motive'].str.startswith('Export').mean():.0%})" for d, g in mot.groupby("effective")))
        last_lnp = mot[mot["effective"] == mot["effective"].max()]
        print(f"  latest LNP with reasons ({last_lnp['effective'].max():%Y-%m-%d}): {last_lnp['motive'].value_counts().to_dict()}")
        print("   export-motive presentations: " + ", ".join(sorted(set(last_lnp.loc[last_lnp['motive'].str.startswith('Export'), 'name']))))

    # ------------------------------------------------------------------------------------ 5
    section("5. The monthly prohibitions of 2026 and Infarmed's indicators")
    pro = lists[lists["type"] == "prohibited"].copy()
    for c in ("sat_consumption", "sat_prescription", "n_shortage_reports", "shortage_rate", "intent_rate",
              "export_rate", "n_same_cnpem"):
        pro[c] = pd.to_numeric(pro[c], errors="coerce")
    print(f"Prohibition lists: {pro['effective'].nunique()} ({', '.join(f'{d:%Y-%m-%d}' for d in sorted(pro['effective'].unique()))}); "
          f"rows {len(pro)}, distinct presentations {pro['reg'].nunique()}")
    print(f"  reason: {pro['motive'].value_counts().to_dict()}")
    print("  indicators, as Infarmed names them: export intentions and exports over 'Taxa 0.4' (the quantity declared for the "
          "0.4% sales levy, i.e. packs the company sold); '% satisfação prescrição' and '% satisfação consumos' are ratios that "
          f"exceed 1 in {(pro['sat_prescription'] > 1).sum()} and {(pro['sat_consumption'] > 1).sum()} of {len(pro)} rows, "
          "so they are read as ratios, not shares")
    for mtv, g in pro.groupby("motive"):
        print(f"  {mtv:12s} n={len(g):3d}  median export intentions / quantity sold {g['intent_rate'].median():.2f}, "
              f"exports / quantity sold {g['export_rate'].median():.3f}, prescription-satisfaction ratio {g['sat_prescription'].median():.2f}, "
              f"consumption-satisfaction ratio {g['sat_consumption'].median():.2f}, pharmacy shortage reports {g['n_shortage_reports'].median():.0f}, "
              f"no export intentions {(g['intent_rate'] == 0).mean():.0%}, no exports {(g['export_rate'] == 0).mean():.0%}")
    z = pro.dropna(subset=["intent_rate", "sat_prescription"])
    lo, hi = boot_corr(z["intent_rate"], z["sat_prescription"], rng)
    print(f"  export intentions against the prescription-satisfaction ratio: Spearman {spearman(z['intent_rate'], z['sat_prescription']):+.2f} "
          f"[{lo:+.2f}, {hi:+.2f}] (n={len(z)})")
    z = pro.dropna(subset=["export_rate", "sat_prescription"])
    print(f"  exports carried out against the prescription-satisfaction ratio: Spearman {spearman(z['export_rate'], z['sat_prescription']):+.2f} (n={len(z)})")
    for mtv, g in pro.dropna(subset=["intent_rate", "sat_prescription"]).groupby("motive"):
        if g["intent_rate"].std() > 0:
            print(f"  within reason {mtv}: intentions against the prescription-satisfaction ratio, Spearman "
                  f"{spearman(g['intent_rate'], g['sat_prescription']):+.2f} (n={len(g)})")
    ex = pro[pro["motive"] == "Exportação"]
    print(f"  export-reason rows: exports above 10% of the quantity sold {(ex['export_rate'] > 0.10).mean():.0%}, "
          f"above 25% {(ex['export_rate'] > 0.25).mean():.0%}, above 50% {(ex['export_rate'] > 0.50).mean():.0%}; "
          f"intentions above 100% {(ex['intent_rate'] > 1).mean():.0%}")
    print("  by list: " + "; ".join(f"{d:%Y-%m}: " + ", ".join(f"{k} {v}" for k, v in g['motive'].value_counts().items())
                                     for d, g in pro.groupby("effective")))
    top = (ex.groupby("name").agg(lists=("effective", "nunique"), export=("export_rate", "median"), intent=("intent_rate", "median"))
           .sort_values(["lists", "export"], ascending=False).head(12))
    print("  export-reason presentations most often prohibited (lists, median exports / quantity sold, median intentions / quantity sold):")
    print("   " + "; ".join(f"{n} {int(r['lists'])} {r['export']:.2f} {r['intent']:.2f}" for n, r in top.iterrows()))
    # Are the medicines Portugal lists for export reasons the ones traded in parallel across Europe?
    lnp_m = lists[(lists["type"] == "lnp") & lists["motive"].notna()].copy()
    lnp_m["key"] = lnp_m["name"].map(lambda x: fold(x).split()[0] if fold(x) else "")
    first_word = act["key"].map(lambda x: x.split()[0] if x else "")
    pp = first_word.value_counts()
    for mtv, g in lnp_m.drop_duplicates(["reg", "motive"]).groupby("motive"):
        keys = g["key"].drop_duplicates()
        counts = keys.map(lambda k: int(pp.get(k, 0)))
        print(f"  LNP with reason {mtv}: {len(keys)} distinct names, {(counts > 0).mean():.0%} have active EU parallel "
              f"distribution notices; median notices among those {counts[counts > 0].median() if (counts > 0).any() else 0:.0f}")
    figure["prohibited"] = [{"name": r["name"], "motive": r["motive"], "intent": None if pd.isna(r["intent_rate"]) else float(r["intent_rate"]),
                             "export": None if pd.isna(r["export_rate"]) else float(r["export_rate"]),
                             "sat_presc": None if pd.isna(r["sat_prescription"]) else float(r["sat_prescription"])}
                            for _, r in pro.iterrows()]

    # ------------------------------------------------------------------------------------ 6
    section("6. Why medicines run short: reasons reported by companies")
    seg = r21[r21.find("Motivos das ruturas"):][:700]
    c21 = {k: int(v) for k, v in re.findall(r"(Fabrico da substância ativa|Aumento da procura|Inesperado ou desastres naturais|Regulamentar|Qualidade|Logístico|Comercial|Preço|Fabrico) (\d+)", seg)}
    n21s = sum(c21.values())
    print(f"2021 (shortages notified, counts): {c21}; total {n21s}; manufacturing share "
          f"{(c21.get('Fabrico', 0) + c21.get('Fabrico da substância ativa', 0)) / n21s:.0%}")
    print("  2021 as shares: " + ", ".join(f"{k} {100 * v / n21s:.1f}%" for k, v in c21.items()))
    # 2024 and 2025: pie charts; shares read off the charts, checked against the percentages printed.
    reasons = {2024: (r24, {"Fabrico": 57.44, "Aumento da procura": 21.17, "Logístico": 6.01, "Comercial": 4.94,
                            "Qualidade": 4.15, "Fabrico da substância ativa": 3.07, "Regulamentar": 2.47, "Preço": 0.47,
                            "Inesperado ou desastres naturais": 0.28}),
               2025: (r25, {"Fabrico": 54.28, "Aumento da procura": 21.67, "Logístico": 8.03, "Comercial": 6.34,
                            "Qualidade": 3.89, "Regulamentar": 3.20, "Fabrico da substância ativa": 1.76,
                            "Inesperado ou desastres naturais": 0.47, "Preço": 0.36})}
    for year, (t, shares) in reasons.items():
        i = t.rfind("Motivo das ruturas")
        seg = t[i:t.find("Legenda", i)]
        printed = sorted(float(x.replace(",", ".")) for x in re.findall(r"(\d+,\d+)%", seg))
        if printed != sorted(shares.values()):
            raise SystemExit(f"shortage reasons {year} changed: {printed}")
        manuf = shares["Fabrico"] + shares["Fabrico da substância ativa"]
        print(f"{year} (share of presentations in shortage, Infarmed's chart): {shares}; manufacturing {manuf:.1f}%, "
              f"demand {shares['Aumento da procura']:.1f}%, price {shares['Preço']:.2f}%")
    figure["reasons"] = {"2021": c21, "2024": reasons[2024][1], "2025": reasons[2025][1]}
    n24 = grab(r24, r"72% das ruturas \((\d[\d ]*\d) ruturas notificadas\)")
    print(f"2024: effective shortages notified {n24}")

    # ------------------------------------------------------------------------------------ 7
    section("7. The price of the exported brands in Part 3's rebuilt rule")
    # Part 3's output lists, for the largest protected brands sold in French pharmacies, the ex-factory
    # price in France, Spain, Italy and Belgium and the Portuguese ceiling the rule gives.
    p3 = (PART3 / "output.txt").read_text(encoding="utf-8")
    exported = {fold(n) for n in ever_names(lists, "Export")}
    print("Brands that Infarmed lists for export reasons and that Part 3 priced (ex-factory euros per pack):")
    found = 0
    for line in p3.splitlines():
        m = re.match(r"^\s+([a-z][a-z0-9 ]+?)\s{2,}(.+?)\s+(x\d+)\s+FR\s+([\d.]+)\s+ES\s+([\d.]+|n/a)\s+IT\s+([\d.]+|n/a)"
                     r"\s+BE\s+([\d.]+|n/a)\s+ceiling\s+([\d.]+)", line)
        if m and fold(m.group(1)) in exported:
            found += 1
            print(f"  {m.group(1):12s} {m.group(2)} {m.group(3)}: France {m.group(4)}, Spain {m.group(5)}, Italy {m.group(6)}, "
                  f"Belgium {m.group(7)}; Portuguese ceiling {m.group(8)} ({float(m.group(8)) / float(m.group(4)):.2f} x France)")
    print(f"  brands found: {found}")

    (HERE / "figure_data.json").write_text(json.dumps(figure, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
