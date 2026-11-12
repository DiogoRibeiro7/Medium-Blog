"""One table per country of oral solid presentations with an ex-factory price, in a common layout.

Columns of every table
  country    ES, FR, IT, BE or PT
  brand      first word of the product name, lower case and without accents ("eliquis")
  molecule   erp.stem of the active substances ("apixaban", "sacubitril + valsartan")
  strengths  tuple of strengths in mg, one per active substance, sorted
  units      tablets or capsules in the pack
  pva        ex-factory price of the pack, euros (Portugal's PVA, France's PFHT)
  label      the country's own label, for reading the output

Each builder keeps oral solids (tablets and capsules) only, drops parallel imports where the source
marks them, and keeps the presentation in force (Spain: "ALTA"; Belgium: authorised, with a price).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import drugprices as dp  # noqa: E402
import erp  # noqa: E402

DATA = HERE / "data"
PART2 = HERE.parent / "part2" / "data"
COLS = ["country", "brand", "molecule", "strengths", "units", "pva", "label"]


def brand_of(name: str) -> str:
    s = erp.strip_accents(str(name)).lower()
    m = re.match(r"[a-z][a-z0-9\-]*", s.strip())
    return m.group(0) if m else ""


def _num(s: pd.Series) -> pd.Series:
    return s.map(dp.french_number)


# -------------------------------------------------------------------------------------------- France
def france(data: Path = PART2) -> pd.DataFrame:
    """BDPM presentations sold in community pharmacies in 2024 (Open Medic), with the PFHT backed out
    of the published retail price; the boxes sold in 2024 are kept as weights."""
    fr = dp.fr_tables(data).rows
    fr = fr[fr["oral_solid"] & fr["prix_1"].notna() & (fr["prix_1"] > 0) & fr["units"].notna()].copy()
    comp = pd.read_csv(data / "bdpm_cis_compo.csv.gz", dtype=str)
    comp["ft"] = (comp["nature_composant"] == "FT").astype(int)
    comp["has_ft"] = comp.groupby(["cis", "numero_liaison"])["ft"].transform("max")
    comp = comp[comp["ft"] == comp["has_ft"]]
    dose = comp.groupby("cis")["dosage"].apply(lambda d: tuple(sorted(x for v in d for x in erp.strengths(v))))
    fr["strengths"] = fr["denomination"].map(erp.strengths)
    from_compo = fr["cis"].map(dose)
    n_sub = fr["molecule"].fillna("").str.count(r"\+") + 1
    bad = fr["strengths"].map(len) != n_sub
    fr.loc[bad, "strengths"] = from_compo[bad]
    fr["brand"] = fr["denomination"].map(brand_of)
    fr["molecule_fr"] = fr["molecule"]
    fr["molecule"] = fr["molecule"].fillna("").map(erp.stem)
    fr["pva"] = erp.fr_pfht(fr["prix_1"].to_numpy())
    fr["country"] = "FR"
    fr["label"] = fr["denomination"] + " / " + fr["libelle_presentation"].str.slice(0, 60)
    fr["units"] = fr["units"].astype(int)
    keep = COLS + ["cip13", "segment", "boites", "prix_1", "molecule_fr"]
    return fr[keep].reset_index(drop=True)


# --------------------------------------------------------------------------------------------- Spain
ES_UNITS = re.compile(r"(\d+)\s*(?:comprimidos|comprimido|comp\b|capsulas|capsula|caps\b|grageas)", re.I)
ES_FORM = re.compile(r"comprimid|\bcomp\b|capsul|\bcaps\b|gragea", re.I)


def spain(data: Path = DATA) -> pd.DataFrame:
    es = pd.read_csv(data / "es_nomenclator.csv.gz", dtype=str)
    es.columns = [erp.strip_accents(c) for c in es.columns]
    name = es["Nombre del producto farmaceutico"].fillna("")
    plain = name.map(erp.strip_accents)
    es = es[(es["Estado"] == "ALTA") & plain.str.contains(ES_FORM) & ~plain.str.contains(r"envase clinico", case=False)]
    es = es.assign(pvp=pd.to_numeric(es["Precio venta al publico con IVA"], errors="coerce"))
    es = es[es["pvp"] > 0].copy()
    plain = es["Nombre del producto farmaceutico"].map(erp.strip_accents)
    # The pack count is the last "<n> comprimidos" in the name ("60 comprimidos, 10 comprimidos x 6
    # blisters" is a pack of 60); names that state no count are dropped.
    es["units"] = plain.map(lambda s: (lambda m: float(m[0]) if m else np.nan)(ES_UNITS.findall(s)))
    es = es[es["units"] > 0]
    es["alta"] = pd.to_datetime(es["Fecha de alta en el nomenclator"], dayfirst=True, errors="coerce")
    out = pd.DataFrame({
        "country": "ES",
        "brand": es["Nombre del producto farmaceutico"].map(brand_of),
        "molecule": es["Principio activo o asociacion de principios activos"].map(erp.stem),
        "strengths": es["Nombre del producto farmaceutico"].map(erp.strengths),
        "units": es["units"].astype(int),
        "pva": es["pvp"].map(erp.es_pva),
        "label": es["Nombre del producto farmaceutico"],
        "generic": es["Tipo de farmaco"].fillna("").str.contains("Generic"),
        "lab": es["Nombre del laboratorio ofertante"],
        "alta": es["alta"],
        "pvp": es["pvp"],
    })
    # Parallel distributors list the same pack under their own code; keep the oldest listing of each
    # brand, strength and pack, which is the marketing authorisation holder's.
    out = out.sort_values(["alta", "label"]).drop_duplicates(["brand", "molecule", "strengths", "units"])
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------------- Italy
IT_FORM = re.compile(r"\b(?:cpr|cps|compresse|capsule|cp|confetti)\b", re.I)


def italy(data: Path = DATA) -> pd.DataFrame:
    it = pd.read_csv(data / "it_classe_a.csv.gz", sep=";", dtype=str, encoding="latin-1")
    it.columns = [c.strip() for c in it.columns]
    price_col = next(c for c in it.columns if c.startswith("Prezzo al pubblico"))
    den = it["Denominazione e Confezione"].fillna("")
    # Titration packs hold tablets of two strengths ("11 cpr 0,5 mg + 14 cpr 1 mg"); the group label
    # then shows only the first strength, so they are dropped.
    titration = den.str.contains(r"\+\s*\d+\s*(?:x\s*\d+\s*)?(?:cpr|cps|compresse|capsule)", case=False)
    it = it[it["Descrizione gruppo"].str.contains("USO ORALE", na=False)
            & den.str.contains(IT_FORM) & ~titration].copy()
    units = it["Descrizione gruppo"].str.extract(r"(\d+)\s+UNITA'")[0].astype(float)
    desc = it["Descrizione gruppo"].str.replace(r"\s+\d+\s+UNITA'.*$", "", regex=True)
    out = pd.DataFrame({
        "country": "IT",
        "brand": it["Denominazione e Confezione"].str.split("*").str[0].map(brand_of),
        "molecule": it["Principio attivo"].map(erp.stem),
        "strengths": desc.map(erp.strengths),
        "units": units,
        "pva": _num(it[price_col]).map(erp.it_pva),
        "label": it["Denominazione e Confezione"],
        "holder": it["Titolare AIC"],
        "pvp": _num(it[price_col]),
    })
    out = out[(out["units"] > 0) & (out["pva"] > 0)]
    out["units"] = out["units"].astype(int)
    return out.reset_index(drop=True)


# ------------------------------------------------------------------------------------------- Belgium
BE_FORM = re.compile(r"comprim|g[ée]lule|capsule", re.I)


def belgium(data: Path = DATA) -> pd.DataFrame:
    a = pd.read_csv(data / "be_sam_amp.csv.gz", dtype=str, low_memory=False)
    vmp = pd.read_csv(data / "be_sam_vmp.csv.gz", dtype=str, usecols=["@code", "Vtm.Name.Fr"])
    a = a[(a["Status"] == "AUTHORIZED") & (a["ampp.Status"] == "AUTHORIZED")
          & (a["ampp.ParallelCircuit"].fillna("0") == "0")
          & a["AmpComponent.PharmaceuticalForm.Name.Fr"].fillna("").str.contains(BE_FORM)
          & a["AmpComponent.RouteOfAdministration.Name.Fr"].fillna("").str.contains("orale")].copy()
    # ExFactoryPrice is the price the reimbursement rules use; the price notified to the Economy
    # ministry (OfficialExFactoryPrice) fills the gaps for packs that are not reimbursed.
    price = pd.to_numeric(a["ampp.ExFactoryPrice"], errors="coerce")
    price = price.fillna(pd.to_numeric(a["ampp.OfficialExFactoryPrice"], errors="coerce"))
    a["pva"] = price
    a = a[a["pva"] > 0]
    a["units"] = pd.to_numeric(a["ampp.PackDisplayValue.Fr"].str.extract(r"^\s*(\d+)")[0], errors="coerce")
    a = a[a["units"] > 0]
    a = a.merge(vmp.rename(columns={"@code": "@vmpCode"}), on="@vmpCode", how="left")
    mol = a["Vtm.Name.Fr"].fillna(a["AmpComponent.RealActualIngredient.Substance.Name.Fr"].fillna(""))
    out = pd.DataFrame({
        "country": "BE",
        "brand": a["Name.Fr"].map(brand_of),
        "molecule": mol.map(erp.stem),
        "strengths": a["Name.Fr"].map(erp.strengths),
        "units": a["units"].astype(int),
        "pva": a["pva"],
        "label": a["ampp.PrescriptionNameFamhp.Fr"],
        "reimbursed": a["ampp.ExFactoryPrice"].notna(),
    })
    out = out.sort_values(["reimbursed", "label"], ascending=[False, True])
    out = out.drop_duplicates(["brand", "molecule", "strengths", "units"])
    return out.reset_index(drop=True)


# ------------------------------------------------------------------------------------------ Portugal
PT_ROW = re.compile(r"^\s*(?:GH\d{4}\s+)?(\d{7})\s+(.*?)\s+(\d+)\s+unidade\(s\)\s+(.*)$")


def portugal_gh(data: Path = DATA) -> pd.DataFrame:
    """Presentations in Infarmed's homogeneous-group list (third quarter of 2026): every medicine
    that has generics in Portugal, the originals included, with its maximum and current retail price."""
    import gzip
    text = gzip.decompress((data / "pt_gh_2026q3.txt.gz").read_bytes()).decode("utf-8")
    rows = []
    for line in text.splitlines():
        m = PT_ROW.match(line)
        if not m:
            continue
        reg, mid, units, tail = m.groups()
        parts = re.split(r"\s{2,}", mid.strip())
        nums = re.findall(r"\d+(?:[.,]\d+)?", tail)
        if len(nums) < 2:
            continue
        name = parts[0]
        dose = parts[-1]
        form = next((p for p in parts[1:] if re.search(r"Comprimido|Cápsula|Capsula|Pastilha|Drageia", p)), "")
        rows.append({"registo": reg, "name": name, "form": form, "dose": dose, "units": int(units),
                     "pvp_max": dp.french_number(nums[0]), "pvp": dp.french_number(nums[1])})
    pt = pd.DataFrame(rows)
    pt["oral_solid"] = pt["form"].ne("")
    pt["parallel"] = pt["name"].str.contains(r"\bIP\b")
    pt["brand"] = pt["name"].map(brand_of)
    pt["strengths"] = pt["dose"].map(erp.strengths)
    return pt


def all_reference() -> pd.DataFrame:
    fr = france()
    tabs = [fr[COLS], spain()[COLS], italy()[COLS], belgium()[COLS]]
    return pd.concat(tabs, ignore_index=True)
