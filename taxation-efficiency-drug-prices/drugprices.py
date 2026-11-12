"""Shared tools for the drug-price parts of the taxation-efficiency-drug-prices series.

The functions here turn the public US and French files fetched by part2/fetch.py into one table per
country at the level of molecule x market segment, add the segments up to the molecule, sort the
molecules into baskets by how much generic competition they face in each country, and compute
bilateral price indices on the molecules both countries share.

Molecules. Names are matched across languages by normalising the international non-proprietary
names: accents and salt or hydrate words are removed and a final "e" is dropped, so that the French
"dapagliflozine propanediol monohydratee" and the US "Dapagliflozin Propanediol" both become
"dapagliflozin". Combination products become sorted lists of their components
("sacubitril + valsartan"). A few US adopted names are mapped to the international ones
(acetaminophen to paracetamol), and US names that Part D cuts short are completed from the French
list when only one French molecule fits (harmonise).

Segments. "originator" is the brand-name product of the company that holds the original approval;
"generic" is everything sold under the molecule's own name or classified as a generic (and, in the
US, products approved through an abbreviated application).

Baskets. single-source molecules have no generic competition in either country, multi-source
molecules have it in both, and mixed molecules in one country only (molecule_baskets).

Units. Prices are per dosage unit, which is only comparable for oral solid forms (tablets and
capsules), so the basket is restricted to them.

Negotiated prices. mfp_table reads the Medicare negotiated prices (maximum fair prices) per unit
and sets them against what pharmacies paid for the same package codes in 2024 (NADAC).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

US_SALTS = set("""
hcl hydrochloride dihydrochloride sodium potassium calcium magnesium besylate maleate mesylate tartrate succinate
fumarate citrate phosphate sulfate acetate bromide hbr hydrobromide propanediol monohydrate dihydrate er xl sr dr cd
la odt oral phos esylate hyclate anhydrous dipropionate propionate furoate disodium hemifumarate medoxomil axetil pivoxil trihydrate
bitartrate dimesylate tosylate hemihydrate lysine erbumine mofetil camsylate oxalate nitrate lactate gluconate
malate meglumine bisulfate
""".split())

FR_SALTS = set("""
chlorhydrate dichlorhydrate de d du sodique disodique potassique calcique magnesien magnesienne besilate besylate
maleate mesilate dimesilate tartrate bitartrate succinate fumarate hemifumarate citrate phosphate sulfate acetate
bromhydrate propanediol monohydrate monohydratee dihydrate dihydratee trihydrate trihydratee hemihydrate hemihydratee
anhydre base embonate malate hemipentahydrate complexe sous forme lysine tosilate hyclate oxalate nitrate lactate
gluconate erbumine mofetil medoxomil axetil pivoxil dipropionate propionate furoate camsilate sel l arginine tert
butylamine sesquihydrate pentahydrate meglumine
""".split())


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def inn_key(word: str) -> str:
    word = word.strip()
    return word[:-1] if word.endswith("e") else word


# US adopted names that differ from the international non-proprietary names used in France, written
# after inn_key (final "e" dropped).
US_TO_INN = {"acetaminophen": "paracetamol", "glyburid": "glibenclamid", "albuterol": "salbutamol",
             "cyclosporin": "ciclosporin", "rifampin": "rifampicin", "aspirin": "acid acetylsalicyliqu",
             "valacyclovir": "valaciclovir", "acyclovir": "aciclovir"}


def molecule_us(generic_name: str) -> str:
    s = strip_accents(str(generic_name)).lower().replace("*", "")
    parts = re.split(r"\s*/\s*|\s*-\s*|\s+with\s+|\s+and\s+", s)
    out = []
    for p in parts:
        words = [US_TO_INN.get(inn_key(t), inn_key(t)) for t in re.findall(r"[a-z]+", p)
                 if t not in US_SALTS and len(t) > 1]
        if words:
            out.append(" ".join(words))
    return " + ".join(sorted(set(out)))


def harmonise(us_cells: pd.DataFrame, fr_names) -> pd.DataFrame:
    """Rename US molecules whose Part D names are cut short to the French name they abbreviate.

    Part D truncates long generic names ("Bictegrav/Emtricit/Tenofov Ala"). A US molecule that has no
    exact French match is renamed when exactly one French molecule has the same number of components
    and every word of every US component (three letters or more) begins the matching French word, as
    "tenofov ala" begins "tenofovir alafenamid". Cells that end up with the same name are added up.
    """
    fr_set = set(fr_names)
    fr_split = {f: [c.split(" ") for c in f.split(" + ")] for f in fr_set}

    def fits(us_comp, fr_comp):
        return len(us_comp) == len(fr_comp) and all(len(a) >= 3 and b.startswith(a) for a, b in zip(us_comp, fr_comp))

    def find(name):
        if name in fr_set:
            return name
        comps = [c.split(" ") for c in name.split(" + ")]
        hits = [f for f, fc in fr_split.items() if len(fc) == len(comps) and all(fits(a, b) for a, b in zip(comps, fc))]
        return hits[0] if len(hits) == 1 else name

    out = us_cells.copy()
    out["molecule"] = out["molecule"].map({n: find(n) for n in out["molecule"].unique()})
    keys = ["molecule", "segment"]
    agg = {"q_us": "sum", "v_us": "sum"}
    if "mol_oral_share" in out.columns:
        agg["mol_oral_share"] = "min"
    out = out.groupby(keys, as_index=False).agg(agg)
    out["p_us"] = out["v_us"] / out["q_us"]
    return out


def substance_fr(name: str) -> str:
    s = strip_accents(str(name)).lower()
    s = re.sub(r"[()']", " ", s)
    words = [inn_key(t) for t in re.findall(r"[a-z]+", s) if t not in FR_SALTS]
    return " ".join(words)


def french_number(x) -> float:
    """'1.466,29', '24,34', '1466.29' -> float."""
    x = str(x).strip().replace(" ", "").replace(" ", "")
    if x in ("", "nan", "None"):
        return np.nan
    if "," in x:
        x = x.replace(".", "").replace(",", ".")
    elif x.count(".") > 1:
        x = x.replace(".", "", x.count(".") - 1)
    try:
        return float(x)
    except ValueError:
        return np.nan


def units_per_box(label: str) -> float:
    """Number of tablets or capsules in a French presentation, from its BDPM label."""
    s = strip_accents(str(label)).lower()
    m = re.search(r"(\d+)\s*x\s*(\d+)\s*(comprim|gelule|capsule)", s)
    if m:
        return float(int(m.group(1)) * int(m.group(2)))
    inner = re.search(r"de\s+(\d+)\s*(comprim|gelule|capsule|lyophilisat)", s)
    if not inner:
        lead = re.search(r"^\s*(\d+)\s*(comprim|gelule|capsule)", s)
        return float(lead.group(1)) if lead else np.nan
    outer = re.match(r"^\s*(\d+)\s+(plaquette|flacon|tube|pilulier|sachet|boite|etui|recipient|pot)", s)
    return float(int(inner.group(1)) * (int(outer.group(1)) if outer else 1))


# ------------------------------------------------------------------------------------------- tables
@dataclass
class UsTables:
    rows: pd.DataFrame          # Part D "Overall" rows with segment, oral-solid share and molecule
    totals: dict


def us_tables(data: Path, oral_threshold: float = 0.75) -> UsTables:
    """Part D "Overall" rows with a market segment, an oral-solid flag and a molecule.

    Part D reports each brand name across all its forms and strengths, so the form has to come from
    elsewhere. Each row is looked up in the FDA NDC directory by its brand name, dropping words from
    the end of the name until it matches ("Janumet XR" -> "Janumet"); a row sold under the molecule's
    own name that matches nothing takes the figures of its molecule. oral_share is the share of the matched FDA package listings
    that are tablets or capsules taken by mouth; mol_oral_share is the same share for every listing
    of the molecule, so a molecule with mol_oral_share of 1 has no other form that could be mixed
    into its Part D units.
    """
    pdd = pd.read_csv(data / "cms_partd_2024.csv.gz", dtype=str)
    ov = pdd[pdd["Mftr_Name"].str.strip() == "Overall"].copy()
    for c in ("Tot_Spndng_2024", "Tot_Dsg_Unts_2024", "Tot_Clms_2024"):
        ov[c] = pd.to_numeric(ov[c], errors="coerce")

    fda = pd.read_csv(data / "fda_ndc.csv.gz", dtype=str)
    fda["pkey"] = fda["PROPRIETARYNAME"].str.strip().str.upper()
    form = fda["DOSAGEFORMNAME"].str.upper().fillna("")
    route = fda["ROUTENAME"].str.upper().fillna("")
    # Kits (starter packs, often of tablets, listed without a route) and package rows without product
    # details are left out of the shares.
    oral = (form.str.contains("TABLET|CAPSULE") & route.str.contains("ORAL")).astype(float)
    fda["oral_solid"] = oral.where(fda["DOSAGEFORMNAME"].notna() & (form != "KIT"))
    cat = fda["MARKETINGCATEGORYNAME"].fillna("")
    fda["is_nda"] = cat.isin(["NDA", "BLA"])
    fda["is_anda"] = cat.isin(["ANDA", "NDA AUTHORIZED GENERIC"])
    fda["molecule"] = (fda["NONPROPRIETARYNAME"].fillna("")
                       .str.replace(r"\s+and\s+|,\s*", "/", regex=True).map(molecule_us))
    agg = dict(oral_share=("oral_solid", "mean"), nda_share=("is_nda", "mean"), anda_share=("is_anda", "mean"))
    by_key = fda.groupby("pkey").agg(**agg)
    by_mol = fda[fda["molecule"] != ""].groupby("molecule").agg(**agg)
    keys = set(by_key.index)

    def lookup(name: str):
        words = re.sub(r"\(.*?\)", " ", str(name)).replace("*", "").upper().split()
        for k in range(len(words), 0, -1):
            key = " ".join(words[:k])
            if key in keys:
                return key
        return None

    ov["molecule"] = ov["Gnrc_Name"].map(molecule_us)
    ov["fda_key"] = ov["Brnd_Name"].map(lookup)
    ov = ov.merge(by_key, left_on="fda_key", right_index=True, how="left")
    same_name = ov["Brnd_Name"].map(molecule_us) == ov["molecule"]
    # Only rows sold under the molecule's own name fall back on the molecule: a brand that is not in
    # the directory, or is listed only as kits, could be an injectable of an oral molecule.
    use_mol = ov["fda_key"].isna() & same_name
    fill = by_mol.reindex(ov["molecule"]).set_axis(ov.index)
    for c in ("oral_share", "nda_share", "anda_share"):
        ov[c] = ov[c].where(~use_mol, fill[c])
    ov["fda_match"] = np.where(ov["fda_key"].notna(), "brand name",
                               np.where(use_mol & fill["oral_share"].notna(), "molecule", "none"))
    ov["mol_oral_share"] = by_mol["oral_share"].reindex(ov["molecule"]).to_numpy()

    seg = np.full(len(ov), "other", dtype=object)
    seg[same_name.to_numpy()] = "generic"
    named = ~same_name
    seg[(named & (ov["nda_share"] >= 0.5)).to_numpy()] = "originator"
    seg[(named & (ov["anda_share"] > 0.5)).to_numpy()] = "generic"
    ov["segment"] = seg
    ov["oral_solid"] = ov["oral_share"].fillna(0) >= oral_threshold

    tot = {
        "spend": float(ov["Tot_Spndng_2024"].sum()),
        "units": float(ov["Tot_Dsg_Unts_2024"].sum()),
        "claims": float(ov["Tot_Clms_2024"].sum()),
        "rows": int(len(ov)),
    }
    return UsTables(ov, tot)


@dataclass
class FrTables:
    rows: pd.DataFrame          # one row per CIP13 presentation with 2024 volume
    totals: dict


def fr_tables(data: Path) -> FrTables:
    comp = pd.read_csv(data / "bdpm_cis_compo.csv.gz", dtype=str)
    comp["sub"] = comp["substance"].map(substance_fr)
    # A substance can be listed as the salt in the product (SA) and as the active moiety (FT) under the
    # same link number; keep the moieties where they are given, and every one of them (Entresto lists
    # sacubitril and valsartan as two FT entries under one link number).
    comp["ft"] = (comp["nature_composant"] == "FT").astype(int)
    comp["has_ft"] = comp.groupby(["cis", "numero_liaison"])["ft"].transform("max")
    comp = comp[comp["ft"] == comp["has_ft"]]
    mol = comp.groupby("cis")["sub"].apply(lambda x: " + ".join(sorted({v for v in x if v}))).rename("molecule")

    cis = pd.read_csv(data / "bdpm_cis.csv.gz", dtype=str)
    cis["oral_solid"] = (cis["forme"].str.contains("comprim|gélule|capsule", na=False)
                         & cis["voies"].str.contains("orale", na=False))
    gen = pd.read_csv(data / "bdpm_cis_gener.csv.gz", dtype=str)
    gtype = gen.groupby("cis")["type_generique"].min()

    cip = pd.read_csv(data / "bdpm_cis_cip.csv.gz", dtype=str)
    cip["units_label"] = cip["libelle_presentation"].map(units_per_box)
    for k in ("prix_1", "prix_2", "prix_3"):
        cip[k] = cip[k].map(french_number)

    fr = (cip.merge(cis[["cis", "denomination", "forme", "voies", "oral_solid"]], on="cis", how="left")
             .merge(mol, left_on="cis", right_index=True, how="left"))
    fr["oral_solid"] = fr["oral_solid"].fillna(False).astype(bool)
    fr["gtype"] = fr["cis"].map(gtype)
    first_word = fr["denomination"].fillna("").map(lambda d: inn_key(strip_accents(d.split(" ")[0]).lower()))
    first_mol = fr["molecule"].fillna("").map(lambda m: m.split(" + ")[0].split(" ")[0])
    is_gen = fr["gtype"].isin(["1", "2", "4"]) | ((first_word == first_mol) & (first_mol != ""))
    fr["segment"] = np.where(is_gen, "generic", "originator")

    om = pd.read_csv(data / "open_medic_2024.csv.gz", dtype={"cip13": str})
    totals = {"boxes": float(om["boites"].sum()), "bse": float(om["bse"].sum()),
              "rem": float(om["rem"].sum()), "presentations": int(len(om))}
    fr = fr.merge(om, on="cip13", how="inner")
    # Open Medic's own label usually ends with the unit count ("OTEZLA 30 MG CPR 56"). The two counts
    # agree for 99.6 percent of oral solid boxes; where they differ the larger is taken, because the
    # BDPM parser misses parts of multi-blister packs ("4 tablets of 10 mg - 4 of 20 mg - 19 of 30 mg")
    # more often than it double counts.
    label_units = fr["l_cip13"].str.extract(r"(\d+)\s*$")[0].astype(float)
    fr["units"] = np.fmax(fr["units_label"], label_units)
    # Dispensing fee per box: BDPM gives it as the third price field; 1.02 euro is the standard fee.
    fr["fee_box"] = fr["prix_3"].fillna(1.02)
    return FrTables(fr, totals)


def cells_us(us: pd.DataFrame, oral_only: bool = True) -> pd.DataFrame:
    r = us[us["segment"].isin(["originator", "generic"]) & (us["Tot_Dsg_Unts_2024"] > 0)]
    if oral_only:
        r = r[r["oral_solid"]]
    c = r.groupby(["molecule", "segment"]).agg(q_us=("Tot_Dsg_Unts_2024", "sum"),
                                               v_us=("Tot_Spndng_2024", "sum"),
                                               mol_oral_share=("mol_oral_share", "min")).reset_index()
    c["p_us"] = c["v_us"] / c["q_us"]
    return c


def cells_fr(fr: pd.DataFrame, include_fee: bool = True) -> pd.DataFrame:
    r = fr[fr["oral_solid"] & (fr["units"] > 0) & (fr["boites"] > 0) & (fr["bse"] > 0) & fr["molecule"].notna()].copy()
    r["q"] = r["boites"] * r["units"]
    r["v"] = r["bse"] + (r["fee_box"] * r["boites"] if include_fee else 0.0)
    c = r.groupby(["molecule", "segment"]).agg(q_fr=("q", "sum"), v_fr=("v", "sum"), v_fr_bse=("bse", "sum")).reset_index()
    c["p_fr"] = c["v_fr"] / c["q_fr"]
    return c


def match(us_cells: pd.DataFrame, fr_cells: pd.DataFrame, usd_per_eur: float) -> pd.DataFrame:
    us_cells = harmonise(us_cells, fr_cells["molecule"].unique())
    m = us_cells.merge(fr_cells, on=["molecule", "segment"], how="inner")
    m["p_fr_usd"] = m["p_fr"] * usd_per_eur
    m["ratio"] = m["p_us"] / m["p_fr_usd"]
    return m


# ------------------------------------------------------------------------------------------ indices
def laspeyres(m: pd.DataFrame, p_us="p_us", p_fr="p_fr_usd") -> float:
    """US-basket index: what the US basket costs at US prices over what it would cost at French prices."""
    return float((m["q_us"] * m[p_us]).sum() / (m["q_us"] * m[p_fr]).sum())


def paasche(m: pd.DataFrame, p_us="p_us", p_fr="p_fr_usd") -> float:
    """French-basket index: what the French basket would cost at US prices over its cost at French prices."""
    return float((m["q_fr"] * m[p_us]).sum() / (m["q_fr"] * m[p_fr]).sum())


def fisher(m: pd.DataFrame, **kw) -> float:
    return float(np.sqrt(laspeyres(m, **kw) * paasche(m, **kw)))


def jevons(m: pd.DataFrame, p_us="p_us", p_fr="p_fr_usd") -> float:
    """Unweighted geometric mean of the price ratios."""
    return float(np.exp(np.log(m[p_us] / m[p_fr]).mean()))


def mixed_index(m: pd.DataFrame, share_a: float, a: str, b: str, col: str = "segment",
                weights: str = "us") -> float:
    """All-drug index when group `a` makes up `share_a` of the units and group `b` the rest.

    Within each group the molecules keep the weights of the chosen country; only the split of units
    between the two groups is set from outside. The result moves from the index of group `b` (at 0)
    to the index of group `a` (at 1), but not along a straight line: it is a ratio of two averages.
    """
    q = "q_us" if weights == "us" else "q_fr"
    out_us = out_fr = 0.0
    for grp, share in ((a, share_a), (b, 1.0 - share_a)):
        s = m[m[col] == grp]
        w = s[q] / s[q].sum()
        out_us += share * float((w * s["p_us"]).sum())
        out_fr += share * float((w * s["p_fr_usd"]).sum())
    return out_us / out_fr


# ----------------------------------------------------------------------------- molecule-level baskets
BASKETS = ("single-source", "multi-source", "mixed")


def molecule_baskets(us_cells: pd.DataFrame, fr_cells: pd.DataFrame, usd_per_eur: float,
                     threshold: float = 0.05) -> pd.DataFrame:
    """One row per molecule sold in both countries, with its basket.

    Each country's brand and generic cells are added up to the molecule, so that a molecule's price
    is what an average unit of it costs in that country, whatever mix of brand and generic is bought.
    The basket is set by the generic share of units in each country:
      single-source  generics below `threshold` in both countries (the molecule is still protected);
      multi-source   generics at or above `threshold` in both;
      mixed          generics in one country only (gen_us and gen_fr say which).
    """
    def level(c, q, v, side, extra=()):
        out = c.groupby("molecule")[[q, v, *extra]].sum()
        gen = c[c["segment"] == "generic"].groupby("molecule")[q].sum()
        out[f"gen_{side}"] = gen.reindex(out.index).fillna(0.0) / out[q]
        return out

    us_cells = harmonise(us_cells, fr_cells["molecule"].unique())
    extra = ("v_fr_bse",) if "v_fr_bse" in fr_cells.columns else ()
    m = level(us_cells, "q_us", "v_us", "us").join(level(fr_cells, "q_fr", "v_fr", "fr", extra), how="inner")
    if "mol_oral_share" in us_cells.columns:
        m = m.join(us_cells.groupby("molecule")["mol_oral_share"].min())
    m = m.reset_index()
    m["p_us"] = m["v_us"] / m["q_us"]
    m["p_fr"] = m["v_fr"] / m["q_fr"]
    m["p_fr_usd"] = m["p_fr"] * usd_per_eur
    m["ratio"] = m["p_us"] / m["p_fr_usd"]
    gu, gf = m["gen_us"] >= threshold, m["gen_fr"] >= threshold
    m["basket"] = np.select([~gu & ~gf, gu & gf], ["single-source", "multi-source"], "mixed")
    m["mixed_side"] = np.select([gu & ~gf, ~gu & gf], ["generic in US only", "generic in France only"], "")
    return m


def indices(m: pd.DataFrame, **kw) -> dict:
    """Laspeyres, Paasche, Fisher, Jevons and the median ratio of a basket."""
    p_us, p_fr = kw.get("p_us", "p_us"), kw.get("p_fr", "p_fr_usd")
    return {"n": int(len(m)), "L": laspeyres(m, **kw), "P": paasche(m, **kw), "F": fisher(m, **kw),
            "J": jevons(m, **kw), "median": float((m[p_us] / m[p_fr]).median())}


def bootstrap(m: pd.DataFrame, rng: np.random.Generator, reps: int = 2000) -> dict:
    """Percentile intervals for L, P and F when the molecules are resampled with replacement.

    This measures how much the index depends on which molecules happen to be in the basket, not
    sampling error in the prices, which are totals over the whole programme.
    """
    idx = rng.integers(0, len(m), size=(reps, len(m)))
    qu, qf = m["q_us"].to_numpy(), m["q_fr"].to_numpy()
    pu, pf = m["p_us"].to_numpy(), m["p_fr_usd"].to_numpy()
    L = (qu[idx] * pu[idx]).sum(1) / (qu[idx] * pf[idx]).sum(1)
    P = (qf[idx] * pu[idx]).sum(1) / (qf[idx] * pf[idx]).sum(1)
    F = np.sqrt(L * P)
    q = lambda x: (float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)))  # noqa: E731
    return {"L": q(L), "P": q(P), "F": q(F)}


# ------------------------------------------------------------------------- NADAC and negotiated prices
def fda_names(data: Path) -> pd.DataFrame:
    """FDA NDC directory reduced to one row per 11-digit package code."""
    fda = pd.read_csv(data / "fda_ndc.csv.gz", dtype=str)
    form = fda["DOSAGEFORMNAME"].str.upper().fillna("")
    route = fda["ROUTENAME"].str.upper().fillna("")
    fda["oral_solid"] = form.str.contains("TABLET|CAPSULE") & route.str.contains("ORAL")
    name = fda["NONPROPRIETARYNAME"].fillna("").str.replace(r"\s+and\s+|,\s*", "/", regex=True)
    fda["molecule"] = name.map(molecule_us)
    return fda.drop_duplicates("ndc11")[["ndc11", "PROPRIETARYNAME", "molecule", "oral_solid"]]


def nadac_molecules(data: Path) -> pd.DataFrame:
    """Median 2024 NADAC per unit, by molecule and class (B brand, G generic), oral solids only.

    NADAC is the average price retail pharmacies pay to acquire a drug, before any dispensing fee.
    The median is taken across package codes, so it does not reflect the mix of strengths bought.
    """
    nd = pd.read_csv(data / "nadac_2024.csv.gz", dtype={"ndc": str})
    nd = nd[(nd["pricing_unit"] == "EA") & (nd["otc"] == "N")]
    nd = nd.merge(fda_names(data), left_on="ndc", right_on="ndc11", how="inner")
    nd = nd[nd["oral_solid"] & (nd["molecule"] != "")]
    nd["cls"] = np.where(nd["classification"].str.startswith("B"), "B", "G")
    out = nd.groupby(["molecule", "cls"])["nadac_mean_2024"].median().unstack()
    return out.rename(columns={"B": "nadac_b", "G": "nadac_g"})


def mfp_table(data: Path) -> pd.DataFrame:
    """Negotiated Medicare prices per unit for the oral solid selected drugs, matched to NADAC 2024.

    For each drug: the first negotiated price (effective 1 January of its price applicability year),
    as the median across its package codes of the price per unit; the 2024 NADAC per unit of the same
    package codes; and the median across package codes of the ratio between the two, which is the
    discount of the negotiated price from what pharmacies paid in 2024.
    """
    mf = pd.read_csv(data / "cms_mfp_cms_negotiation_program_selected_drug_list_and_maximum_fair_.csv.gz",
                     dtype=str)
    mf["ndc11"] = mf["NDC-11"].str.replace("-", "", regex=False)
    mf["eff"] = mf["MFP Effective Date"].str[:10]
    mf["unit_price"] = pd.to_numeric(mf["NDC-9 MFP per Unit Price"], errors="coerce")
    mf["mfp_30"] = pd.to_numeric(mf["Single MFP per 30 DES"], errors="coerce")
    mf = mf[mf["unit_price"].notna() & (mf["eff"] == mf["IPAY"] + "-01-01")]
    mf = mf.merge(fda_names(data), on="ndc11", how="left")
    mf = mf[mf["oral_solid"].fillna(False).astype(bool)]
    nd = pd.read_csv(data / "nadac_2024.csv.gz", dtype={"ndc": str})[["ndc", "nadac_mean_2024", "pricing_unit"]]
    mf = mf.merge(nd, left_on="ndc11", right_on="ndc", how="left")
    mf["r_nadac"] = mf["unit_price"] / mf["nadac_mean_2024"]
    rows = []
    for (ipay, drug), g in mf.groupby(["IPAY", "Selected Drug Name"]):
        mol = g["molecule"].dropna()
        brand = g["PROPRIETARYNAME"].dropna()
        rows.append({
            "ipay": int(ipay), "drug": brand.mode().iloc[0].title() if len(brand) else drug.title(),
            "molecule": mol.mode().iloc[0] if len(mol) else "",
            "ndcs": int(g["ndc11"].nunique()), "mfp_unit": float(g["unit_price"].median()),
            "mfp_30": float(g["mfp_30"].median()),
            "nadac_unit": float(g["nadac_mean_2024"].median()) if g["nadac_mean_2024"].notna().any() else np.nan,
            "mfp_over_nadac": float(g["r_nadac"].median()) if g["r_nadac"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["ipay", "drug"]).reset_index(drop=True)


def fr_current_unit_price(fr: pd.DataFrame, segment: str | None = None) -> pd.Series:
    """Public price per unit before the dispensing fee in the current BDPM file, by molecule.

    Each presentation is weighted by the boxes it sold in 2024. The current file is the price list in
    force when the data were fetched, so this is the French counterpart of a 2026 US price.
    """
    r = fr[fr["oral_solid"] & (fr["units"] > 0) & (fr["boites"] > 0) & fr["prix_1"].notna() & fr["molecule"].notna()]
    if segment is not None:
        r = r[r["segment"] == segment]
    q = r["boites"] * r["units"]
    v = r["boites"] * r["prix_1"]
    return (v.groupby(r["molecule"]).sum() / q.groupby(r["molecule"]).sum()).rename("p_fr_now")


__all__ = [
    "molecule_us", "harmonise", "substance_fr", "french_number", "units_per_box", "us_tables", "fr_tables",
    "cells_us", "cells_fr", "match", "laspeyres", "paasche", "fisher", "jevons", "mixed_index",
    "BASKETS", "molecule_baskets", "indices", "bootstrap", "fda_names", "nadac_molecules", "mfp_table",
    "fr_current_unit_price",
]
