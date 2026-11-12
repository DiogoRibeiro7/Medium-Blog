"""Portugal's external reference pricing rule and the price conversions it relies on.

Portugal sets the maximum price of a new non-generic medicine, and revises it every year, by looking
at what the same medicine costs in four reference countries (Spain, France, Italy and Belgium for
2025-2026, Portaria 394/2025/1). The rule, from Decreto-Lei 97/2015 and Portaria 195-C/2015 as
implemented in Infarmed's own calculation form for 2026, is:

  1. In each reference country take the ex-factory price (PVA, "preco de venda ao armazenista") of the
     same medicine, in the same or the most similar presentation. Countries publish retail prices, so
     the PVA is backed out with fixed factors that Infarmed publishes (es_pva, it_pva; Belgium
     publishes the ex-factory price; France's comes from its retail price, fr_pfht).
  2. Bring each reference price to the Portuguese presentation: multiply by the ratio of strengths
     and by the ratio of pack sizes, and correct the pack ratio with a factor that gives a little
     back on larger packs and takes a little on smaller ones (y_factor), rounding to the cent
     (adjust).
  3. The Portuguese maximum PVA is the average of the adjusted reference prices for the ambulatory
     market (the lowest of them for hospital medicines) (pt_cap).
  4. The retail price (PVP) adds the wholesale and pharmacy margins, Infarmed's fee and VAT, in six
     tiers by PVA (pt_pvp, inverted by pt_pva).

The factors below are the ones in Infarmed's 2026 form and its note on margins and factors.

France. The retail price before the per-box dispensing fee (PPTTC, the first price in the public
medicines database) is the manufacturer price (PFHT) plus the wholesaler's margin (6.93 percent of the
PFHT, at least 0.30 and at most 32.50 euros per box), plus the pharmacy's smoothed degressive margin
(10, 7, 5.5 and 5 percent on the parts of the PFHT up to 1.91, 22.90, 150 and 1930 euros, nothing
above), all times 1.021 for VAT. fr_pfht inverts this.

Names. Reference countries spell international non-proprietary names in their own languages
("empagliflozine", "empagliflozina", "empagliflozin"). stem reduces a name to a common form by
dropping accents, salt and hydrate words in five languages, a final vowel, and the spellings that
differ (th, ph, y). Strengths are read from labels as milligrams (strengths).
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np

# --------------------------------------------------------------------------------------- Portugal
# PVP = a * PVA + b, by PVA tier (Infarmed, 2026). The tiers meet: 1.1475 * 5 + 0.94 = 6.68, and so on.
PT_TIERS = ((5.00, 1.1475, 0.94), (7.00, 1.1460, 1.95), (10.00, 1.1439, 2.66),
            (20.00, 1.1393, 4.17), (50.00, 1.1316, 8.00), (np.inf, 1.1051, 12.73))
PT_RAP_EXEMPT_PVP = 30.0          # the 2026 annual revision leaves retail prices up to 30 euros alone
PT_RAP_MAX_CUT = 0.20             # and cuts the others by at most 20 percent (Portaria 394/2025/1, art. 3)


def y_factor(r: float) -> float:
    """Correction applied to the pack ratio r = Portuguese pack / reference pack."""
    if r < 1 / 3:
        return 1.05
    if r < 1:
        return 0.985 + 0.015 / r
    if r == 1:
        return 1.0
    if r <= 3:
        return 1.015 - 0.015 * r
    return 0.95


def adjust(pva_ref: float, dose_pt: float, dose_ref: float, pack_pt: float, pack_ref: float) -> float:
    """A reference country's PVA brought to the Portuguese presentation, as in Infarmed's form."""
    g = dose_pt / dose_ref
    h = pack_pt / pack_ref
    return round(pva_ref * g * h * y_factor(h) + 1e-9, 2)


def pt_cap(adjusted, rule: str = "mean") -> float:
    """Maximum Portuguese PVA from the adjusted reference prices of the countries that sell it."""
    v = np.array([x for x in adjusted if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return np.nan
    if rule == "mean":
        return round(float(v.mean()) + 1e-9, 2)
    if rule == "min":
        return float(v.min())
    if rule == "median":
        return round(float(np.median(v)) + 1e-9, 2)
    if rule == "max":
        return float(v.max())
    raise ValueError(rule)


def pt_pvp(pva: float) -> float:
    for top, a, b in PT_TIERS:
        if pva <= top + 1e-9:
            return round(a * pva + b + 1e-9, 2)
    raise ValueError(pva)


def pt_pva(pvp: float) -> float:
    """Inverse of pt_pvp, with the tier limits expressed in retail prices as in Infarmed's form."""
    for top, a, b in PT_TIERS:
        if pvp <= round(a * top + b, 2) + 1e-9:
            return (pvp - b) / a
    raise ValueError(pvp)


# ------------------------------------------------------------------------------- reference countries
def es_pva(pvp_iva: float, clinical: bool = False) -> float:
    """Spain: ex-factory price from the retail price with VAT (Infarmed's factors)."""
    if clinical:
        return pvp_iva / 1.169591 / 1.04
    if pvp_iva > 578.1464:
        return pvp_iva / 1.04 - 55.91
    if pvp_iva > 260.9464:
        return pvp_iva / 1.04 - 50.91
    if pvp_iva > 143.0416:
        return pvp_iva / 1.04 - 45.91
    return pvp_iva / 1.561083


def it_pva(pvp: float, classe: str = "A") -> float:
    """Italy: ex-factory price from the retail price, class A (reimbursed) or C (not reimbursed)."""
    return pvp * 0.6091 if classe.strip().upper() == "C" else pvp / 1.6504


FR_WHOLESALE = (0.0693, 0.30, 32.50)
FR_MDL = ((0.0, 1.91, 0.10), (1.91, 22.90, 0.07), (22.90, 150.0, 0.055), (150.0, 1930.0, 0.05))
FR_VAT = 1.021


def fr_ppttc(pfht):
    """France: retail price before the dispensing fee, from the manufacturer price."""
    p = np.asarray(pfht, dtype=float)
    rate, lo, hi = FR_WHOLESALE
    wholesale = np.clip(p * rate, lo, hi)
    mdl = sum(r * np.clip(np.minimum(p, top) - bottom, 0, None) for bottom, top, r in FR_MDL)
    return np.floor((p + wholesale + mdl) * FR_VAT * 100 + 0.5 + 1e-7) / 100


def fr_pfht(ppttc):
    """Inverse of fr_ppttc. Returns the cent value of PFHT that gives back the published price when
    there is one (the published prices were computed from such a value), and otherwise the exact
    solution of the continuous formula. Vectorised."""
    y = np.atleast_1d(np.asarray(ppttc, dtype=float))
    # The formula is increasing and piecewise linear; bisection on the continuous version first.
    lo = np.zeros_like(y)
    hi = np.maximum(y, 1.0)

    def cont(p):
        rate, wlo, whi = FR_WHOLESALE
        mdl = sum(r * np.clip(np.minimum(p, top) - bottom, 0, None) for bottom, top, r in FR_MDL)
        return (p + np.clip(p * rate, wlo, whi) + mdl) * FR_VAT

    for _ in range(60):
        mid = (lo + hi) / 2
        below = cont(mid) < y
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    x = (lo + hi) / 2
    out = x.copy()
    for k, xi in enumerate(x):
        if not np.isfinite(xi):
            continue
        cands = np.round(np.arange(np.floor(xi * 100) - 3, np.ceil(xi * 100) + 4) / 100, 2)
        hit = cands[np.isclose(fr_ppttc(cands), y[k], atol=1e-9)]
        if len(hit):
            out[k] = hit[np.argmin(np.abs(hit - xi))]
    return out if np.ndim(ppttc) else float(out[0])


# ------------------------------------------------------------------------------------------- names
SALTS = set("""
chlorhydrate dichlorhydrate hydrochloride hcl hidrocloruro clorhidrato cloridrato dicloridrato
sodique sodium sodico sodica sodio disodique disodico disodica potassique potassium potasico potasica potassico potassica potassio calcique calcium calcico calcica calcio
magnesien magnesienne magnesium magnesico magnesica magnesio besilate besylate besilato maleate maleato mesilate mesylate mesilato
dimesilate dimesilato tartrate tartrato bitartrate succinate succinato fumarate fumarato hemifumarate emifumarato
hemifumarato citrate citrato phosphate fosfato sulfate sulfato solfato acetate acetato bromhydrate bromidrato
hidrobromuro propanediol propanediolo propanodiol monohydrate monohydratee monohidrato monoidrato dihydrate
dihydratee dihidrato diidrato trihydrate trihydratee trihidrato triidrato hemihydrate hemihidrato emiidrato
sesquihydrate sesquihidrato anhydre anhydrous anhidro anidro base tosilate tosilato tosylate malate malato
meglumine meglumina medoxomil medoxomilo axetil axetilo mofetil mofetilo lysine lisina erbumine erbumina
arginine arginina oxalate oxalato nitrate nitrato lactate lactato gluconate gluconato hyclate hiclato
de d du di del della y e et and sous forme
""".split())


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def stem_word(w: str) -> str:
    w = w.replace("th", "t").replace("ph", "f").replace("y", "i")
    return w[:-1] if len(w) > 4 and w[-1] in "aeo" else w


def stem(name: str) -> str:
    """Common key for a molecule name in English, French, Spanish, Italian or Dutch; combinations
    (written with +, /, ',' or ' e ') become sorted components joined by ' + '."""
    s = strip_accents(str(name)).lower()
    parts = re.split(r"\s*[+/,;]\s*|\s+(?:e|et|y|and|en)\s+", s)
    out = []
    for p in parts:
        words = [stem_word(t) for t in re.findall(r"[a-z]+", p) if t not in SALTS and len(t) > 1]
        if words:
            out.append(" ".join(words))
    return " + ".join(sorted(set(out)))


UNIT_MG = {"mg": 1.0, "g": 1000.0, "mcg": 0.001, "microgrammes": 0.001, "microgramos": 0.001,
           "microgrammi": 0.001, "ug": 0.001, "µg": 0.001, "μg": 0.001, "microgramme": 0.001,
           "microgramo": 0.001, "microgrammo": 0.001}
_UNITS = sorted(UNIT_MG, key=len, reverse=True)
_NUM = r"\d+(?:[.,]\d+)*"
_STRENGTH = re.compile(rf"((?:{_NUM}\s*(?:{'|'.join(map(re.escape, _UNITS))})?\s*\+\s*)*{_NUM})\s*"
                       rf"({'|'.join(map(re.escape, _UNITS))})(?![a-z])", re.IGNORECASE)


def _num(x: str) -> float:
    x = x.strip()
    if "," in x and "." in x:          # 1.000,5
        x = x.replace(".", "").replace(",", ".")
    elif "," in x:
        x = x.replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+", x):   # 1.000 mg
        x = x.replace(".", "")
    return float(x)


def strengths(label: str) -> tuple[float, ...]:
    """Strengths in a label, in milligrams, sorted: 'ENTRESTO 49 mg/51 mg' and
    'SACUBITRIL+VALSARTAN 49+51MG' both give (49.0, 51.0). Per-volume strengths are not handled."""
    s = strip_accents(str(label)).lower().replace("µ", "u")
    s = re.sub(r"(\d)\s*mg\s*/\s*(\d)", r"\1 mg + \2", s)
    out: list[float] = []
    for m in _STRENGTH.finditer(s):
        unit = UNIT_MG.get(m.group(2).lower(), 1.0)
        for piece in re.split(r"\s*\+\s*", m.group(1)):
            val = re.match(rf"({_NUM})\s*([a-z]+)?", piece)
            if not val:
                continue
            u = UNIT_MG.get((val.group(2) or "").lower(), unit)
            out.append(round(_num(val.group(1)) * u, 6))
    return tuple(sorted(out))


# -------------------------------------------------------------------------------------- matching
def reference_price(table, brand: str, strengths: tuple, units: int):
    """The reference price of a presentation in one country, brought to that presentation.

    Infarmed asks for the same medicine in the same or the most similar presentation. Here the same
    medicine is the same brand with the same number of active substances; the most similar
    presentation has the same strengths if there is one (otherwise, for a single substance, the
    nearest strength), and then the nearest pack size, the larger pack on a tie. The price is then
    adjusted for strength and pack as in Infarmed's form.

    table: rows of one country with columns brand, strengths, units, pva, label.
    Returns (adjusted price, kind, label of the reference pack) or None. kind is "identical",
    "same strength" (other pack size) or "other strength".
    """
    if not strengths:
        return None
    cand = table[(table["brand"] == brand) & (table["strengths"].map(len) == len(strengths))]
    if cand.empty:
        return None
    same = cand[cand["strengths"] == strengths]
    if not same.empty:
        cand, kind = same, "same strength"
    elif len(strengths) == 1:
        gap = np.abs(np.log(cand["strengths"].map(lambda s: s[0]).to_numpy(dtype=float) / strengths[0]))
        cand, kind = cand[gap == gap.min()], "other strength"
    else:
        return None
    gap = np.abs(np.log(units / cand["units"].to_numpy(dtype=float)))
    cand = cand[gap == gap.min()].sort_values(["units", "pva"], ascending=[False, True])
    best = cand.iloc[0]
    dose_ref = best["strengths"][0] if kind == "other strength" else strengths[0]
    adjusted = adjust(float(best["pva"]), strengths[0], dose_ref, units, int(best["units"]))
    if kind == "same strength" and int(best["units"]) == units:
        kind = "identical"
    return adjusted, kind, best["label"]
