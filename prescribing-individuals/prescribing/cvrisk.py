"""10-year risk of a first atherosclerotic cardiovascular event: the Pooled Cohort Equations.

Goff et al., 2013 ACC/AHA Guideline on the Assessment of Cardiovascular Risk
(Circulation 2014; 129: S49-S73), Table A. Valid for ages 40 to 79 without prior
cardiovascular disease. As the guideline recommends, the equations for white
adults are used for people of other races and ethnicities.

The implementation reproduces the guideline's worked example (age 55, total
cholesterol 213 mg/dL, HDL 50 mg/dL, untreated systolic 120 mmHg, non-smoker,
no diabetes) to within 0.1 percentage point: the guideline prints 2.1% for white
women, 3.0% for African American women, 5.3% for white men and 6.1% for African
American men; this code gives 5.38% for white men because the guideline rounds
the intermediate sum (60.69 against 60.6995) before exponentiating (see
tests/test_cvrisk.py).
"""

from __future__ import annotations

import numpy as np

COEF = {
    "white_female": dict(lnage=-29.799, lnage2=4.884, lntc=13.540, lnage_lntc=-3.114, lnhdl=-13.578,
                         lnage_lnhdl=3.149, lnsbp_t=2.019, lnage_lnsbp_t=0.0, lnsbp_u=1.957, lnage_lnsbp_u=0.0,
                         smk=7.574, lnage_smk=-1.665, dm=0.661, mean=-29.18, s0=0.9665),
    "black_female": dict(lnage=17.114, lnage2=0.0, lntc=0.940, lnage_lntc=0.0, lnhdl=-18.920,
                         lnage_lnhdl=4.475, lnsbp_t=29.291, lnage_lnsbp_t=-6.432, lnsbp_u=27.820,
                         lnage_lnsbp_u=-6.087, smk=0.691, lnage_smk=0.0, dm=0.874, mean=86.61, s0=0.9533),
    "white_male": dict(lnage=12.344, lnage2=0.0, lntc=11.853, lnage_lntc=-2.664, lnhdl=-7.990,
                       lnage_lnhdl=1.769, lnsbp_t=1.797, lnage_lnsbp_t=0.0, lnsbp_u=1.764, lnage_lnsbp_u=0.0,
                       smk=7.837, lnage_smk=-1.795, dm=0.658, mean=61.18, s0=0.9144),
    "black_male": dict(lnage=2.469, lnage2=0.0, lntc=0.302, lnage_lntc=0.0, lnhdl=-0.307,
                       lnage_lnhdl=0.0, lnsbp_t=1.916, lnage_lnsbp_t=0.0, lnsbp_u=1.809, lnage_lnsbp_u=0.0,
                       smk=0.549, lnage_smk=0.0, dm=0.645, mean=19.54, s0=0.8954),
}


def _one(c, age, tc, hdl, sbp, treated, smoker, diabetes):
    la, lt, lh, ls = np.log(age), np.log(tc), np.log(hdl), np.log(sbp)
    s = (c["lnage"] * la + c["lnage2"] * la ** 2 + c["lntc"] * lt + c["lnage_lntc"] * la * lt
         + c["lnhdl"] * lh + c["lnage_lnhdl"] * la * lh
         + np.where(treated, c["lnsbp_t"] * ls + c["lnage_lnsbp_t"] * la * ls,
                    c["lnsbp_u"] * ls + c["lnage_lnsbp_u"] * la * ls)
         + c["smk"] * smoker + c["lnage_smk"] * la * smoker + c["dm"] * diabetes)
    return 1 - c["s0"] ** np.exp(s - c["mean"])


def pce_risk(age, female, black, tc, hdl, sbp, treated, smoker, diabetes):
    """Vectorised 10-year risk. Inputs are clipped to the ranges the equations were built on."""
    age = np.clip(np.asarray(age, dtype=float), 40, 79)
    tc = np.clip(np.asarray(tc, dtype=float), 130, 320)
    hdl = np.clip(np.asarray(hdl, dtype=float), 20, 100)
    sbp = np.clip(np.asarray(sbp, dtype=float), 90, 200)
    female = np.asarray(female, dtype=bool)
    black = np.asarray(black, dtype=bool)
    treated = np.asarray(treated, dtype=bool)
    smoker = np.asarray(smoker, dtype=float)
    diabetes = np.asarray(diabetes, dtype=float)
    out = np.full(age.shape, np.nan)
    for key, mask in (("white_female", female & ~black), ("black_female", female & black),
                      ("white_male", ~female & ~black), ("black_male", ~female & black)):
        if mask.any():
            out[mask] = _one(COEF[key], age[mask], tc[mask], hdl[mask], sbp[mask], treated[mask],
                             smoker[mask], diabetes[mask])
    return out
