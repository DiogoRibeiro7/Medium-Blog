"""Describe the CPS 1988 population used in experiment 1.

How far is the Mincer equation from the population regression? Adds a squared
education term and an education x experience interaction to the population
regression (HC1 standard errors).

Run:  python 00_population.py
"""

import numpy as np
import statsmodels.api as sm

from common import load_cps, write

c = load_cps()
base = ["educ", "exper", "exper2", "afam", "smsa"]
X = sm.add_constant(c[base])
r0 = sm.OLS(c.logwage, X).fit(cov_type="HC1")
X1 = X.assign(educ2=c.educ ** 2, educ_exper=c.educ * c.exper)
r1 = sm.OLS(c.logwage, X1).fit(cov_type="HC1")
out = [f"CPS 1988 population: N = {len(c)}; mean educ {c.educ.mean():.3f} (SD {c.educ.std():.3f}); "
       f"share with exactly 12 years {np.mean(c.educ == 12):.3f}",
       f"Mean educ: African-American {c.educ[c.afam == 1].mean():.2f}, others "
       f"{c.educ[c.afam == 0].mean():.2f}",
       "Mincer equation (population, HC1): " + ", ".join(
           f"{k} {v:.4f}" for k, v in r0.params.items()) + f"; R2 {r0.rsquared:.3f}",
       f"Adding educ^2 and educ x exper: educ^2 {r1.params['educ2']:.5f} (t {r1.tvalues['educ2']:.1f}), "
       f"educ x exper {r1.params['educ_exper']:.5f} (t {r1.tvalues['educ_exper']:.1f}); R2 "
       f"{r1.rsquared:.3f}",
       f"Implied return to schooling at 12 years of education: experience 5 years "
       f"{r1.params['educ'] + 2 * 12 * r1.params['educ2'] + 5 * r1.params['educ_exper']:.4f}, "
       f"experience 30 years "
       f"{r1.params['educ'] + 2 * 12 * r1.params['educ2'] + 30 * r1.params['educ_exper']:.4f}"]
write("00_population.txt", "\n".join(out))
