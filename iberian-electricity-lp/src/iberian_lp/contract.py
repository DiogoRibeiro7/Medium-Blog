"""Constants fixed by the registered plan (METHODS.md) or by its change log."""

from __future__ import annotations

from datetime import date

# Section 3: the first year of 15-minute periods, and the days drawn from it.
FIRST_DAY = date(2025, 10, 1)
LAST_DAY = date(2026, 9, 30)
DAYS_PER_MONTH = 4
DAYS_SEED = 20261028

# Section 3: the files fetched for each day, from omie.es (data: OMIE, OMI-Polo Español, S.A.).
KINDS = ("curva_pbc", "marginalpdbc", "capacidad_inter_pbc")
DOWNLOAD_URL = "https://www.omie.es/es/file-download?parents={kind}&filename={name}"
USER_AGENT = "Mozilla/5.0 (research; Medium-Blog-research/0.1)"
SOURCE = "OMIE (OMI-Polo Español, S.A.), omie.es"

# The file specification read before the plan (section 2).
SPEC = "Modelo de Ficheros para la distribución pública de información, versión 1.37"
ZONES = ("MI", "ES", "PT")                 # the joint Iberian market; Spain; Portugal
SIMPLE = "S"
NON_CONVEX = ("C01", "C02", "C04")         # simple block, scalable complex, exclusive block
FLOWS = ("Imp FR", "Exp FR", "Imp PT", "Exp PT", "Imp ES", "Exp ES")
FRANCE_BORDER = "3"                        # capacidad_inter_pbc: 2 Portugal, 3 France

# Sections 6 and 7.
BOOTSTRAP_DRAWS = 9_999
BOOTSTRAP_SEED = 20261029
LEVEL = 0.95                               # reported intervals
SIMULTANEOUS_LEVEL = 1 - 0.05 / 2          # Q2 and Q3 (Bonferroni over the two)
E1_TOLERANCE = 0.01                        # EUR/MWh at each end of Λ_R
E1_TOLERANCE_SENSITIVITY = 0.1             # section 7
E1_THRESHOLD = 0.99
VOLUME_TOLERANCE_MW = 0.1
MOVED = 0.01                               # E2: |δ| above this, in EUR/MWh
E3_MW = 1_000                              # one more sell step at 0 EUR/MWh
E3_SENSITIVITY_MW = (500, 2_000)           # section 7
MIDDAY_HOURS = range(11, 18)               # OMIE's hours 11-17: 10:00 to 17:00
