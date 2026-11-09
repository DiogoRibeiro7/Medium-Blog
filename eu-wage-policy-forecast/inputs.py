"""Scenario inputs that do not come from the data snapshot: announced policy paths, published
projections and published estimates. Every number carries its source; forecast.py and
experiments.py read them from here and nowhere else.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------------------------
# Portugal: national minimum wage (euro per month, 14 payments a year, value from January)
# ---------------------------------------------------------------------------------------------
# 2022-2026 are law (Eurostat earn_mw_cur, converted back to 14 payments, checked in empirics.py).
# 2027: 970 in the Acordo Tripartido sobre Valorizacao Salarial e Crescimento Economico 2025-2028
#   (1 Oct 2024) and in the Relatorio do Orcamento do Estado 2027 (8 Oct 2026).
# 2028: 1020 in the same agreement. The agreement ends in 2028.
# 2029: the Programa do XXV Governo (June 2025) targets 1100 at the end of the legislature.
# UGT asked for 1000 in 2027 (16 Sep 2026); CGTP for 1100 from January 2027.
MW_HISTORY = {2022: 705.0, 2023: 760.0, 2024: 820.0, 2025: 870.0, 2026: 920.0}

MW_PATHS = {
    # name: (label, {year: euro, or None for "grows with the median that year"})
    # the reference: the minimum wage grows with the projected median, so its bite stays at the 2026 level
    "constant_bite": ("Indexed to the median from 2027", {2027: None, 2028: None, 2029: None}),
    "agreement": ("Agreement, then +50 in 2029", {2027: 970.0, 2028: 1020.0, 2029: 1070.0}),
    "programme": ("Agreement, then 1,100 in 2029", {2027: 970.0, 2028: 1020.0, 2029: 1100.0}),
    "front_loaded": ("1,000 in 2027, 1,100 in 2029", {2027: 1000.0, 2028: 1050.0, 2029: 1100.0}),
    "union_2027": ("1,100 in 2027, then indexed", {2027: 1100.0, 2028: None, 2029: None}),
}
# 2029 values scanned for the risk curve, with 2027 and 2028 on the agreement
MW_2029_GRID = tuple(range(1020, 1161, 5))   # from a freeze at the 2028 value
FORECAST_YEARS = (2027, 2028, 2029)

# ---------------------------------------------------------------------------------------------
# Portugal: nominal compensation per employee (national accounts), % growth, published projections
# ---------------------------------------------------------------------------------------------
# Ministerio das Financas, Relatorio OE2027 (8 Oct 2026): 4.7 (2026), 4.5 (2027).
# Banco de Portugal, Boletim Economico June 2026: 4.0 (2026), 3.8 (2027), 3.6 (2028); the projection
#   includes minimum-wage updates of 5.7%, 5.4% and 5.2% (footnote 9).
# Conselho das Financas Publicas, Perspetivas Economicas e Orcamentais 2026-2030, update of Sep 2026
#   (Relatorio 07/2026, Quadro 2): 4.9, 3.9, 3.2, 2.8 (2026-2029).
WAGE_PROJECTIONS = {
    "Ministry of Finance (Oct 2026)": {2026: 4.7, 2027: 4.5},
    "Banco de Portugal (Jun 2026)": {2026: 4.0, 2027: 3.8, 2028: 3.6},
    "Conselho das Financas Publicas (Sep 2026)": {2026: 4.9, 2027: 3.9, 2028: 3.2, 2029: 2.8},
}
# National accounts growth in the first half of 2026: 4.7% and 4.2% year on year in Q1 and Q2
# (Relatorio OE2027). The 2026 draw is centred on the projections with a smaller spread.
SD_2026_PP = 0.4

# Portuguese employees (persons, national accounts 2025) are read from the snapshot.

# ---------------------------------------------------------------------------------------------
# Portugal: the minimum wage against the median base wage (Banco de Portugal)
# ---------------------------------------------------------------------------------------------
# Banco de Portugal, Boletim Economico June 2026, Caixa 5 (Social Security microdata, private sector):
# Kaitz index 91% in 2025, 87% in 2019.
KAITZ_BASE = {2019: 0.87, 2025: 0.91}

# Quadros de Pessoal share of full-time employees paid the minimum wage (GEP/MTSSS), as used in the
# earlier companion: 14.2% (2013), 23.9% (2020), 20.7% (2023).
SHARE_AT_MW_QP = {2013: 0.142, 2020: 0.239, 2023: 0.207}

# ---------------------------------------------------------------------------------------------
# Employment: published own-wage elasticities (OWE) of employment for minimum-wage increases
# ---------------------------------------------------------------------------------------------
# Dube and Zipperer (2024), NBER WP 32925 and the OWE repository: median OWE -0.14 across 88 studies
# (-0.13 across the 72 published in journals); median -0.04 for broad groups of workers in studies
# published since 2010 (Dube and Zipperer via EPI, 2024).
OWE = {"broad groups since 2010": -0.04, "all 88 studies": -0.14}

# ---------------------------------------------------------------------------------------------
# Pay transparency: Portuguese draft transposition of Directive (EU) 2023/970
# ---------------------------------------------------------------------------------------------
# Draft published in Separata do BTE n.o 26, 5 Aug 2026 (CMS, ECIJA summaries, Aug 2026):
# reporting by employers with 250+ (annual) and 150-249 (every three years) from 7 June 2027 on 2026
# data, and 50-149 (every three years) from 7 June 2031. Pay range before the contract is signed,
# a ban on salary-history questions and void pay-secrecy clauses apply to every employer.
REPORTING_THRESHOLDS = {2027: 150, 2031: 50}

# Published effects of transparency mandates (used as an evidence range, not estimated here):
EVIDENCE = {
    # Cullen and Pakzad-Hurson (2023, Econometrica): US state laws protecting pay discussion, wages fall ~2%.
    "us_wage_level": -0.02,
    # Bennedsen, Simintzi, Tsoutsoura and Wolfenzon (2022, J. Finance): Denmark 2006, firms with 35+
    # employees; gender gap -1.9 pt, 13% of the pre-law gap; male wage growth -1.7 pt.
    "dk_gap_share": -0.13, "dk_male_wage": -0.017,
    # Blundell, Duchini, Simion and Turrell (2025, AEJ: Policy): UK 2018, firms with 250+ employees;
    # gap -19%, entirely through slower male pay growth (about -3%).
    "uk_gap_share": -0.19, "uk_male_wage": -0.03,
}
# Gender pay gap in Portugal, 2023 (Government, exposicao de motivos of the draft, via Sabado):
# 15.4% on total earnings, 12.5% on base pay. Eurostat's unadjusted hourly gap is read from the snapshot.
PT_GAP_2023 = {"total earnings": 0.154, "base pay": 0.125}

# Calibration target for the model's equal-pay rule: the US wage response, with 1% and 3% as bounds
TRANSPARENCY_TARGETS = (-0.01, -0.02, -0.03)
TRANSPARENCY_TARGET_YEARS = (3, 4, 5)        # years after the law over which the target is averaged

# ---------------------------------------------------------------------------------------------
# EU: Directive (EU) 2022/2041, Article 5(4): indicative reference values such as 60% of the gross
# median wage and 50% of the gross average wage (upheld by the Court of Justice, C-19/23, 11 Nov 2025).
# ---------------------------------------------------------------------------------------------
REFERENCE_KAITZ = 0.60
