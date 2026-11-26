"""Figures transcribed by hand from operator, regulator and press sources, each with the document it comes from.

These are the numbers no statistical API publishes: the metro's own counts by ticket type, its train-km and
circulations, the efficacy indicators of its concession contract, the other operators' annual totals, a
calendar of closures, and motorway traffic around Lisbon. They were read on 26 November 2026. A value
reported only in the press is marked as such; run_analysis.py never mixes them with the INE series without
saying so in its output.
"""

from __future__ import annotations

AMT_ML_2024 = "https://amt-autoridade.pt/media/5617/2024_relatorio_desempenho_mlepe.pdf"
AMT_ML_2025 = "https://www.amt-autoridade.pt/media/5945/2025_relatorio_desempenho_mlepe.pdf"
AMT_METRO_2018_2020 = "https://www.amt-autoridade.pt/media/3324/relatorio_dos_sistemas_de_metro_2018-2020.pdf"
RACC_2025 = "https://www.metrolisboa.pt/institucional/wp-content/uploads/sites/2/2026/06/RACC-2025_VF.pdf"
RTECC_2026Q1 = "https://www.metrolisboa.pt/institucional/wp-content/uploads/sites/2/2026/06/RTECC-1T2026.pdf"
RTECC_2026Q2 = "https://www.metrolisboa.pt/institucional/wp-content/uploads/sites/2/2026/08/RTECC_2oT2026.pdf"
IMT_Q2_2026 = "https://www.imt-ip.pt/wp-content/uploads/2026/09/RelatoriodeTrafego-2oTrimestrede2026.pdf"

# --- Metropolitano de Lisboa: supply and output, from the annual reports to the regulator (AMT) -----------------
# Regulation (EC) 1370/2007 reports. Units as printed: passengers (number), passenger-km (thousands),
# train-km ("comboios x km", thousands), place-km (millions), circulations (number), indices (%).
ML_ANNUAL = {
    2019: {
        "source": AMT_METRO_2018_2020,
        "passengers_m": 183.8,  # millions, as printed
        "train_km_thousand": 5100.0,  # 5.1 million, read from a rounded figure label
        "place_km_million": 3536.0,
        "network_km": 44.5,
        "stations": 50,
    },
    2024: {
        "source": AMT_ML_2024,
        "passengers": 176_727_285,
        "passenger_km_thousand": 922_919,
        "train_km_thousand": 5053.36,
        "place_km_million": 3767.71,
        "circulations": 473_366,
        "regularity_index": 97.95,
        "punctuality_ip5": 99.89,
        "occupancy_rate": 24.50,
        "network_km": 44.459,
        "cars": 333,
        "fleet_age": 25.8,
        "tariff_revenue_keur": 87_910,
        "revenue_single_ticket_keur": 16_759,
        "revenue_zapping_keur": 14_034,
        "revenue_navegante_metropolitano_keur": 27_546,
        "revenue_navegante_lisboa_keur": 8_834,
    },
    2025: {
        "source": AMT_ML_2025,
        "passengers": 173_475_554,
        "passenger_km_thousand": 899_262,
        "train_km_thousand": 4959.97,
        "place_km_million": 3809.89,
        "circulations": 458_237,
        "regularity_index": 95.95,
        "punctuality_ip5": 99.89,
        "occupancy_rate": 23.60,
        "fraud_detection_rate": 2.52,
        "network_km": 44.459,
        "cars": 333,
        "fleet_age": 26.8,
        "satisfaction_0_10": 5.75,
        "complaints": 5_034,
        "tariff_revenue_keur": 82_724,
        "revenue_single_ticket_keur": 13_893,
        "revenue_zapping_keur": 12_310,
        "revenue_navegante_metropolitano_keur": 25_445,
        "revenue_navegante_lisboa_keur": 8_687,
    },
}

# --- Metropolitano de Lisboa: the company's own counts (press, from the annual report and data sent to newspapers)
ML_COUNTS = {
    "2024_valid_ticket": (171_898_118, "Lusa via Executive Digest, 1 Jul 2025, from the 2024 report and accounts"),
    "2024_fare_evasion_estimate": (4_829_166, "same"),
    "2025_valid_ticket_m": (169.2, "Jornal de Notícias, Jan 2026, data sent by the company"),
    "2025_lost_vs_2024_m": (2.7, "same"),
    "occasional_2024_m": (31.6, "same"),
    "occasional_2025_m": (28.5, "same"),
    "customer_paid_passes_2024_m": (105.2, "same"),
    # JN prints 'para 96...' in the part of the article that could be read; the decimal is not known.
    "customer_paid_passes_2025_m_low": (96.0, "same, lower end of the printed value"),
    "customer_paid_passes_2025_m_high": (96.9, "same, upper end of the printed value"),
    "jan_sep_2026_m": (123.8, "Público, 25 Nov 2026, company data: -3.9% on Jan-Sep 2025"),
    "jan_sep_2026_change": (-3.9, "same"),
}

# --- Concession-contract efficacy indicators (%), Metropolitano de Lisboa ------------------------------------
# DMC fleet availability, FSC share of scheduled circulations run, REG share of circulations within the
# programmed interval. VR is the contractual reference. 2025 by quarter (RACC 2025, Table 1); 2026 by month
# (RTECC Q1 and Q2 2026, Table 1).
KPI_REFERENCE = {"DMC": 85.89, "FSC": 97.00, "REG": 80.00}
KPI_2025_QUARTER = {
    "source": RACC_2025,
    "DMC": [85.54, 90.75, 83.32, 81.05],
    "FSC": [97.88, 97.44, 95.91, 96.74],
    "REG": [86.62, 77.48, 76.48, 78.72],
}
KPI_2026_MONTH = {
    "source": [RTECC_2026Q1, RTECC_2026Q2],
    "DMC": [76.99, 79.06, 77.91, 77.43, 76.34, 82.01],
    "FSC": [98.07, 98.25, 98.15, 97.40, 97.05, 95.90],
    "REG": [78.16, 78.37, 76.44, 78.06, 75.15, 75.19],
}
KPI_CAUSES = (
    "RACC 2025 and RTECC 2026: concentrated mandatory revision of the motor bogies synchronised their "
    "maintenance cycles and recurring technical problems forced vehicles out of service; driver absenteeism; "
    "contingency measures since Q4 2025; from 2025 six-car trains all day"
)

# --- Days the network was shut, fully or partly (share of that day's service lost, an assumption) -------------
# Partial strikes closed the network from the start of service to 10:30; the assumed loss is the share of a
# weekday's entries before 10:30 (0.35, tested from 0.25 to 0.45). The blackout of 28 April 2025 stopped the
# metro from about 11:30; assumed loss 0.6. Full-day closures lose 1.
CLOSURES_LISBOA = {
    "2024-12-03": (0.35, "partial strike, closed until 10:30 (Lusa, 3 Dec 2024)"),
    "2024-12-10": (0.35, "partial strike, closed until 10:30 (Lusa, 3 Dec 2024)"),
    "2025-04-28": (0.60, "Iberian blackout, service suspended from late morning (Lusa, 28 Apr 2025)"),
    "2025-09-09": (0.35, "partial strike, closed until 10:30 (Lusa, 9 Sep 2025)"),
    "2025-09-11": (0.35, "partial strike, closed until 10:30 (Lusa, 11 Sep 2025)"),
    "2025-12-11": (1.00, "general strike, network closed all day (Lusa, 10 Dec 2025)"),
    "2026-04-09": (1.00, "24-hour strike, network closed (Lusa, 9 Apr 2026)"),
    "2026-06-03": (1.00, "general strike, network closed all day (Lusa, 1-3 Jun 2026)"),
}
# Metro do Porto ran a reduced service on the trunk section during the general strikes; assumed loss 0.6.
CLOSURES_PORTO = {
    "2025-04-28": (0.60, "Iberian blackout"),
    "2025-12-11": (0.60, "general strike, reduced service (assumed)"),
    "2026-06-03": (0.60, "general strike, trunk section and part of the Yellow line only (Lusa, 3 Jun 2026)"),
}
CLOSURES_MST = {
    "2025-04-28": (0.60, "Iberian blackout"),
    "2025-12-11": (0.60, "general strike (assumed)"),
    "2026-06-03": (0.60, "general strike (assumed)"),
}
# Works that closed part of the Lisbon network (not adjusted for; flagged in the output).
WORKS_LISBOA = [
    ("2024-08", "3-19 Aug 2024: sections and stations of the Green and Yellow lines closed for the Circular line works"),
    ("2026-05", "2-5 May 2026: Rossio-Baixa-Chiado-Cais do Sodré closed on the Green line"),
    ("2026-08", "8-26 Aug 2026: Cais do Sodré closed; Blue line reduced hours on August weekends for signalling tests"),
]

# --- Other operators in the Lisbon metro area (press releases and reports; not the INE survey) -----------------
OTHER_OPERATORS = {
    "carris_metropolitana_2024_m": (174.0, "Lusa via ECO, 21 Jan 2025"),
    "carris_metropolitana_2025_m": (194.0, "Lusa via Sábado, 27 Jan 2026"),
    "carris_metropolitana_2025_into_lisbon_m": (69.8, "same"),
    "carris_metropolitana_jan_may_2026_change": (7.2, "Lusa via Renascença, 15 Jun 2026"),
    "carris_2025_m": (131.8, "Carris, Relatório de Sustentabilidade 2025"),
    "carris_2025_change": (-1.5, "same; attributed mainly to the closure of the Santa Justa lift and the other historic lines"),
    "cp_lisbon_suburban_2025_validations": (139_222_413, "24notícias, 2026, from CP data"),
    "cp_lisbon_suburban_2025_change": (-2.1, "same"),
    "metro_porto_2025_validations_m": (94.54, "Metro do Porto report and accounts via Observador, 28 May 2026"),
    "navegante_share_of_trips": (90.0, "TML via Lusa/ECO, 18 Dec 2025; pass price unchanged for the 7th year in 2026"),
}

# --- Other cities ----------------------------------------------------------------------------------------------
MADRID_METRO = {"2024": 715_208_882, "2025": 736_874_012, "source": "CRTM press release, 16 Jan 2026"}

# --- Roads around Lisbon: weighted-average monthly daily traffic (vehicles a day), IMT quarterly report --------
# Whole-motorway weighted averages; A2 starts at the 25 de Abril bridge, A12 at the Vasco da Gama bridge.
MOTORWAY_TMDM = {
    "source": IMT_Q2_2026,
    "A1": {"2025": [44_267, 45_098, 45_337], "2026": [44_378, 45_547, 45_726]},
    "A2": {"2025": [25_800, 25_372, 30_789], "2026": [25_958, 26_305, 30_528]},
    "A5": {"2025": [96_232, 104_663, 100_157], "2026": [98_779, 102_085, 100_079]},
    "A12": {"2025": [46_070, 48_824, 49_993], "2026": [48_806, 48_817, 50_044]},
}
ROAD_VKM_H1 = {
    "2025_bn": 12.8,
    "2026_bn": 13.2,
    "source": "Zero, citing IMT, main roads, first half of the year (RTP, 22 Sep 2026)",
}
