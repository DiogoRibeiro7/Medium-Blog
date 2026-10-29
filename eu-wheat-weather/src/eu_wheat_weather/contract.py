"""The data the study uses, fixed in one place (METHODS.md, section 3)."""

from __future__ import annotations

EU27: tuple[str, ...] = (
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU",
    "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK",
)

# Eurostat crop statistics by NUTS-2 region.
EUROSTAT_API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
CROPS_DATASET = "apro_cpshr"
CROP = "C1110"                      # common wheat and spelt
MAIZE = "C1500"                     # grain maize and corn-cob-mix (METHODS.md, section 9)
CROP_NAMES: dict[str, str] = {CROP: "wheat", MAIZE: "maize"}
AREA = "AR_THS_HA"                  # area, thousand hectares
PRODUCTION = "HPRD_HUMD_EU_THS_T"   # harvested production, EU standard humidity, thousand t
MEASURES: tuple[str, ...] = (AREA, PRODUCTION)
FIRST_YEAR, LAST_YEAR = 2000, 2024

# E-OBS daily gridded observations, ensemble mean on the 0.25 degree regular grid.
EOBS_VERSION = "v33.0e"
EOBS_BASE = ("https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/"
             "Grid_0.25deg_reg_ensemble/")
EOBS_FILES: dict[str, str] = {
    "tx": f"tx_ens_mean_0.25deg_reg_{EOBS_VERSION}.nc",
    "tn": f"tn_ens_mean_0.25deg_reg_{EOBS_VERSION}.nc",
    "rr": f"rr_ens_mean_0.25deg_reg_{EOBS_VERSION}.nc",
    "elev": f"elev_ens_0.25deg_reg_{EOBS_VERSION}.nc",
}

# GISCO NUTS-2 boundaries at 1:3 million, every version a crop series can be coded in.
NUTS_VERSIONS: tuple[int, ...] = (2006, 2010, 2013, 2016, 2021, 2024)
TARGET_NUTS = 2021
GISCO_BASE = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"


# Amendment A1 (METHODS.md, section 8.1): the GGCMI Phase 3 crop calendar, rainfed
# winter wheat. CC BY 4.0; Jägermeyr et al. (2021), doi:10.5281/zenodo.5062513.
CALENDAR_VERSION = "v1.01"
CALENDAR_FILE = f"wwh_rf_ggcmi_crop_calendar_phase3_{CALENDAR_VERSION}.nc4"
MAIZE_CALENDAR_FILE = f"mai_rf_ggcmi_crop_calendar_phase3_{CALENDAR_VERSION}.nc4"   # section 9
CALENDAR_BASE = "https://zenodo.org/records/5062513/files/"
CALENDAR_SNAPSHOT = f"snapshots/ggcmi-crop-calendar-{CALENDAR_VERSION}"

# Amendment A2 (METHODS.md, section 10.1): the MIRCA-OS map of maize, 2015, 5 arcminutes.
# CC BY 4.0; Kebede et al. (2025), doi:10.1038/s41597-024-04313-w; data
# doi:10.4211/hs.60a890eb841c460192c03bb590687145.
MIRCA_VERSION = "v0.1"
MIRCA_ARCHIVE = "Annual_Harvested_Area_Grids.rar"
MIRCA_URL = ("https://www.hydroshare.org/resource/60a890eb841c460192c03bb590687145/data/"
             "contents/Annual%20Harvested%20Area%20Grids/" + MIRCA_ARCHIVE)
MIRCA_MEMBER_DIR = "Annual Harvested Area Grids/2015/5-arcminute/"
MIRCA_LAYERS: dict[str, str] = {"irrigated": "MIRCA-OS_Maize_2015_ir.tif",
                                "rainfed": "MIRCA-OS_Maize_2015_rf.tif"}
MIRCA_MANIFEST = f"snapshots/mirca-os-{MIRCA_VERSION}.json"
# Section 10.3's check: Eurostat's irrigated and total grain maize in 2023, holdings above
# the survey threshold (the only level published for all four countries).
IRRIGATION_COUNTRIES: tuple[str, ...] = ("ES", "PT", "IT", "EL")
IRRIGATION_FILTERS: dict[str, dict[str, str]] = {
    "ef_lus_irri": {"freq": "A", "statinfo": "ABVTH", "crops": "C1500T", "so_eur": "TOTAL",
                    "uaarea": "TOTAL", "unit": "HA"},
    "ef_lus_allcrops": {"freq": "A", "statinfo": "ABVTH", "crops": "C1500", "so_eur": "TOTAL",
                        "uaarea": "TOTAL", "unit": "HA"},
}
IRRIGATION_SNAPSHOT = "snapshots/eurostat-maize-irrigation-2026-10-29"


def irrigation_url(dataset: str) -> str:
    query = "&".join([*(f"{k}={v}" for k, v in IRRIGATION_FILTERS[dataset].items()),
                      *(f"geo={g}" for g in IRRIGATION_COUNTRIES), "time=2023"])
    return ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
            f"{dataset}?format=JSON&lang=EN&{query}")


def gisco_file(version: int) -> str:
    return f"NUTS_RG_03M_{version}_4326_LEVL_2.geojson"


class DataError(RuntimeError):
    """A source does not have the structure or content the study relies on."""
