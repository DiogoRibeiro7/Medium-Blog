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


def gisco_file(version: int) -> str:
    return f"NUTS_RG_03M_{version}_4326_LEVL_2.geojson"


class DataError(RuntimeError):
    """A source does not have the structure or content the study relies on."""
