"""Canonical package façade for private Google Sheets ingestion."""

from blood_pressure_missingness._compat import reexport_legacy

_legacy = reexport_legacy("refresh_from_google_sheets", globals())

if __name__ == "__main__":
    _legacy.main()
