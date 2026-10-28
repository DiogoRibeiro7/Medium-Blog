"""Reading OMIE's files: numbers, periods and the curve format (METHODS.md, section 2)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from iberian_lp.omie import cents, period_number, read_capacity, read_curve, read_prices, tenths

DAY = date(2026, 10, 15)


def test_numbers_are_read_exactly() -> None:
    assert tenths("2.150,8") == 21508
    assert tenths("0,1") == 1
    assert tenths("-4.373,0") == -43730
    assert tenths("0") == 0
    assert cents("1.500,00") == 150000
    assert cents("-500,00") == -50000
    assert cents("219.86", ".") == 21986
    assert cents("226", ".") == 22600
    assert cents("-0.01", ".") == -1
    with pytest.raises(ValueError):
        tenths("1,25")
    with pytest.raises(ValueError):
        cents("1,005")


def test_periods() -> None:
    assert period_number("H1Q1") == (1, 1, 1)
    assert period_number("H3Q2") == (10, 3, 2)
    assert period_number("H24Q4") == (96, 24, 4)
    with pytest.raises(ValueError):
        period_number("1")


def test_files_are_parsed(tmp_path: Path) -> None:
    curve = tmp_path / "curva_pbc_20261015.1"
    curve.write_text(
        "OMIE - Mercado de electricidad;Fecha Emisión :14/10/2026 - 13:32;;15/10/2026;"
        "Mercado diario;;;;\n\n"
        "Periodo;Fecha;Pais;Unidad;Tipo Oferta;Potencia Compra/Venta;Precio Compra/Venta;"
        "Ofertada (O)/Casada (C);Tipología de Oferta;\n"
        "H1Q1;15/10/2026;MI;;C;557,0;1.500,00;O;S;\n"
        "H1Q2;15/10/2026;MI;;V;1.382,5;-0,01;C;Imp FR;\n"
        ";;;;;;;;;\n", encoding="latin-1")
    steps = list(read_curve(curve, DAY))
    assert [(s.period, s.zone, s.side, s.tenths, s.cents, s.status, s.kind) for s in steps] \
        == [(1, "MI", "C", 5570, 150000, "O", "S"), (2, "MI", "V", 13825, -1, "C", "Imp FR")]
    with pytest.raises(ValueError, match="dated"):
        list(read_curve(curve, date(2026, 10, 16)))
    prices = tmp_path / "marginalpdbc_20261015.1"
    prices.write_text("MARGINALPDBC;\n2026;10;15;1;226;226;\n2026;10;15;2;219.86;210.5;\n*\n",
                      encoding="latin-1")
    assert read_prices(prices, DAY) == {1: (22600, 22600), 2: (21986, 21050)}
    capacity = tmp_path / "capacidad_inter_pbc_20261015.1"
    capacity.write_text(
        "OMIE;;\n\nPeriodo;Fecha;Frontera;...\n"
        "H1Q1;15/10/2026;2;-4.373,0;0;-6.881,2;6.435,0;2.508,2;3.926,8;\n"
        "H1Q1;15/10/2026;3;-3.000,0;-1.200,5;-1.799,5;3.000,0;0;3.000,0;\n", encoding="latin-1")
    assert read_capacity(capacity, "3") == {1: {
        "import_capacity": -30000, "import_occupied": -12005, "import_free": -17995,
        "export_capacity": 30000, "export_occupied": 0, "export_free": 30000}}
