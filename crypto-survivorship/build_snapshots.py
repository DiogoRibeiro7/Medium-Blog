"""Turn saved CoinMarketCap historical snapshot pages into one normalised CSV per date.

    python build_snapshots.py <folder>      # reads YYYYMMDD.html or YYYYMMDD.csv, writes data/snapshots/YYYY-MM-DD.csv

The pages are saved by hand from https://coinmarketcap.com/historical/YYYYMMDD/ after loading 200 rows.
Two forms are read:
- a saved page (.html). Rows come from the page's embedded JSON when it has them, else from the table, with the
  coin identified by its /currencies/<slug>/ link;
- a table copied into a spreadsheet and saved as .csv, with the columns as shown on the page. The coin is then
  identified by name and symbol.
The output has the columns key, name, symbol, rank, market_cap, price, supply, sorted by rank, and the run
prints a SHA-256 for every input file so the inputs can be recorded without being redistributed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import crypto_core as C

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "snapshots"


class Table(HTMLParser):
    """Rows of every <tr>: (cell texts, first /currencies/<slug>/ link in the row)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[list[str], str | None]] = []
        self._cells: list[str] | None = None
        self._text: list[str] | None = None
        self._slug: str | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._cells, self._slug = [], None
        elif tag in ("td", "th") and self._cells is not None:
            self._text = []
        elif tag == "a" and self._cells is not None and self._slug is None:
            m = re.search(r"/currencies/([^/?#]+)/", dict(attrs).get("href") or "")
            if m:
                self._slug = m.group(1)

    def handle_data(self, data):
        if self._text is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._text is not None and self._cells is not None:
            self._cells.append(" ".join("".join(self._text).split()))
            self._text = None
        elif tag == "tr" and self._cells is not None:
            if self._cells:
                self.rows.append((self._cells, self._slug))
            self._cells = None


def from_json(text: str) -> list[C.Coin]:
    """Rows from embedded JSON: any list of objects with a CoinMarketCap rank, slug and USD quote."""
    coins: list[C.Coin] = []
    for m in re.finditer(r"<script[^>]*>(.*?)</script>", text, flags=re.S):
        body = m.group(1).strip()
        if "cmcRank" not in body and "cmc_rank" not in body:
            continue
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            continue
        stack = [data]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                stack.extend(x.values())
            elif isinstance(x, list):
                if x and all(isinstance(e, dict) and ("cmcRank" in e or "cmc_rank" in e) for e in x):
                    for e in x:
                        q = e.get("quote", {}).get("USD") or next(iter(e.get("quotes", [])), {})
                        coins.append(
                            C.Coin(
                                key=e.get("slug") or f"{e.get('name')}|{e.get('symbol')}",
                                name=e.get("name", ""),
                                symbol=e.get("symbol", ""),
                                rank=int(e.get("cmcRank") or e.get("cmc_rank")),
                                market_cap=float(q.get("marketCap") or q.get("market_cap") or "nan"),
                                price=float(q.get("price") or "nan"),
                                supply=float(e.get("circulatingSupply") or e.get("circulating_supply") or "nan"),
                            )
                        )
                    return coins
                stack.extend(x)
    return coins


def from_cells(header: list[str], rows: list[tuple[list[str], str | None]]) -> list[C.Coin]:
    """Rows from a table whose header names Rank (or #), Name, Symbol, Market Cap, Price and Circulating Supply."""
    h = [c.lower() for c in header]

    def col(*names):
        for n in names:
            for i, c in enumerate(h):
                if c.startswith(n):
                    return i
        raise KeyError(names)

    i_rank, i_name = col("rank", "#"), col("name")
    i_sym, i_cap = col("symbol"), col("market cap")
    i_price, i_supply = col("price"), col("circulating supply")
    coins = []
    for cells, slug in rows:
        if len(cells) <= max(i_rank, i_name, i_sym, i_cap, i_price, i_supply) or not cells[i_rank].strip().isdigit():
            continue
        name, sym = cells[i_name].strip(), cells[i_sym].strip()
        coins.append(
            C.Coin(
                key=slug or f"{name}|{sym}",
                name=name,
                symbol=sym,
                rank=int(cells[i_rank]),
                market_cap=C.number(cells[i_cap]),
                price=C.number(cells[i_price]),
                supply=C.number(cells[i_supply]),
            )
        )
    return coins


def read_page(path: Path) -> list[C.Coin]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".csv":
        rows = list(csv.reader(text.splitlines()))
        return from_cells(rows[0], [(r, None) for r in rows[1:]])
    coins = from_json(text)
    if coins:
        return coins
    t = Table()
    t.feed(text)
    header = next((cells for cells, _ in t.rows if any(c.lower().startswith("market cap") for c in cells)), None)
    if header is None:
        raise ValueError(f"{path.name}: no table with a Market Cap column and no embedded ranking")
    return from_cells(header, t.rows)


def main(folder: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bad = 0
    for path in sorted(Path(folder).iterdir()):
        m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})\.(html?|csv)", path.name)
        if not m:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        coins = [c for c in read_page(path) if c.market_cap > 0 and c.price > 0]
        date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        if len(coins) < 100:
            print(f"SHORT   {path.name}: only {len(coins)} usable rows", file=sys.stderr)
            bad += 1
        with (OUT / f"{date}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["key", "name", "symbol", "rank", "market_cap", "price", "supply"])
            for c in sorted(coins, key=lambda c: c.rank):
                w.writerow([c.key, c.name, c.symbol, c.rank, repr(c.market_cap), repr(c.price), repr(c.supply)])
        print(f"{date}  {len(coins):>4} coins  {path.stat().st_size:>9,} bytes  sha256 {digest}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
