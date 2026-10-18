# OECD inputs of the shift-share feasibility study, frozen 2026-10-18

Inputs of [SHIFT_SHARE_PROTOCOL.md](../../SHIFT_SHARE_PROTOCOL.md). Do not modify these files.
The audit that precedes the computation
(`eu_okun.shift_share.audit_shift_share`) does two things:
- it checks the raw responses against their SHA-256 hashes;
- it re-derives every processed CSV from them.

| | |
| --- | --- |
| Downloaded (UTC) | 2026-10-18T10:10:19Z |
| Downloaded by | `eu-okun fetch-shift-share --output-dir snapshots/shift-share-2026-10-18` |
| Query URLs | recorded in `manifest.json` |

| File | Source | SHA-256 |
| --- | --- | --- |
| `raw/tiva_fdva.csv` | OECD TiVA 2025 edition, origin of value added in final demand (`FDVA`), all activities, 2005-2007 | `ceab424fa3906f29b872356a59fc3439252bf0ea2774d7f209a12db410e78ca3` |
| `raw/tiva_mainlv.csv` | OECD TiVA 2025 edition, principal indicators: value added (total and B-E) and gross exports, 2005-2007 | `a63e98bd5678444e01f67649761e4b33869246081e7518742385a2d7e56e6517` |
| `raw/oecd_qna.csv` | OECD quarterly national accounts: GDP, chain-linked volume, % change on the previous quarter, seasonally adjusted, 2010-Q1 to 2026-Q2 | `afa0e96feddc692b27b6e90b5fedd7cbc83424a0643030ebcd85e9d8d9c0f4a1` |
| `manifest.json` | | `c5c914bb910b1e29c5e6d8e5239310d4957dd9e9e0e9b665856cad75ec18d3de` |

**Processed files.**

- `processed/exposure.csv` holds the shares of each EU country's value added absorbed in
  each area's final demand. They are given per year and as the 2005-2007 mean.
- `processed/balance.csv` holds the industry and gross-export shares of value added.
- `processed/shifts.csv` holds partner GDP growth. China starts in 2011-Q1; every other
  partner starts in 2010-Q1.

Source: OECD (2025), Trade in Value Added (TiVA) 2025 edition; OECD (2026), Quarterly
National Accounts. OECD data are available under the Creative Commons Attribution 4.0
licence (CC BY 4.0); see the OECD terms and conditions.
