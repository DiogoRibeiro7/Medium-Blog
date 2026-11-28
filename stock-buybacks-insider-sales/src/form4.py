"""Parse Form 4 XML into open-market transaction rows."""
import xml.etree.ElementTree as ET


def raw_xml_name(primary_doc):
    """Submissions lists the XSL-rendered path (xslF345X05/x.xml); the raw XML is the basename."""
    return primary_doc.rsplit("/", 1)[-1]


def _t(node, path):
    el = node.find(path)
    return el.text.strip() if el is not None and el.text else None


def parse_form4(xml_bytes):
    """Return a list of dicts, one per non-derivative transaction."""
    root = ET.fromstring(xml_bytes)
    issuer_cik = _t(root, "issuer/issuerCik")
    rel = root.find("reportingOwner/reportingOwnerRelationship")
    is_officer = (_t(rel, "isOfficer") in ("1", "true")) if rel is not None else False
    is_director = (_t(rel, "isDirector") in ("1", "true")) if rel is not None else False
    title = _t(rel, "officerTitle") if rel is not None else None
    owner_cik = _t(root, "reportingOwner/reportingOwnerId/rptOwnerCik")
    plan_10b5 = _t(root, "aff10b5One") in ("1", "true")
    out = []
    for tx in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        shares = _t(tx, "transactionAmounts/transactionShares/value")
        price = _t(tx, "transactionAmounts/transactionPricePerShare/value")
        out.append({
            "issuer_cik": issuer_cik,
            "owner_cik": owner_cik,
            "is_officer": is_officer,
            "is_director": is_director,
            "title": title,
            "ceo": bool(title) and any(k in title.lower() for k in ("chief executive", "ceo", "principal executive")),
            "tx_date": _t(tx, "transactionDate/value"),
            "code": _t(tx, "transactionCoding/transactionCode"),
            "ad": _t(tx, "transactionAmounts/transactionAcquiredDisposedCode/value"),
            "shares": float(shares) if shares else 0.0,
            "price": float(price) if price else None,
            "plan_10b5": plan_10b5,
        })
    return out


def open_market_sales(rows):
    """Open-market sales: transaction code S, disposed."""
    return [r for r in rows if r["code"] == "S" and r["ad"] == "D" and r["shares"] > 0]
