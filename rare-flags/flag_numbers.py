"""Every number in the article, printed in order, plus the tables in results/ that figures.py draws.

python fetch_data.py             # once
python flag_numbers.py > output.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

import flags_core as fc

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
EU_CARD_FRAUD = 0.00015  # share of card payments by number that were fraudulent, EEA, H1 2023 (EBA and ECB 2024)
ESSAY_BASES = (0.03, 0.11)  # Turnitin, April 2024: 3% of papers at least 80% AI-written, 11% at least 20%


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def per_thousand(n, base, sens, fpr):
    f = fc.frequencies(n, base, sens, fpr)
    flags = f["true_flags"] + f["false_flags"]
    return (
        f"out of {n:,}: {f['positive']:.0f} positive, {f['true_flags']:.1f} of them flagged; "
        f"{f['negative']:.0f} negative, {f['false_flags']:.1f} of them flagged; "
        f"{flags:.1f} flags, {pct(f['true_flags'] / flags)} right"
    )


def cases_section():
    print("== Worked cases")
    rows = []

    print("mammography screening (Gigerenzer et al. 2007): base rate 1%, sensitivity 90%, false-positive rate 9%")
    print("  " + per_thousand(1000, 0.01, 0.90, 0.09))
    rows.append(("Mammography screening", 0.01, 0.90, 0.09))

    print("Eddy (1982): base rate 1%, sensitivity 80%, false-positive rate 9.6%")
    print(f"  share of positives that are right: {pct(float(fc.ppv(0.01, 0.80, 0.096)))}")

    alerts, true, crowd = 2470, 173, 170_000
    print("face matching, Champions League final, Cardiff 2017 (South Wales Police figures)")
    print(
        f"  {alerts:,} alerts, {true} right, {alerts - true:,} wrong: {pct(true / alerts)} right; "
        f"the wrong alerts are {pct((alerts - true) / crowd, 2)} of a crowd of {crowd:,}"
    )

    print("OpenAI's text classifier (January 2023): flags 26% of AI-written text and 9% of human text")
    for b in ESSAY_BASES:
        print(f"  if {pct(b, 0)} of essays are AI-written: " + per_thousand(1000, b, 0.26, 0.09))
        rows.append((f"OpenAI classifier, {pct(b, 0)} AI-written", b, 0.26, 0.09))

    print("a detector with a 1% false-positive rate (Turnitin's stated bound, full documents), 3% AI-written:")
    for f in (0.01, 0.005):
        v = [float(fc.ppv(0.03, s, f)) for s in (0.5, 0.75, 1.0)]
        print(
            f"  false-positive rate {pct(f)}: right {pct(v[0])}, {pct(v[1])}, {pct(v[2])} "
            "for sensitivity 50%, 75%, 100%"
        )
    rows.append(("Detector with 1% false alarms, 3% AI-written", 0.03, 0.75, 0.01))

    print(
        f"card payments: {pct(EU_CARD_FRAUD, 3)} fraudulent by number (EEA, first half of 2023); detector catching 90%"
    )
    for f in (0.01, 0.001, 0.0001):
        print(
            f"  false-positive rate {pct(f, 2)}: {pct(float(fc.ppv(EU_CARD_FRAUD, 0.9, f)))} of alerts right, "
            f"{float(fc.false_per_true(EU_CARD_FRAUD, 0.9, f)):.1f} false alerts per fraud"
        )
        rows.append((f"Card alert, {pct(f, 2)} false alarms", EU_CARD_FRAUD, 0.9, f))
    return pd.DataFrame(rows, columns=["case", "base", "sens", "fpr"])


def detectors_section():
    print("\n== AI-text detectors on real essays (Liang et al. 2023 data)")
    c = fc.detector_counts(DATA / "detectors")
    c["rate"] = c.flagged / c.essays
    lo_hi = [fc.jeffreys(k, n) for k, n in zip(c.flagged, c.essays, strict=True)]
    c["lo"], c["hi"] = [x[0] for x in lo_hi], [x[1] for x in lo_hi]
    c.to_csv(RESULTS / "detectors.csv", index=False)
    w = c.pivot(index="detector", columns="group", values="rate")
    n = c.groupby("group").essays.first()
    print(
        f"human essays: {n['native']} by US eighth-graders, {n['non-native']} TOEFL essays by non-native writers, "
        f"{n['college']} college admission essays; AI essays: {n['ai']} plain, {n['ai_edited']} after a "
        "self-edit prompt"
    )
    print("share flagged as AI (score above 0.5), by detector:")
    for det, r in w.iterrows():
        print(
            f"  {det:13s} native {pct(r['native'], 0):>4}, non-native {pct(r['non-native'], 0):>4}, college "
            f"{pct(r['college'], 0):>3} | AI plain {pct(r['ai'], 0):>4}, AI edited {pct(r['ai_edited'], 0):>4}"
        )
    m = w.mean()
    print(
        f"average of the seven: native {pct(m['native'])}, non-native {pct(m['non-native'])}, college "
        f"{pct(m['college'])}; AI plain {pct(m['ai'])}, AI edited {pct(m['ai_edited'])}"
    )
    print(f"non-native false alarms against native, ratio of averages: {m['non-native'] / m['native']:.0f}")
    zero = c[(c.group == "native") & (c.flagged == 0)]
    for r in zero.itertuples():
        print(f"  {r.detector}: 0 of {r.essays} native essays flagged; 95% upper bound {pct(r.hi)}")

    rows = []
    k = c.set_index(["group", "detector"])
    for b in ESSAY_BASES:
        print(f"share of flags that are right if {pct(b, 0)} of essays are AI-written (95% interval):")
        for det in fc.DETECTORS:
            ks, ns = k.loc[("ai", det), "flagged"], k.loc[("ai", det), "essays"]
            parts = []
            for g in ("native", "non-native"):
                kf, nf = k.loc[(g, det), "flagged"], k.loc[(g, det), "essays"]
                point = float(fc.ppv(b, ks / ns, kf / nf))
                lo, hi = fc.ppv_interval(b, ks, ns, kf, nf, seed=7)
                parts.append(f"{g} {pct(point, 0)} [{pct(lo, 0)}, {pct(hi, 0)}]")
                rows.append({"base": b, "detector": det, "group": g, "ppv": point, "lo": lo, "hi": hi})
            print(f"  {det:13s} " + "; ".join(parts))
    pd.DataFrame(rows).to_csv(RESULTS / "detector_ppv.csv", index=False)

    print("the average detector, per 1,000 essays with 3% AI-written:")
    for g in ("native", "non-native"):
        print(f"  {g}: " + per_thousand(1000, 0.03, m["ai"], m[g]))
    f = pd.DataFrame([{"group": g, **fc.frequencies(1000, 0.03, m["ai"], m[g])} for g in ("native", "non-native")])
    f.to_csv(RESULTS / "thousand_essays.csv", index=False)
    return m


def fraud_section():
    print("\n== A real card-fraud model (ULB and Worldline data, September 2013)")
    d = fc.load_cards(DATA / "cards" / "data-raw" / "creditcard.csv.zip")
    print(f"{len(d):,} transactions, {int(d.Class.sum())} frauds ({pct(d.Class.mean(), 3)}), over two days")
    for day, g in d.groupby("day"):
        print(f"  day {day}: {len(g):,} transactions, {int(g.Class.sum())} frauds ({pct(g.Class.mean(), 3)})")
    y, p, auc, ap = fc.fraud_model(d)
    base = float(y.mean())
    print(f"logistic regression trained on day 1, scored on day 2: AUC {auc:.3f}, average precision {ap:.3f}")
    print(
        f"at the data's fraud rate ({pct(base, 3)}) and at the EEA rate for card payments "
        f"({pct(EU_CARD_FRAUD, 3)}), same detector:"
    )
    rows = []
    for r in (0.5, 0.6, 0.7, 0.8, 0.9):
        a = fc.at_recall(y, p, r)
        prec = a["true"] / a["alerts"]
        prec_eu, fpt_eu = fc.rebase(a["true"], a["false"], base, EU_CARD_FRAUD)
        print(
            f"  catching {pct(r, 0)}: {a['alerts']:,} alerts, {a['true']} frauds, {a['false']:,} false "
            f"(false-positive rate {pct(a['fpr'], 3)}); right {pct(prec)}, "
            f"{a['false'] / a['true']:.1f} false per fraud | at the EEA rate: right {pct(prec_eu)}, "
            f"{fpt_eu:.0f} false per fraud"
        )
        rows.append({"recall": r, **a, "precision": prec, "precision_eu": prec_eu, "false_per_true_eu": fpt_eu})
    pd.DataFrame(rows).to_csv(RESULTS / "fraud_recall.csv", index=False)
    r1, p1 = fc.precision_recall_curve(y, p)
    r2, p2 = fc.precision_recall_curve(y, p, EU_CARD_FRAUD)
    pd.DataFrame({"recall": r1, "precision": p1, "precision_eu": p2}).to_csv(RESULTS / "fraud_curve.csv", index=False)
    assert np.allclose(r1, r2)
    return base


def needs_section(m):
    print("\n== What a flag needs")
    for b in (0.11, 0.03, 0.01, EU_CARD_FRAUD):
        req = [float(fc.required_fpr(b, 0.9, t)) for t in (0.5, 0.9, 0.99)]
        print(
            f"base rate {pct(b, 3)}, sensitivity 90%: false-positive rate at most {pct(req[0], 3)} for half the flags "
            f"to be right, {pct(req[1], 3)} for 90%, {pct(req[2], 4)} for 99%"
        )
    for b in (0.03, EU_CARD_FRAUD):
        print(
            f"sensitivity / false-positive rate needed for 90% right flags at {pct(b, 3)}: "
            f"{fc.likelihood_ratio_needed(b, 0.9):,.0f}"
        )
    print(
        f"the average detector: sensitivity / false-positive rate {m['ai'] / m['native']:.1f} on native essays, "
        f"{m['ai'] / m['non-native']:.2f} on non-native essays; OpenAI's classifier {0.26 / 0.09:.1f}; "
        f"Cardiff, if every one of 170,000 faces was checked once and all were innocent but 173: "
        f"at most {1 / (2297 / 170_000):.0f}"
    )


def main():
    RESULTS.mkdir(exist_ok=True)
    cases = cases_section()
    m = detectors_section()
    fraud_section()
    needs_section(m)
    cases.to_csv(RESULTS / "cases.csv", index=False)


if __name__ == "__main__":
    main()
