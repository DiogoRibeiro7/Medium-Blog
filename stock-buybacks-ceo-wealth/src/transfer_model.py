"""Arithmetic of a buyback: EPS accretion and who gains when price differs from value.

Everything here is algebra on stated assumptions, not data. Usage:
    python3 -I transfer_model.py OUT.json
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from series import BUYBACKS

out = {}

# 1) EPS arithmetic. N shares, price P, earnings E, buy back a fraction f of shares with cash that earns r (after tax).
def eps_after(E, N, P, f, r):
    cost = f * N * P
    return (E - r * cost) / (N * (1 - f))

N, P, f = 1000.0, 50.0, 0.05
cases = {}
for label, E in {"pe_20": 2500.0, "pe_40": 1250.0}.items():
    eps0 = E / N
    for r in (0.03,):
        eps1 = eps_after(E, N, P, f, r)
        cases[label] = {"earnings_yield": round(E / (N * P), 4), "cash_yield_after_tax": r,
                        "eps_before": round(eps0, 4), "eps_after": round(eps1, 4),
                        "eps_change_pct": round(100 * (eps1 / eps0 - 1), 2)}
out["eps_arithmetic"] = {"shares": N, "price": P, "buyback_fraction": f, "cases": cases,
                         "break_even_rule": "EPS rises if and only if the after-tax earnings yield E/(N*P) exceeds the after-tax yield on the cash spent"}

# 2) Value per continuing share after a buyback at price P when intrinsic value is V.
def v_after(V, P, f):
    return V + f * (V - P) / (1 - f)

V = 50.0
rows = []
for P_ in (40.0, 45.0, 50.0, 55.0, 60.0, 75.0):
    va = v_after(V, P_, 0.05)
    rows.append({"price": P_, "price_over_value": round(P_ / V, 3), "value_per_continuing_share": round(va, 3),
                 "continuing_holder_change_pct": round(100 * (va / V - 1), 3),
                 "seller_gain_per_share_vs_value": round(P_ - V, 2)})
out["value_after_buyback_f5pct"] = rows

# 3) Total transfer from continuing holders to sellers, for 2024 S&P 500 buyback volume B, price premium m = (P-V)/V.
B = BUYBACKS[2024]
sens = []
for m in (-0.10, -0.05, 0.0, 0.02, 0.05, 0.10, 0.20):
    sens.append({"price_premium_over_value": m, "transfer_to_sellers_usd_bn": round(B * m / (1 + m), 1)})
out["transfer_2024_volume"] = {"buybacks_usd_bn": B, "scenarios": sens,
                              "note": "positive = wealth moves from continuing holders to selling holders; negative = the reverse"}

# 4) What share of that transfer a CEO could capture if CEOs were a share theta of the shares sold
theta = (0.001, 0.005, 0.01)
cap = []
for m in (0.05, 0.10):
    for t in theta:
        cap.append({"price_premium": m, "ceo_share_of_shares_sold": t, "ceo_gain_usd_bn": round(B * m / (1 + m) * t, 3)})
out["ceo_capture_scenarios"] = cap

# 5) Scale reference points (different samples, order of magnitude only)
out["scale_references"] = {
    "insider_gains_in_sec_staff_sample_usd_m": 75.1, "buyback_announcements_in_that_sample": 385,
    "ceo_realized_pay_top350_2024_usd_bn": round(22.98e6 * 350 / 1e9, 2),
    "sp500_buybacks_2024_usd_bn": B, "sp500_buybacks_plus_dividends_2024_usd_bn": 1572.0,
}
json.dump(out, open(sys.argv[1], "w"), indent=2)
print(json.dumps(out, indent=2))
