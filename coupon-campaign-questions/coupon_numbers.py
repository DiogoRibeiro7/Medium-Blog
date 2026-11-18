"""Every number in the article, section by section, from the Complete Journey receipts.

    python fetch_data.py
    python coupon_numbers.py > output.txt
    python figures.py

Writes the tables the figures are drawn from to results/. All randomness comes from one seeded
household bootstrap (and one seeded split of coupons in section 8), so a rerun with the pinned
versions reproduces output.txt exactly.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import coupon_core as cc

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 2000
SEED = 20261118
SPLIT_SEED = 7
MARGINS = np.round(np.arange(0.10, 0.61, 0.05), 2)
Z = norm.ppf(0.975) + norm.ppf(0.80)          # two-sided 5% test with 80% power


def head(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def ci(draws, level=0.95):
    a = (1 - level) / 2
    return float(np.nanquantile(draws, a)), float(np.nanquantile(draws, 1 - a))


def fmt(x, digits=4, scale=1.0):
    """Estimate (row 0) with its 95% bootstrap interval (rows 1:)."""
    lo, hi = ci(x[1:])
    return f"{x[0] * scale:+.{digits}f} [{lo * scale:+.{digits}f}, {hi * scale:+.{digits}f}]"


def pval(x):
    se = np.nanstd(x[1:], ddof=1)
    return float(2 * (1 - norm.cdf(abs(x[0]) / se))) if se > 0 else np.nan


class Pool:
    """Accumulates a pooled effect: sum of (effect x recipients x weeks) over campaigns, per bootstrap draw."""

    def __init__(self, draws):
        self.num = np.zeros(draws)
        self.den = np.zeros(draws)

    def add(self, att, n_t, weeks):
        if weeks <= 0 or np.isnan(att[0]):
            return
        self.num += att * n_t * weeks
        self.den += n_t * weeks

    @property
    def effect(self):
        return self.num / self.den

    @property
    def total(self):
        return self.num


def main():
    RESULTS.mkdir(exist_ok=True)
    data = cc.load()
    tx, desc = data.tx, data.desc
    typ = desc.set_index("campaign_id").campaign_type
    hpos = pd.Series(np.arange(len(data.households)), index=data.households)
    W = cc.bootstrap_weights(len(data.households), DRAWS, SEED)
    B = W.shape[0]

    S8 = cc.eligible(desc, 8)
    S16 = set(cc.eligible(desc, 16))
    S24 = set(cc.eligible(desc, 24))
    history = {c: (24 if c in S24 else 16 if c in S16 else 8) for c in S8}
    panels = {c: cc.build_panel(data, c, pre_weeks=history[c]) for c in S8}
    known = [c for c in S8 if typ[c] != "Type A"]
    targeted = [c for c in S8 if typ[c] == "Type A"]
    programme = {"known": dict(ids=known, outcome="cov_trips", cells=cc.cells_known_coupons,
                               label="Types B and C, baskets with a covered product per recipient-week"),
                 "targeted": dict(ids=targeted, outcome="all_spend", cells=cc.cells_targeted,
                                  label="Type A, total spend in dollars per recipient-week")}

    # ------------------------------------------------------------------------------------------
    head("1. The data")
    print(f"receipt lines {len(tx):,}; baskets {tx.basket_id.nunique():,}; households {tx.household_id.nunique():,}; "
          f"products {tx.product_id.nunique():,}; stores {tx.store_id.nunique():,}")
    print(f"first day {tx.date.min().date()}, last receipt day {tx.date.max().date()} (whole days to {cc.LAST_DAY.date()})")
    shelf = tx.sales_value + tx.retail_disc + tx.coupon_match_disc
    print(f"sales value (what the retailer receives) ${tx.sales_value.sum():,.0f}; loyalty-card discounts "
          f"${tx.retail_disc.sum():,.0f} on {(tx.retail_disc > 0).mean():.1%} of lines; manufacturer coupon "
          f"discounts ${tx.coupon_disc.sum():,.0f} on {(tx.coupon_disc > 0).sum():,} lines "
          f"({(tx.coupon_disc > 0).mean():.2%}); retailer coupon match ${tx.coupon_match_disc.sum():,.0f}")
    print(f"loyalty discounts as a share of shelf value: {tx.retail_disc.sum() / shelf.sum():.1%}")
    n_rec = data.recipients.groupby("campaign_id").household_id.nunique()
    n_red = data.redemptions.groupby("campaign_id").size()
    for t in ["Type A", "Type B", "Type C"]:
        ids = desc.loc[desc.campaign_type == t, "campaign_id"]
        print(f"{t}: {len(ids)} campaigns; recipient-campaign pairs {int(n_rec.reindex(ids).fillna(0).sum()):,}; "
              f"redemptions {int(n_red.reindex(ids).fillna(0).sum()):,}")
    print(f"households sent at least one campaign: {data.recipients.household_id.nunique():,} of {len(data.households):,}")
    print(f"coupon redemptions: {len(data.redemptions):,} by {data.redemptions.household_id.nunique():,} households")
    print("campaigns (id, type, start, end, recipients, coupons, covered products, redemptions, weeks of history used):")
    cov_n = data.coupons.groupby("campaign_id").product_id.nunique()
    upc_n = data.coupons.groupby("campaign_id").coupon_upc.nunique()
    for r in desc.itertuples():
        print(f"  {r.campaign_id:>2} {r.campaign_type} {r.start.date()} {r.end.date()} "
              f"{int(n_rec.get(r.campaign_id, 0)):>5} {int(upc_n.get(r.campaign_id, 0)):>4} "
              f"{int(cov_n.get(r.campaign_id, 0)):>6} {int(n_red.get(r.campaign_id, 0)):>4}  "
              f"{history.get(r.campaign_id, '-')}")
    print(f"campaigns with eight weeks of history and at least one complete campaign week: {len(S8)} "
          f"({len(known)} with known coupons, {len(targeted)} targeted); with 16 weeks: {len(S16)}; with 24 weeks: {len(S24)}")

    # ------------------------------------------------------------------------------------------
    head("2. The thirty-second answers")
    lines = cc.redemption_lines(data)
    matched = lines.drop_duplicates(["household_id", "coupon_upc", "campaign_id", "date"])
    print(f"redemptions matched to a same-day receipt line with a coupon discount on a covered product: "
          f"{len(matched):,} of {len(data.redemptions):,} ({len(matched) / len(data.redemptions):.1%})")
    cov_any = data.redemptions.merge(data.coupons, on=["coupon_upc", "campaign_id"]).merge(
        tx[["household_id", "product_id", "date"]].drop_duplicates(), on=["household_id", "product_id", "date"])
    n_any = cov_any.drop_duplicates(["household_id", "coupon_upc", "campaign_id", "date"]).shape[0]
    print(f"matched to a same-day purchase of a covered product, with or without a coupon discount: {n_any:,} "
          f"({n_any / len(data.redemptions):.1%})")
    ul = lines.drop_duplicates(["basket_id", "product_id"])
    baskets = tx[tx.basket_id.isin(ul.basket_id.unique())]
    all_cpn = tx[tx.coupon_disc > 0]
    print(f"sales value of the matched coupon lines ${ul.sales_value.sum():,.0f} (manufacturer coupon value "
          f"${ul.coupon_disc.sum():,.0f}, retailer match ${ul.coupon_match_disc.sum():,.0f}); sales value of the "
          f"{ul.basket_id.nunique():,} whole baskets that held them ${baskets.sales_value.sum():,.0f}")
    print(f"every manufacturer coupon line in the receipts, campaign or not: {len(all_cpn):,} lines in "
          f"{all_cpn.basket_id.nunique():,} baskets, sales value ${all_cpn.sales_value.sum():,.0f}, whole-basket sales "
          f"${tx[tx.basket_id.isin(all_cpn.basket_id.unique())].sales_value.sum():,.0f}")
    rows = []
    for cid in targeted:
        p = panels[cid]
        d0, d1 = cc.campaign_span(p)
        pre = cc.window_mean(p, "all_spend", -8, -1)
        dur = cc.window_mean(p, "all_spend", d0, d1)
        grp = np.where(p.redemption_rows > 0, "redeemers",
                       np.where(p.treated, "recipients, no redemption", "not sent the campaign"))
        rows.append(pd.DataFrame({"campaign": cid, "group": grp, "pre": pre, "during": dur}))
    naive = pd.concat(rows)
    g = naive.groupby("group")[["pre", "during"]].mean().reindex(
        ["redeemers", "recipients, no redemption", "not sent the campaign"])
    g["units"] = naive.groupby("group").size()
    g.to_csv(RESULTS / "naive_groups.csv")
    print(f"targeted campaigns {', '.join(targeted)} pooled; households active before; mean weekly spend in dollars:")
    for name, r in g.iterrows():
        print(f"  {name:<27} units {int(r.units):>5}  eight weeks before ${r.pre:6.2f}  during ${r.during:6.2f}  "
              f"change ${r.during - r.pre:+6.2f}")
    a, b = g.loc["redeemers"], g.loc["recipients, no redemption"]
    print(f"redeemers spent {a.during / b.during - 1:.0%} more per week during the campaigns than recipients who did not "
          f"redeem (${a.during - b.during:.2f}); before the campaigns the gap was {a.pre / b.pre - 1:.0%} "
          f"(${a.pre - b.pre:.2f}); difference in changes ${(a.during - a.pre) - (b.during - b.pre):+.2f}")

    # ------------------------------------------------------------------------------------------
    head("3. Who was sent the coupons")
    sel = []
    for c in S8:
        p = panels[c]
        sp = cc.window_mean(p, "all_spend", -8, -1)
        ct = cc.window_mean(p, "cov_trips", -8, -1)
        sel.append(dict(campaign=c, type=typ[c], rec=sp[p.treated].mean(), ctl=sp[~p.treated].mean(),
                        rec_cov=(ct[p.treated] > 0).mean(), ctl_cov=(ct[~p.treated] > 0).mean(),
                        n_t=p.treated.sum(), n_c=(~p.treated).sum(), inactive=p.n_recipients_inactive,
                        recipients=p.n_recipients))
    sel = pd.DataFrame(sel)
    for t, s in sel.groupby("type"):
        r_, c_ = np.average(s.rec, weights=s.n_t), np.average(s.ctl, weights=s.n_c)
        print(f"{t}: recipients spent ${r_:.2f} a week in the eight weeks before, other active households ${c_:.2f} "
              f"(ratio {r_ / c_:.2f}); bought a covered product in those weeks: recipients "
              f"{np.average(s.rec_cov, weights=s.n_t):.1%}, others {np.average(s.ctl_cov, weights=s.n_c):.1%}")
    print(f"recipients with no receipt in the eight weeks before their campaign (outside every panel): "
          f"{int(sel.inactive.sum()):,} of {int(sel.recipients.sum()):,} recipient-campaign pairs")
    red = data.redemptions.merge(desc[["campaign_id", "campaign_type", "start"]], on="campaign_id")
    x = red.merge(data.coupons, on=["coupon_upc", "campaign_id"]).merge(
        tx[["household_id", "product_id", "date"]].drop_duplicates(), on=["household_id", "product_id"],
        how="left", suffixes=("", "_tx"))
    prior = {}
    for weeks in [8, 13]:
        x["prior"] = (x.date_tx < x.start) & (x.date_tx >= x.start - pd.Timedelta(weeks=weeks))
        pr = x.groupby(["household_id", "coupon_upc", "campaign_id", "date", "campaign_type"]).prior.any().reset_index()
        by = pr.groupby("campaign_type").prior.mean()
        prior[weeks] = by
        print(f"redemptions where the household had bought a product the coupon covers in the {weeks} weeks before the "
              f"campaign: {pr.prior.mean():.1%} (A {by['Type A']:.1%}, B {by['Type B']:.1%}, C {by['Type C']:.1%}; "
              f"B and C together {pr.loc[pr.campaign_type != 'Type A', 'prior'].mean():.1%})")

    # ------------------------------------------------------------------------------------------
    head("4. The standard pipeline: match on the eight weeks before, then difference in differences")
    per = []
    for key, prog in programme.items():
        pool = Pool(B)
        pool_post = Pool(B)
        for c in prog["ids"]:
            p = panels[c]
            hp = hpos.reindex(p.households).to_numpy()
            cells = prog["cells"](p, -8, -1)
            d0, d1 = cc.campaign_span(p)
            q0, q1 = cc.post_span(p)
            base = cc.window_mean(p, prog["outcome"], -8, -1)
            att, n = cc.did(cc.window_mean(p, prog["outcome"], d0, d1) - base, p.treated, cells, hp, W)
            pool.add(att, n, p.n_dur_obs)
            if p.n_post_obs:
                att_p, n_p = cc.did(cc.window_mean(p, prog["outcome"], q0, q1) - base, p.treated, cells, hp, W)
                pool_post.add(att_p, n_p, p.n_post_obs)
            per.append(dict(campaign=c, type=typ[c], design="matched", estimate=att[0], lo=ci(att[1:])[0],
                            hi=ci(att[1:])[1], p=pval(att), n_t=int(n[0]), weeks=p.n_dur_obs))
        base_level = np.average([cc.window_mean(panels[c], prog["outcome"], -8, -1)[panels[c].treated].mean()
                                 for c in prog["ids"]],
                                weights=[panels[c].treated.sum() * panels[c].n_dur_obs for c in prog["ids"]])
        print(f"{prog['label']}: {len(prog['ids'])} campaigns, {pool.den[0]:,.0f} recipient-weeks; recipients' "
              f"baseline {base_level:.4f}")
        print(f"  effect during the campaigns {fmt(pool.effect)} = {pool.effect[0] / base_level:+.1%} of baseline; "
              f"total {fmt(pool.total, 1)}")
        print(f"  effect in the eight weeks after the campaigns {fmt(pool_post.effect)} ({pool_post.den[0]:,.0f} "
              f"recipient-weeks)")
        programme[key]["baseline"] = base_level
    per = pd.DataFrame(per)
    per["p_bh"] = cc.bh(per.p)
    print(f"campaigns analysed separately: {len(per)}; nominal p < 0.05: {(per.p < 0.05).sum()} "
          f"(all with positive estimates: {bool((per.loc[per.p < 0.05, 'estimate'] > 0).all())}); "
          f"Benjamini-Hochberg adjusted p < 0.05: {(per.p_bh < 0.05).sum()}; expected by chance if nothing worked "
          f"{0.05 * len(per):.1f}")

    # ------------------------------------------------------------------------------------------
    head("5. Asking each design what it reports when this campaign's coupons did not exist yet")
    ledger = []
    event_rows = []
    for key, prog in programme.items():
        o = prog["outcome"]
        ids16 = [c for c in prog["ids"] if c in S16]
        ids24 = [c for c in prog["ids"] if c in S24]
        est = {k: Pool(B) for k in ["matched", "plain", "trend"]}
        post_trend = Pool(B)
        plac = {k: Pool(B) for k in ["matched", "plain", "trend"]}
        plac16 = {k: Pool(B) for k in ["matched", "plain"]}
        short_plain = Pool(B)
        R = {k: np.zeros(B) for k in ["matched", "plain", "trend"]}
        for c in ids16:
            p = panels[c]
            hp = hpos.reindex(p.households).to_numpy()
            zero = np.zeros(len(p.treated), int)
            d0, d1 = cc.campaign_span(p)
            q0, q1 = cc.post_span(p)
            base = cc.window_mean(p, o, -8, -1)
            dur = cc.window_mean(p, o, d0, d1)
            cells8 = prog["cells"](p, -8, -1)
            a_m, n_m = cc.did(dur - base, p.treated, cells8, hp, W)
            a_p, n_p = cc.did(dur - base, p.treated, zero, hp, W)
            tr, nw = cc.trend_residual(p, o, -16, -1, d0, d1)
            a_t, n_t = cc.did(tr, p.treated, zero, hp, W)
            est["matched"].add(a_m, n_m, p.n_dur_obs)
            est["plain"].add(a_p, n_p, p.n_dur_obs)
            est["trend"].add(a_t, n_t, nw)
            trp, nwp = cc.trend_residual(p, o, -16, -1, q0, q1)
            if nwp:
                a_tp, n_tp = cc.did(trp, p.treated, zero, hp, W)
                post_trend.add(a_tp, n_tp, nwp)
            # placebos on the same campaigns, 16 weeks of history
            b16 = cc.window_mean(p, o, -16, -9)
            a, n = cc.did(base - b16, p.treated, prog["cells"](p, -16, -9), hp, W)
            plac16["matched"].add(a, n, 8)
            a, n = cc.did(base - b16, p.treated, zero, hp, W)
            plac16["plain"].add(a, n, 8)
            a, n = cc.did(cc.window_mean(p, o, -4, -1) - cc.window_mean(p, o, -8, -5), p.treated, zero, hp, W)
            short_plain.add(a, n, 4)
            # redemption trips by the recipients each design uses
            keep_m = p.treated & np.isin(cells8, np.unique(cells8[~p.treated]))
            R["matched"] += W[:, hp[keep_m]] @ p.redemption_trips[keep_m]
            R["plain"] += W[:, hp[p.treated]] @ p.redemption_trips[p.treated]
            R["trend"] += W[:, hp[p.treated]] @ p.redemption_trips[p.treated]
            for name, a_, n_ in [("matched", a_m, n_m), ("plain", a_p, n_p), ("trend", a_t, n_t)]:
                per.loc[len(per)] = dict(campaign=c, type=typ[c], design=name + "16", estimate=a_[0],
                                         lo=ci(a_[1:])[0], hi=ci(a_[1:])[1], p=pval(a_), n_t=int(n_[0]),
                                         weeks=p.n_dur_obs, p_bh=np.nan)
        for c in ids24:
            p = panels[c]
            hp = hpos.reindex(p.households).to_numpy()
            zero = np.zeros(len(p.treated), int)
            base = cc.window_mean(p, o, -8, -1)
            b16 = cc.window_mean(p, o, -16, -9)
            a, n = cc.did(base - b16, p.treated, prog["cells"](p, -16, -9), hp, W)
            plac["matched"].add(a, n, 8)
            a, n = cc.did(base - b16, p.treated, zero, hp, W)
            plac["plain"].add(a, n, 8)
            tr, nw = cc.trend_residual(p, o, -24, -9, -8, -1)
            a, n = cc.did(tr, p.treated, zero, hp, W)
            plac["trend"].add(a, n, nw)
        programme[key].update(est=est, R=R, ids16=ids16, ids24=ids24)
        print(f"{prog['label']}")
        print(f"  estimates on the {len(ids16)} campaigns with 16 weeks of history ({', '.join(ids16)}); placebos on the "
              f"{len(ids24)} with 24 weeks ({', '.join(ids24)})")
        names = {"matched": "matched on weeks -8..-1, change from weeks -8..-1",
                 "plain": "all active households, change from weeks -8..-1",
                 "trend": "all active households, each against its own line over weeks -16..-1"}
        pnames = {"matched": "matched on weeks -16..-9, change from -16..-9 to -8..-1",
                  "plain": "change from weeks -16..-9 to -8..-1",
                  "trend": "line over weeks -24..-9, residual in weeks -8..-1"}
        for k in ["matched", "plain", "trend"]:
            print(f"  {k:<8} estimate  {fmt(est[k].effect)}  ({names[k]})")
            print(f"  {'':<8} placebo   {fmt(plac[k].effect)}  ({pnames[k]})")
            for kind, pool in [("estimate", est[k]), ("placebo", plac[k])]:
                lo, hi = ci(pool.effect[1:])
                ledger.append(dict(programme=key, design=k, kind=kind, value=pool.effect[0], lo=lo, hi=hi,
                                   p=pval(pool.effect)))
        print(f"  placebos on the 16-week campaigns instead: matched {fmt(plac16['matched'].effect)}, plain "
              f"{fmt(plac16['plain'].effect)}")
        print(f"  short placebo for the plain design (weeks -4..-1 against -8..-5): {fmt(short_plain.effect)}")
        print(f"  trend design in the eight weeks after the campaigns: {fmt(post_trend.effect)}")
        if key == "targeted":
            for fit_lo in [-12, -16, -20, -24]:
                pool = Pool(B)
                used = []
                for c in prog["ids"]:
                    p = panels[c]
                    if -fit_lo > history[c]:
                        continue
                    hp = hpos.reindex(p.households).to_numpy()
                    d0, d1 = cc.campaign_span(p)
                    tr, nw = cc.trend_residual(p, o, fit_lo, -1, d0, d1)
                    a, n = cc.did(tr, p.treated, np.zeros(len(p.treated), int), hp, W)
                    pool.add(a, n, nw)
                    used.append(c)
                print(f"  trend design fitted on weeks {fit_lo}..-1 (campaigns {', '.join(used)}): {fmt(pool.effect, 2)}")
        # event study on the campaigns with 24 weeks of history
        labs = cc.event_labels(24)
        lines_ = {"matched on weeks -8..-1": (-8, -1, True), "matched on weeks -16..-9": (-16, -9, True),
                  "no matching": (-8, -1, False)}
        for line, (lo_, hi_, use_cells) in lines_.items():
            num = np.zeros((B, len(labs)))
            den = np.zeros((B, len(labs)))
            for c in ids24:
                p = panels[c]
                hp = hpos.reindex(p.households).to_numpy()
                e = cc.event_changes(p, o, lo_, hi_, labs)
                cells = prog["cells"](p, lo_, hi_) if use_cells else np.zeros(len(p.treated), int)
                a, n = cc.did(e, p.treated, cells, hp, W)
                ok = ~np.isnan(a)
                num += np.where(ok, a * n[:, None], 0)
                den += np.where(ok, n[:, None], 0)
            ev = num / np.where(den > 0, den, np.nan)
            for j, lab in enumerate(labs):
                event_rows.append(dict(programme=key, line=line, position=lab, effect=ev[0, j],
                                       lo=np.nanquantile(ev[1:, j], 0.025), hi=np.nanquantile(ev[1:, j], 0.975),
                                       units=den[0, j]))
    ledger = pd.DataFrame(ledger)
    ledger.to_csv(RESULTS / "design_ledger.csv", index=False)
    ev = pd.DataFrame(event_rows)
    ev.to_csv(RESULTS / "event_study.csv", index=False)
    for key in programme:
        print(f"event study, {key} (recipients minus controls, each line from its own baseline window):")
        for line in ev.line.unique():
            s = ev[(ev.programme == key) & (ev.line == line)]
            scale = 100 if key == "known" else 1
            print(f"  {line:<26} " + " ".join(f"{r.position}:{r.effect * scale:+.1f}" for r in s.itertuples()
                                              if not np.isnan(r.effect)))

    # ------------------------------------------------------------------------------------------
    head("6. One question per campaign, three designs")
    per16 = per[per.design.str.endswith("16")].copy()
    for d in ["matched16", "plain16", "trend16"]:
        s = per16[per16.design == d].copy()
        s["p_bh"] = cc.bh(s.p)
        per16.loc[s.index, "p_bh"] = s.p_bh
        sig = s[s.p < 0.05]
        listed = ", ".join(f"{r.campaign}{'+' if r.estimate > 0 else '-'}" for r in sig.itertuples())
        print(f"{d[:-2]:<8}: {len(s)} campaigns; nominal p < 0.05: {len(sig)} ({listed}); "
              f"BH-adjusted p < 0.05: {(s.p_bh < 0.05).sum()}")
    per_all = pd.concat([per[per.design == "matched"], per16])
    per_all.to_csv(RESULTS / "campaigns_by_design.csv", index=False)

    # ------------------------------------------------------------------------------------------
    head("7. How many redemptions would have happened anyway (known-coupon campaigns)")
    kp = programme["known"]
    shares = []
    for k in ["matched", "plain", "trend"]:
        inc = kp["est"][k].total
        Rk = kp["R"][k]
        s = inc / Rk
        lo, hi = ci(s[1:])
        shares.append(dict(design=k, share=s[0], lo=lo, hi=hi, incremental=inc[0], redemption_trips=Rk[0]))
        print(f"{k:<8}: incremental covered baskets {fmt(inc, 1)}; redemption trips {Rk[0]:.0f}; incremental share "
              f"{fmt(s, 2)}; would have happened anyway {1 - s[0]:+.2f} [{1 - hi:+.2f}, {1 - lo:+.2f}]")
    pd.DataFrame(shares).to_csv(RESULTS / "incremental_shares.csv", index=False)
    rw = sum(panels[c].treated.sum() * panels[c].n_dur_obs for c in known)
    rt = sum(panels[c].redemption_trips[panels[c].treated].sum() for c in known)
    var_d = np.average([np.var((cc.window_mean(panels[c], "cov_trips", *cc.campaign_span(panels[c]))
                                - cc.window_mean(panels[c], "cov_trips", -8, -1))[panels[c].treated], ddof=1)
                        for c in known], weights=[panels[c].treated.sum() for c in known])
    r = rt / rw
    print(f"all {len(known)} known-coupon campaigns: {rt:.0f} redemption trips over {rw:,} recipient-weeks = "
          f"{r:.4f} per recipient-week ({r * 100:.2f} per 100)")
    print(f"variance across recipients of the change in weekly covered baskets (campaign weeks minus the eight weeks "
          f"before): {var_d:.4f} (standard deviation {np.sqrt(var_d):.3f})")
    for share in [0.25, 0.5, 1.0]:
        delta = share * r
        n50 = Z ** 2 * var_d / (0.25 * delta ** 2)
        n90 = Z ** 2 * var_d / (0.09 * delta ** 2)
        print(f"  households needed to detect an incremental share of {share:.2f} (effect {delta:.4f} per recipient-week, "
              f"80% power, 5% two-sided): {n50:,.0f} with a 50/50 holdout, {n90:,.0f} with a 10% holdout")
    pd.DataFrame([dict(var_d=var_d, r=r, z=Z)]).to_csv(RESULTS / "power_inputs.csv", index=False)
    # within-household design: random half of each campaign's coupons
    rng = np.random.default_rng(SPLIT_SEED)
    vdiff, vsum, wts = [], [], []
    for c in known:
        cp = data.coupons[data.coupons.campaign_id == c]
        upcs = np.array(sorted(cp.coupon_upc.unique()))
        if len(upcs) < 2:
            continue
        half = set(upcs[rng.permutation(len(upcs))[: len(upcs) // 2]])
        pa = set(cp.loc[cp.coupon_upc.isin(half), "product_id"])
        pb = set(cp.loc[~cp.coupon_upc.isin(half), "product_id"]) - pa
        p = panels[c]
        d0, d1 = cc.campaign_span(p)
        out = []
        for prod in [pa, pb]:
            q = cc.build_panel(data, c, covered=prod, pre_weeks=history[c])
            out.append((cc.window_mean(q, "cov_trips", d0, d1) - cc.window_mean(q, "cov_trips", -8, -1))[q.treated])
        vdiff.append(np.var(out[0] - out[1], ddof=1))
        vsum.append(np.var(out[0] + out[1], ddof=1))
        wts.append(p.treated.sum())
    ratio = np.average(vdiff, weights=wts) / np.average(vsum, weights=wts)
    print(f"randomising which half of a campaign's coupons each recipient gets, instead of whether it gets any: "
          f"variance ratio {ratio:.2f} ({len(wts)} campaigns), i.e. {1 - ratio:.0%} fewer households for the same power")

    # ------------------------------------------------------------------------------------------
    head("8. Whose margin, and at what discount the coupons pay")
    bc = lines[lines.campaign_id.isin(known)].drop_duplicates(["basket_id", "product_id"])
    bc = bc.assign(shelf=bc.sales_value + bc.retail_disc + bc.coupon_match_disc)
    trips = bc.groupby(["household_id", "date"]).agg(coupon=("coupon_disc", "sum"), match=("coupon_match_disc", "sum"),
                                                    shelf=("shelf", "sum"))
    depth = trips.coupon.sum() / trips.shelf.sum()
    print(f"known-coupon redemption trips matched to receipt lines: {len(trips):,}; manufacturer coupon per trip "
          f"${trips.coupon.mean():.2f} (median ${trips.coupon.median():.2f}); retailer match per trip "
          f"${trips.match.mean():.2f}; shelf value of the couponed items ${trips.shelf.mean():.2f}")
    print(f"coupon depth (manufacturer coupon / shelf value of the couponed items): {depth:.1%}; with the retailer's "
          f"match {(trips.coupon.sum() + trips.match.sum()) / trips.shelf.sum():.1%}")
    be = pd.DataFrame({"margin": MARGINS, "break_even_share": depth / MARGINS})
    be.to_csv(RESULTS / "break_even.csv", index=False)
    print("break-even incremental share = depth / contribution margin on shelf value: " +
          "  ".join(f"m={m:.2f}: {s:.2f}" for m, s in zip(be.margin, be.break_even_share)))
    sh = pd.DataFrame(shares).set_index("design")
    for k in ["plain", "trend"]:
        print(f"margin at which the coupons break even if the incremental share is the {k} estimate "
              f"({sh.loc[k, 'share']:.2f}): {depth / sh.loc[k, 'share']:.2f}")
    print(f"prior-buyer share of known-coupon redemptions (bought a covered product in the 8 / 13 weeks before): "
          f"B {prior[8]['Type B']:.1%} / {prior[13]['Type B']:.1%}, C {prior[8]['Type C']:.1%} / {prior[13]['Type C']:.1%}")
    tp = programme["targeted"]
    red_a = sum(panels[c].redemption_rows[panels[c].treated].sum() for c in tp["ids16"])
    for k in ["plain", "trend"]:
        tot = tp["est"][k].total
        print(f"targeted campaigns, {k} design: change in recipients' total spend {fmt(tot, 0)} dollars over "
              f"{tp['est'][k].den[0]:,.0f} recipient-weeks; per redeemed coupon ${tot[0] / red_a:,.2f} "
              f"({red_a:.0f} redemptions)")
    print(f"retailer's own outlay on campaign coupons in the receipts (coupon match on matched lines, all campaigns): "
          f"${ul.coupon_match_disc.sum():,.0f}")

    # ------------------------------------------------------------------------------------------
    head("9. The larger discount")
    print(f"loyalty-card discounts ${tx.retail_disc.sum():,.0f} ({tx.retail_disc.sum() / shelf.sum():.1%} of shelf value, on "
          f"{(tx.retail_disc > 0).mean():.1%} of lines); all manufacturer coupons ${tx.coupon_disc.sum():,.0f}; "
          f"campaign coupon value on matched lines ${ul.coupon_disc.sum():,.0f}; ratio of loyalty discounts to "
          f"all manufacturer coupons {tx.retail_disc.sum() / tx.coupon_disc.sum():.0f} to 1")
    hh = tx.assign(shelf=shelf).groupby("household_id")[["retail_disc", "shelf"]].sum()
    s = hh.retail_disc / hh.shelf
    print(f"loyalty discount as a share of each household's shelf value: median {s.median():.1%}, quartiles "
          f"{s.quantile(0.25):.1%} to {s.quantile(0.75):.1%}")


if __name__ == "__main__":
    main()
