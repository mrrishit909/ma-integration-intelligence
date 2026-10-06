"""Held-out evaluation on data rooms and deal databases the demo never uses (seeds 41, 51, 61, 71, 81). Every model is built the way
the API builds it and scored against the generator's truth, beside the baseline a deal team would otherwise use.

    python -m ma.evaluate > docs/evaluation.md
"""
import time

import numpy as np

from . import engine as E, world as W

SEEDS = (41, 51, 61, 71, 81)
TEST_DEALS = 2000
FATES = 300


def pct(x):
    return f"{100 * x:.0f}%"


def build(seed):
    r = W.data_room(seed)
    rows, _ = W.comparables(seed)
    names = [c["name"] for c in r["customers"]]
    rev = np.array([c["revenue"] for c in r["customers"]], float)
    match = E.match_customers(names, r["acquirer_customers"])
    fin = {}
    for x in r["financials"][-4 * 7:]:
        fin[x["metric"]] = fin.get(x["metric"], 0) + x["value"]
    f = E.deal_features(W.DEAL, fin["revenue"], fin["ebitda"], float(rev[list(match)].sum() / rev.sum()))
    return r, rows, names, rev, match, f, fin["revenue"]


def synergy_scores(rows, seed):
    """Fresh deals from the true process, predicted by the fitted model and by the baselines."""
    mod = E.fit_synergy(rows)
    rng = np.random.default_rng([seed, 77])
    f = W.features(rng, TEST_DEALS)
    lc, lr = W.true_log_r(f, rng.standard_normal((TEST_DEALS, 2)))
    out = {}
    for kind, true in (("cost", np.exp(lc)), ("rev", np.exp(lr))):
        p10, p50, p90 = [], [], []
        for i in range(TEST_DEALS):
            fi = {k: float(v[i]) for k, v in f.items()}
            draws = E.predict(mod, fi, fi["fast"] == 1, 400, np.random.default_rng(i))[0 if kind == "cost" else 1]
            a, b, c = np.quantile(draws, [0.1, 0.5, 0.9])
            p10.append(a), p50.append(b), p90.append(c)
        p10, p50, p90 = map(np.array, (p10, p50, p90))
        err = lambda pred: float(np.median(np.abs(pred - true)))      # noqa: E731
        out[kind] = {"model": err(p50), "management": err(np.ones(TEST_DEALS)), "comps": err(np.full(TEST_DEALS, mod["comps"][kind])),
                     "latest": err(np.full(TEST_DEALS, mod["naive_latest"][kind])), "coverage": float(((true >= p10) & (true <= p90)).mean()),
                     "bias": float(np.median(p50 / true) - 1), "comps_level": mod["comps"][kind], "true_mean": float(true.mean())}
    return mod, out


def main():
    t0 = time.time()
    out = ["# Evaluation", "", f"Five data rooms and deal databases the demo never uses (seeds {', '.join(map(str, SEEDS))}). Each model is fitted the way the API",
           "fits it and scored against what the generator knows. Reproduce with `python -m ma.evaluate`.", ""]
    syn, er, parse, apps, erp_rows, integ, track_rows, phr = [], [], [], [], [], [], [], {}
    for seed in SEEDS:
        r, rows, names, rev, match, f, revenue = build(seed)
        mod, s = synergy_scores(rows, seed)
        syn.append((seed, s))
        # entity resolution
        truth = {i for i, c in enumerate(r["customers"]) if c["true_overlap"]}
        ex = E.match_exact(names, r["acquirer_customers"])
        er.append((seed, len(set(match) & truth) / len(match), len(set(match) & truth) / len(truth), len(set(ex) & truth) / max(1, len(ex)), len(set(ex) & truth) / len(truth),
                   float(rev[list(match)].sum() / rev.sum()), float(rev[list(ex)].sum() / rev.sum()), r["truth"]["customer_overlap_share"]))
        # contract rules
        tp = fp = pos = tfc_tp = tfc_fp = tfc_pos = 0
        for doc, ct in zip([d for d in r["documents"] if d["kind"] == "customer_contract"], r["truth"]["clauses"]):
            h = E.parse_contract(doc["text"])
            pos += ct["coc"] is not None
            tp += bool(h["coc"]) and ct["coc"] is not None
            fp += bool(h["coc"]) and ct["coc"] is None
            tfc_pos += ct["tfc"] is not None
            tfc_tp += bool(h["tfc"]) and ct["tfc"] is not None
            tfc_fp += bool(h["tfc"]) and ct["tfc"] is None
            if ct["coc"] is not None:
                phr.setdefault(ct["coc"], [0, 0])
                phr[ct["coc"]][0] += bool(h["coc"])
                phr[ct["coc"]][1] += 1
        parse.append((seed, tp, fp, pos, tfc_tp, tfc_fp, tfc_pos))
        # application clusters
        a = r["apps"]
        lab = E.cluster_apps(a)
        pairs = {(i, j) for i, x in enumerate(a) for j, y in enumerate(a) if x["company"] == "acquirer" and y["company"] == "target"}
        tru = {p for p in pairs if a[p[0]]["true_capability"] == a[p[1]]["true_capability"]}
        found = {p for p in pairs if lab[p[0]] == lab[p[1]]}
        prod = {p for p in pairs if a[p[0]]["product"] == a[p[1]]["product"]}
        apps.append((seed, len(found & tru) / max(1, len(found)), len(found & tru) / len(tru), len(prod & tru) / max(1, len(prod)), len(prod & tru) / len(tru)))
        # ERP programmes
        e = E.fit_erp(rows)
        erp_rows.append((seed, e))
        # integration choice
        erp_ref = next(x["ref"] for x in a if x["company"] == "target" and x["true_capability"] == "erp")
        ws = E.workstreams([(x["ref"], x["product"], x["users"]) for x in a if any(s_ == erp_ref and d_ == x["ref"] for s_, d_, _ in r["edges"])])
        tf = {**f, "overlap": r["truth"]["customer_overlap_share"]}
        res = {}
        for ap in ("fast", "staged"):
            lanes, _ = E.best_lanes(ws, mod, e, f, ap, W.DEAL, revenue)
            o = E.evaluate_option(ws, mod, e, f, ap, lanes, 10000, 2026, W.DEAL, revenue, truth_f=tf)
            res[ap] = (o["sim"]["value"], o["truth"]["value"], E.management_case(ws, mod, ap, lanes, W.DEAL, revenue)["value"][0])
        prem = W.DEAL["price"] - W.DEAL["standalone_value"]
        integ.append((seed, res, prem))
        # tracking: many hidden fates for this deal, four quarters each
        hits = {"cost": 0, "rev": 0}
        errs = {"cost": [], "rev": [], "cost_line": [], "rev_line": [], "cost_prior": [], "rev_prior": []}
        for i in range(FATES):
            ap = "fast" if i % 2 else "staged"
            reps, hid = W.realisation_reports(100000 * seed + i, ap, 4, tf, W.DEAL["management_case"]["cost_run_rate"], W.DEAL["management_case"]["revenue_run_rate"])
            t = E.track(mod, f, ap, reps, W.DEAL["management_case"], n=4000, seed=i)
            for kind, key in (("cost", "cost_ratio"), ("rev", "rev_ratio")):
                lo, mid, hi = t[kind]["posterior"]
                hits[kind] += lo <= hid[key] <= hi
                errs[kind].append(abs(mid - hid[key]))
                errs[f"{kind}_line"].append(abs(t[kind]["straight_line"] - hid[key]))
                errs[f"{kind}_prior"].append(abs(t[kind]["prior"][1] - hid[key]))
        track_rows.append((seed, {k: v / FATES for k, v in hits.items()}, {k: float(np.median(v)) for k, v in errs.items()}))

    out += ["## Synergy realisation, on 2,000 fresh deals per database", "",
            "Realised run-rate as a share of the announced run-rate three years after close. Error is the median absolute error in that share (0.10 = ten points).", "",
            "| Database | Kind | Model | Management case (1.0) | Bankers' comparables | Latest disclosure as is | Model's 80% interval covers | Model median vs truth |", "|---|---|---|---|---|---|---|---|"]
    for seed, s in syn:
        for kind, label in (("cost", "cost"), ("rev", "revenue")):
            x = s[kind]
            out.append(f"| {seed} | {label} | {x['model']:.3f} | {x['management']:.3f} | {x['comps']:.3f} (level {x['comps_level']:.2f}) | {x['latest']:.3f} | {pct(x['coverage'])} | {100 * x['bias']:+.0f}% |")
    out += ["", "The comparables are the deals at least three years old that still disclose. Deals that fell behind tend to stop disclosing, so the",
            "comparables run high; taking every latest disclosure as final runs low, because a third of the deals are still ramping.", "",
            "## ERP programmes", "", "| Database | Programmes (running) | Lateness, censored fit | Finished only | Truth | Incident rate fast / staged | Truth |", "|---|---|---|---|---|---|---|"]
    for seed, e in erp_rows:
        inc = {a: e["incident"][a]["a"] / (e["incident"][a]["a"] + e["incident"][a]["b"]) for a in ("fast", "staged")}
        out.append(f"| {seed} | {e['programmes']} ({e['running']}) | ×{np.exp(e['bias_mu']):.2f} | ×{np.exp(e['finished_only_mu']):.2f} | ×{np.exp(W.TRUTH['erp']['bias_mu']):.2f} | {pct(inc['fast'])} / {pct(inc['staged'])} | {pct(W.TRUTH['erp']['p_incident']['fast'])} / {pct(W.TRUTH['erp']['p_incident']['staged'])} |")
    out += ["", "Only 8–17% of the programmes are still running in these databases, so dropping them moves the estimate by at most 0.03 here; the",
            "censored fit matters more when a database is young. Lateness is recovered to within 0.05; the fast incident rate, from 40–60 finished",
            "fast programmes, is off by up to twelve points.", "",
            "## Fast against staged ERP, for the demo deal's terms on each data room", "",
            "| Data room | Fast: model / truth / plan as written | Staged: model / truth / plan as written | Fast minus staged: model / truth | Same choice | 80% range, fast (model) | P(covers the $260M premium), fast: model / truth |", "|---|---|---|---|---|---|---|"]
    agree = 0
    for seed, res, prem in integ:
        m = {a: res[a][0].mean() / 1e6 for a in res}
        t = {a: res[a][1].mean() / 1e6 for a in res}
        same = (m["fast"] > m["staged"]) == (t["fast"] > t["staged"])
        agree += same
        out.append(f"| {seed} | ${m['fast']:.0f}M / ${t['fast']:.0f}M / ${res['fast'][2] / 1e6:.0f}M | ${m['staged']:.0f}M / ${t['staged']:.0f}M / ${res['staged'][2] / 1e6:.0f}M | "
                   f"{m['fast'] - m['staged']:+.0f} / {t['fast'] - t['staged']:+.0f} | {'yes' if same else 'no'} | ${(np.quantile(res['fast'][0], 0.9) - np.quantile(res['fast'][0], 0.1)) / 1e6:.0f}M | {pct((res['fast'][0] >= prem).mean())} / {pct((res['fast'][1] >= prem).mean())} |")
    out += ["", f"The model picks the truth's better option on {agree} of {len(integ)} data rooms: no better than a coin. The truth's own gap between the",
            "options is at most $8M against an 80% range of $180–200M, so a wrong pick costs little; the question the model does answer is how",
            "far the plan as written ($77–110M above the truth) sits from either option, and whether the premium is covered.", "",
            f"## Tracking realisation: {FATES} hidden fates per data room, four quarterly reports each", "",
            "| Data room | 80% interval covers, cost / revenue | Median error, cost: tracker / straight line / no reports | Revenue: tracker / straight line / no reports |", "|---|---|---|---|"]
    for seed, cov, err in track_rows:
        out.append(f"| {seed} | {pct(cov['cost'])} / {pct(cov['rev'])} | {err['cost']:.3f} / {err['cost_line']:.3f} / {err['cost_prior']:.3f} | {err['rev']:.3f} / {err['rev_line']:.3f} / {err['rev_prior']:.3f} |")
    out += ["", "The straight line extends the latest report to month 36 in proportion. The tracker divides by a ramp fitted to the deal database, which has",
            "the same shape as the generator's (a power of time); a real ramp of another shape would hurt it. Its cost intervals are too narrow on",
            "two databases (68% and 49% coverage): the fitted cost ramp exponent comes out 0.81–0.83 against a true 0.80, three or four of its own",
            "standard errors high, because deals that go quiet are not missing at random once revenue figures are missing too, and the model is",
            "linear where the truth has kinks. A few hundredths on the exponent moves a month-3 report by about 5%.", "",
            "## Customer matching, contract rules, application clusters", "",
            "| Data room | Matcher precision / recall | Exact name precision / recall | Overlap share: matcher / exact / truth | Change of control found / false / present | Convenience found / false / present | Clusters precision / recall | Same product precision / recall |",
            "|---|---|---|---|---|---|---|---|"]
    for (seed, p, rc, pe, re_, ov, ove, ovt), (_, tp, fp, pos, ttp, tfp, tpos), (_, ap_, ar, pp, pr) in zip(er, parse, apps):
        out.append(f"| {seed} | {p:.2f} / {rc:.2f} | {pe:.2f} / {re_:.2f} | {pct(ov)} / {pct(ove)} / {pct(ovt)} | {tp} / {fp} / {pos} | {ttp} / {tfp} / {tpos} | {ap_:.2f} / {ar:.2f} | {pp:.2f} / {pr:.2f} |")
    out += ["", "Change-of-control recall by wording, all five data rooms:", ""]
    for k in sorted(phr):
        out.append(f"- \"{W.COC[k]}\": {phr[k][0]} of {phr[k][1]}")
    out += ["", "These three are circular. The matcher's normalisation, the contract rules and the tags were written by the same hand as the generator's",
            "spelling variants, clause wordings and tag sets, so recall here is an upper bound. The matcher's lost precision is the sister entities",
            "(the same name with another legal suffix), which a normaliser that drops suffixes cannot tell apart. The rules miss the one wording",
            "that never names a change of control.", "", f"Run time {time.time() - t0:.0f} s on one core."]
    print("\n".join(out))


if __name__ == "__main__":
    main()
