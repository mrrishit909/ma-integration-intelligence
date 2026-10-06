"""The models. Each one learns from what the data room or the deal database shows, never from world.TRUTH.

entity resolution (customers)  ->  the overlap the synergy model needs
contract and cyber parsers     ->  risks with their evidence
application clustering         ->  overlapping systems and the IT savings pool
synergy realisation model      ->  synergies as distributions, by integration approach
ERP programme model            ->  how late programmes run and how often cutovers go wrong
integration simulator          ->  fast against staged, with the team size chosen by enumeration
realisation tracker            ->  quarterly reports turned into an updated forecast
"""
import difflib
import graphlib
import re

import numpy as np
from scipy import optimize, stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from . import world as W

VERSION = "ma-1.0"
FEATURES = ["related", "pct", "rel_size", "fast", "overlap"]          # pct = the announced synergy as a share of its base
LEGAL = {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "co", "company"}
ABBR = {"mfg": "manufacturing", "intl": "international", "eng": "engineering", "tech": "technologies", "svcs": "services",
        "dist": "distribution", "inds": "industries", "sys": "systems", "comp": "components"}


# ---------------------------------------------------------------- entity resolution

def normalise(name):
    s = re.sub(r"[.,]", "", name.lower().replace("&", " and "))
    toks = [ABBR.get(t, t) for t in s.split()]
    if toks and toks[0] == "the":
        toks = toks[1:]
    while toks and toks[-1] in LEGAL:
        toks = toks[:-1]
    return " ".join(toks)


def match_customers(target_names, acquirer_names, threshold=0.9):
    """-> {target index: (acquirer index, score, method)}. Exact on the normalised name, else the closest name sharing the first
    word if its similarity clears the threshold."""
    norm_a = [normalise(n) for n in acquirer_names]
    exact, block = {}, {}
    for j, n in enumerate(norm_a):
        exact.setdefault(n, j)
        block.setdefault(n.split(" ")[0], []).append(j)
    out = {}
    for i, name in enumerate(target_names):
        n = normalise(name)
        if n in exact:
            out[i] = (exact[n], 1.0, "normalised name")
            continue
        best = max(((difflib.SequenceMatcher(None, n, norm_a[j]).ratio(), j) for j in block.get(n.split(" ")[0], [])), default=(0, None))
        if best[0] >= threshold:
            out[i] = (best[1], round(best[0], 3), "fuzzy")
    return out


def match_exact(target_names, acquirer_names):
    idx = {n: j for j, n in enumerate(acquirer_names)}
    return {i: (idx[n], 1.0, "exact") for i, n in enumerate(target_names) if n in idx}


# ---------------------------------------------------------------- document parsers (rules, not a language model)

# a rule is one pattern, or several that must all appear in the sentence, in any order
COC_RULES = [(r"change (of|in) control", "names a change of control"),
             ((r"\b(acquired|merged? (into|with)|merger|sale of (all or )?substantially all)\b", r"\b(terminat\w*|end this agreement)\b"), "acquisition or merger gives a right to end"),
             ((r"\bassign\w*\b", r"\b(operation of law|merger)\b", r"\bconsent\b"), "assignment by merger needs consent")]
TFC_RULES = [(r"for convenience", "terminable for convenience"), (r"without cause", "terminable without cause"), (r"for any reason", "terminable for any reason")]


def parse_contract(text):
    """-> {'coc': [(clause no, sentence, rule)], 'tfc': [...]}"""
    out = {"coc": [], "tfc": []}
    for line in text.splitlines()[1:]:
        m = re.match(r"(\d+)\. (.*)", line)
        if not m:
            continue
        low = m[2].lower()
        for kind, rules in (("coc", COC_RULES), ("tfc", TFC_RULES)):
            hit = next((why for rx, why in rules if all(re.search(x, low) for x in ((rx,) if isinstance(rx, str) else rx))), None)
            if hit:
                out[kind].append((int(m[1]), m[2], hit))
    return out


FINDING = re.compile(r"Finding (F-\d+) — severity (\w+) \(CVSS ([\d.]+)\)\. (Internet-facing\. )?(.*?) Status: (open since|remediated) (\d{4}-\d{2}-\d{2})\.")
REMEDIATION = {"Critical": 1.2e6, "High": 5e5, "Medium": 1.5e5, "Low": 3e4}       # planning assumptions, stated on the page


def parse_cyber(text):
    return [{"id": m[1], "severity": m[2], "cvss": float(m[3]), "internet_facing": bool(m[4]), "finding": m[5], "open": m[6] == "open since", "date": m[7]}
            for m in FINDING.finditer(text)]


def concentration(revenue):
    s = np.sort(np.asarray(revenue, float))[::-1] / np.sum(revenue)
    return {"top1": float(s[0]), "top10": float(s[:10].sum()), "hhi": float((s ** 2).sum() * 10000)}


# ---------------------------------------------------------------- application overlap

def jaccard_distance(apps):
    tags = [set(a["tags"]) for a in apps]
    n = len(apps)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = 1 - len(tags[i] & tags[j]) / len(tags[i] | tags[j])
    return d


def cluster_apps(apps, cut=0.75):
    """Average-linkage clusters on the Jaccard distance between functional tags. -> cluster label per app."""
    if len(apps) < 2:
        return np.zeros(len(apps), int)
    z = hierarchy.linkage(squareform(jaccard_distance(apps)), "average")
    return hierarchy.fcluster(z, cut, "distance")


def overlap_pairs(apps, labels):
    """Cross-company pairs placed in the same cluster."""
    return {(i, j) for i, a in enumerate(apps) for j, b in enumerate(apps)
            if a["company"] == "acquirer" and b["company"] == "target" and labels[i] == labels[j]}


# ---------------------------------------------------------------- synergy realisation

def design(rows, kind):
    """-> per deal: features x (with the announced share for this kind), and its reports as (log months/36, log ratio)."""
    pct = "cost_pct" if kind == "cost" else "rev_pct"
    out = []
    for r in rows:
        reps = [(np.log(min(1.0, p["month"] / W.HORIZON_M)), np.log(p[kind])) for p in r["reports"] if p[kind] is not None]
        if reps:
            out.append((r["ref"], np.array([1.0, r["related"], r[pct], r["rel_size"], r["fast"], r["overlap"]]), reps))
    return out


def _gls(deals, su, se):
    """Generalised least squares for y = [x, log(m/36)] . beta + u_deal + e_report, given the two standard deviations.
    -> beta, its covariance, log-likelihood. Deals have one or two reports."""
    A, bvec = 0, 0
    blocks = []
    for ref, x, reps in deals:
        X = np.array([np.append(x, lm) for lm, _ in reps])
        y = np.array([v for _, v in reps])
        V = np.full((len(y), len(y)), su ** 2) + np.eye(len(y)) * se ** 2
        Vi = np.linalg.inv(V)
        A = A + X.T @ Vi @ X
        bvec = bvec + X.T @ Vi @ y
        blocks.append((X, y, V, Vi))
    cov = np.linalg.inv(A)
    beta = cov @ bvec
    ll = 0.0
    for X, y, V, Vi in blocks:
        e = y - X @ beta
        ll -= 0.5 * (np.linalg.slogdet(V)[1] + e @ Vi @ e + len(y) * np.log(2 * np.pi))
    return beta, cov, ll


def fit_synergy(rows):
    """Log realisation ratio, linear in the deal's features, plus k * log(months / 36) for how far up its ramp a deal was at each
    report, plus a deal effect (the deal's own surprise) and reporting noise, by maximum likelihood. Every deal's first-anniversary
    report is in the database, so deals that later went quiet still count, and young deals count at the point they have reached."""
    out, effects = {"n": len(rows)}, {}
    for kind in ("cost", "rev"):
        deals = design(rows, kind)
        res = optimize.minimize(lambda th: -_gls(deals, *np.exp(th))[2], np.log([0.3, 0.1]), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-3})
        su, se = np.exp(res.x)
        beta, cov, ll = _gls(deals, su, se)
        out[kind] = {"beta": beta[:-1].tolist(), "k": float(beta[-1]), "sd": float(su), "report_sd": float(se), "cov": cov.tolist(), "n": len(deals),
                     "reports": sum(len(d[2]) for d in deals)}
        effects[kind] = {ref: np.mean([v - np.append(x, lm) @ beta for lm, v in reps]) for ref, x, reps in deals}
    both = sorted(set(effects["cost"]) & set(effects["rev"]))
    out["rho"] = float(np.corrcoef([effects["cost"][r] for r in both], [effects["rev"][r] for r in both])[0, 1])
    latest = {r["ref"]: r["reports"][-1] for r in rows}
    surv = [latest[r["ref"]] for r in rows if latest[r["ref"]]["month"] >= W.HORIZON_M]
    out["comps"] = {"cost": float(np.mean([p["cost"] for p in surv])), "rev": float(np.mean([p["rev"] for p in surv if p["rev"] is not None])), "n": len(surv)}
    out["naive_latest"] = {"cost": float(np.mean([p["cost"] for p in latest.values()])), "rev": float(np.mean([p["rev"] for p in latest.values() if p["rev"] is not None]))}
    out["silent"] = sum(not r["still_disclosing"] for r in rows)
    out["young"] = sum(r["months_since_close"] < W.HORIZON_M for r in rows)
    return out


def deal_features(deal, revenue, ebitda, overlap):
    """The synergy model's inputs for a deal: what the acquirer knows, with the overlap from entity resolution."""
    mc = deal["management_case"]
    return {"related": float(deal["relatedness"]), "cost_pct": mc["cost_run_rate"] / (revenue - ebitda), "rev_pct": mc["revenue_run_rate"] / revenue,
            "rel_size": revenue / float(deal["acquirer_revenue"]), "overlap": overlap}


def x_of(f, kind, fast):
    return np.array([1.0, f["related"], f["cost_pct" if kind == "cost" else "rev_pct"], f["rel_size"], 1.0 if fast else 0.0, f["overlap"]])


def predict(model, f, fast, n, rng):
    """Predictive draws of the realisation ratios (cost, revenue): coefficient uncertainty plus the deal's own surprise."""
    out = []
    z = rng.standard_normal((n, 2))
    zc, zr = z[:, 0], model["rho"] * z[:, 0] + np.sqrt(1 - model["rho"] ** 2) * z[:, 1]
    for kind, eps in (("cost", zc), ("rev", zr)):
        m = model[kind]
        x = x_of(f, kind, fast)
        b = rng.multivariate_normal(m["beta"], np.array(m["cov"])[:-1, :-1], n)      # the ramp term is zero at month 36
        out.append(np.exp(b @ x + m["sd"] * eps))
    return out[0], out[1]


def truth_draws(f, fast, n, rng):
    ff = {k: np.full(n, v, float) for k, v in f.items()}
    ff["fast"] = np.full(n, 1.0 if fast else 0.0)
    lc, lr = W.true_log_r(ff, rng.standard_normal((n, 2)))
    return np.exp(lc), np.exp(lr)


def q(a, p=(0.1, 0.5, 0.9)):
    return [float(x) for x in np.quantile(a, p)]


# ---------------------------------------------------------------- ERP programmes

def fit_erp(rows):
    """Actual / planned weeks is lognormal. A finished programme gives its ratio; one still running says only that its ratio is above
    weeks-so-far / planned (right-censored maximum likelihood). Cutover incidents: a Beta(1, 1) prior updated by finished programmes."""
    progs = [r["erp"] for r in rows if r["erp"]]
    done = np.array([p["done"] for p in progs])
    lr = np.log([p["weeks"] / p["planned_weeks"] for p in progs])

    def nll(th):
        mu, ls = th
        sd = np.exp(ls)
        return -(stats.norm.logpdf(lr[done], mu, sd).sum() + stats.norm.logsf(lr[~done], mu, sd).sum())
    res = optimize.minimize(nll, [lr[done].mean(), np.log(lr[done].std() + 1e-3)], method="Nelder-Mead")
    inc = {}
    for a in ("fast", "staged"):
        fin = [p for p in progs if p["done"] and p["approach"] == a]
        k = sum(p["incident"] for p in fin)
        inc[a] = {"incidents": k, "programmes": len(fin), "a": 1 + k, "b": 1 + len(fin) - k}
    wk = np.log([p["incident_weeks"] for p in progs if p["incident_weeks"]])
    return {"bias_mu": float(res.x[0]), "bias_sd": float(np.exp(res.x[1])), "programmes": len(progs), "running": int((~done).sum()),
            "finished_only_mu": float(lr[done].mean()), "incident": inc, "weeks_mu": float(wk.mean()), "weeks_sd": float(wk.std(ddof=1)), "incidents_seen": len(wk)}


# ---------------------------------------------------------------- the integration plan

def workstreams(interfaces):
    """interfaces: [(app ref, product, users)] that touch the target's ERP. Planned weeks are the integration team's estimates."""
    ifs = [{"key": f"if-{ref}", "name": f"Rebuild interface: {product}", "weeks": float(np.clip(round(2 + np.log10(max(users, 10))), 3, 8))} for ref, product, users in interfaces]
    return {"core": [{"key": "coa", "name": "Chart of accounts and entity mapping", "weeks": 6.0, "after": []},
                     {"key": "clean", "name": "Master data cleansing", "weeks": 8.0, "after": []},
                     {"key": "migrate", "name": "Data migration build", "weeks": 10.0, "after": ["coa", "clean"]},
                     {"key": "test", "name": "Integrated testing", "weeks": 6.0, "after": ["migrate", "interfaces"]},
                     {"key": "train", "name": "Training", "weeks": 4.0, "after": ["test"]},
                     {"key": "cutover", "name": "Cutover", "weeks": 2.0, "after": ["train"]},
                     {"key": "hyper", "name": "Hypercare and legacy decommission", "weeks": 6.0, "after": ["cutover"]}],
            "interfaces": ifs}


def lanes_of(ifs, lanes):
    """Longest-processing-time assignment of interface rebuilds to parallel teams. -> weeks until the last lane finishes."""
    load = [0.0] * lanes
    for t in sorted(ifs, key=lambda t: -t["weeks"]):
        load[load.index(min(load))] += t["weeks"]
    return max(load)


def critical_path(tasks):
    """tasks: [{key, weeks, after}] -> {key: (start, finish)} by longest path in topological order."""
    ts = graphlib.TopologicalSorter({t["key"]: t["after"] for t in tasks})
    by, out = {t["key"]: t for t in tasks}, {}
    for k in ts.static_order():
        s = max((out[a][1] for a in by[k]["after"]), default=0.0)
        out[k] = (s, s + by[k]["weeks"])
    return out


LATER_WAVE = {"migrate": 0.5, "test": 0.6, "train": 0.6, "cutover": 1.0, "hyper": 0.6}


def plan(ws, approach, lanes):
    """Planned schedule. Fast: one cutover for every entity. Staged: the first wave builds everything for its entity; later waves
    repeat migration, testing, training and cutover at reduced effort, each after the previous cutover.
    -> {'tasks': [(wave, key, name, start, finish)], 'cutovers': [(week, revenue share)], 'done': week}"""
    core = [dict(t) for t in ws["core"]] + [{"key": "interfaces", "name": f"Interface rebuilds ({len(ws['interfaces'])}, {lanes} teams)", "weeks": lanes_of(ws["interfaces"], lanes), "after": []}]
    cp = critical_path(core)
    names = {t["key"]: t["name"] for t in core}
    tasks = [(1, k, names[k], *cp[k]) for k in names]
    if approach == "fast":
        return {"tasks": tasks, "cutovers": [(cp["cutover"][1], 1.0)], "done": cp["hyper"][1], "waves": ["All entities"]}
    ents = W.ENTITIES
    cutovers, t0 = [(cp["cutover"][1], ents[0][1])], cp["cutover"][1]
    done = cp["hyper"][1]
    for w, (name, share) in enumerate(ents[1:], 2):
        prev = None
        for k in ("migrate", "test", "train", "cutover", "hyper"):
            s = t0 if prev is None else prev
            f = s + next(t["weeks"] for t in ws["core"] if t["key"] == k) * LATER_WAVE[k]
            tasks.append((w, k, f"{names[k]} ({name})", s, f))
            prev = f if k != "hyper" else prev
            if k == "cutover":
                cutovers.append((f, share))
                t0 = f
            if k == "hyper":
                done = max(done, f)
    return {"tasks": tasks, "cutovers": cutovers, "done": done, "waves": [e[0] for e in ents]}


COSTS = {"core_team_week": {"fast": 2.6e5, "staged": 1.6e5}, "lane_week": 4.5e4, "dual_running_week": 7e4, "lost_margin_on_disrupted_revenue": 0.30}
WEEKS = 260
DISCOUNT, MULTIPLE = 0.09, 9.0       # synergies valued over five years, then at nine times run-rate EBITDA


def simulate(p, rc, rr, ratio, incident, inc_weeks, k_cost, k_rev, approach, lanes, deal, revenue):
    """Vectorised value of one plan over draws. rc, rr: realisation ratios; ratio: actual / planned duration; incident: (draws, cutovers)
    booleans; inc_weeks: (draws, cutovers) weeks of disruption. -> dict of arrays."""
    n = len(rc)
    mc = deal["management_case"]
    w = np.arange(WEEKS)[None, :]
    disc = (1 + DISCOUNT) ** (-w / 52)
    m = (w + 0.5) / 4.345
    done = p["done"] * ratio[:, None]
    it_on = np.clip((w - done) / 26, 0, 1)
    cost_syn = mc["cost_run_rate"] * rc[:, None] / 52 * ((1 - mc["it_share"]) * W.ramp(m, k_cost) + mc["it_share"] * it_on)
    rev_syn = mc["revenue_run_rate"] * mc["revenue_margin"] * rr[:, None] / 52 * W.ramp(m, k_rev)
    burn = (COSTS["core_team_week"][approach] + lanes * COSTS["lane_week"] + COSTS["dual_running_week"]) * (w < done)
    pv = ((cost_syn + rev_syn - burn) * disc).sum(1)
    loss = np.zeros(n)
    for c, (week, share) in enumerate(p["cutovers"]):
        t = week * ratio
        loss += incident[:, c] * inc_weeks[:, c] * revenue / 52 * share * COSTS["lost_margin_on_disrupted_revenue"] * (1 + DISCOUNT) ** (-t / 52)
    run_rate = mc["cost_run_rate"] * rc * ((1 - mc["it_share"]) + mc["it_share"] * it_on[:, -1]) + mc["revenue_run_rate"] * mc["revenue_margin"] * rr
    terminal = run_rate * MULTIPLE * (1 + DISCOUNT) ** (-WEEKS / 52)
    return {"value": pv - loss + terminal, "done_weeks": done[:, 0], "disrupted": incident.any(1), "loss": loss,
            "cost": ((burn * disc).sum(1)), "run_rate": run_rate}


def draws_model(model, erp, f, approach, n, seed, cutovers):
    rng = np.random.default_rng(seed)
    rc, rr = predict(model, f, approach == "fast", n, rng)
    ratio = np.exp(rng.normal(erp["bias_mu"], erp["bias_sd"], n))
    pinc = rng.beta(erp["incident"][approach]["a"], erp["incident"][approach]["b"], n)
    incident = rng.random((n, cutovers)) < pinc[:, None]
    weeks = np.exp(rng.normal(erp["weeks_mu"], erp["weeks_sd"], (n, cutovers)))
    return rc, rr, ratio, incident, weeks


def draws_truth(f, approach, n, seed, cutovers):
    rng = np.random.default_rng(seed)
    rc, rr = truth_draws(f, approach == "fast", n, rng)
    e = W.TRUTH["erp"]
    ratio = np.exp(rng.normal(e["bias_mu"], e["bias_sd"], n))
    incident = rng.random((n, cutovers)) < e["p_incident"][approach]
    weeks = np.exp(rng.normal(e["weeks_mu"], e["weeks_sd"], (n, cutovers)))
    return rc, rr, ratio, incident, weeks


def evaluate_option(ws, model, erp, f, approach, lanes, n, seed, deal, revenue, truth_f=None):
    p = plan(ws, approach, lanes)
    k = len(p["cutovers"])
    sim = simulate(p, *draws_model(model, erp, f, approach, n, seed, k), model["cost"]["k"], model["rev"]["k"], approach, lanes, deal, revenue)
    out = {"plan": p, "sim": sim}
    if truth_f is not None:
        out["truth"] = simulate(p, *draws_truth(truth_f, approach, n, seed + 1, k), W.TRUTH["ramp"]["cost"], W.TRUTH["ramp"]["rev"], approach, lanes, deal, revenue)
    return out


def best_lanes(ws, model, erp, f, approach, deal, revenue, n=4000, seed=7):
    """Integration team size by enumeration: the number of parallel interface teams with the highest expected value."""
    options = range(1, 9) if approach == "fast" else range(1, 7)
    scores = {L: float(evaluate_option(ws, model, erp, f, approach, L, n, seed, deal, revenue)["sim"]["value"].mean()) for L in options}
    return max(scores, key=scores.get), scores


def management_case(ws, model, approach, lanes, deal, revenue):
    """The plan as written: every synergy at its announced run-rate, every task on its planned duration, no cutover incident."""
    p = plan(ws, approach, lanes)
    one, none = np.ones(1), np.zeros((1, len(p["cutovers"])))
    return simulate(p, one, one, one, none.astype(bool), none, model["cost"]["k"], model["rev"]["k"], approach, lanes, deal, revenue)


# ---------------------------------------------------------------- realisation tracking

def track(model, f, approach, reports, mc, n=20000, seed=5):
    """Normal update of the deal's log realisation ratio from its quarterly run-rate reports. Prior: the model's predictive for the
    approach taken. Each report, divided by the fitted ramp, measures the final ratio with the fitted reporting noise, plus the error
    in the fitted ramp, which every report shares and which is largest in the first months."""
    rng = np.random.default_rng(seed)
    rc, rr = predict(model, f, approach == "fast", n, rng)
    out = {}
    for kind, prior, base, col in (("cost", rc, mc["cost_run_rate"], "cost_run_rate"), ("rev", rr, mc["revenue_run_rate"], "rev_run_rate")):
        m = model[kind]
        k_var = np.array(m["cov"])[-1, -1]
        mu0, var0 = float(np.log(prior).mean()), float(np.log(prior).var())
        prior_q = q(prior)
        lm = np.log(np.array([r["month"] for r in reports]) / W.HORIZON_M)
        y = np.log(np.array([r[col] for r in reports]) / base) - m["k"] * lm
        Si = np.linalg.inv(m["report_sd"] ** 2 * np.eye(len(y)) + k_var * np.outer(lm, lm))
        prec = 1 / var0 + Si.sum()
        mu, var = (mu0 / var0 + (Si @ y).sum()) / prec, 1 / prec
        post = np.exp(rng.normal(mu, np.sqrt(var), n))
        last = reports[-1]
        straight = last[col] / base * W.HORIZON_M / last["month"]          # the PMO's straight line to month 36
        out[kind] = {"prior": prior_q, "posterior": q(post), "p_meets_case": float((post >= 1).mean()), "p_below_70pct": float((post < 0.7).mean()),
                     "straight_line": float(straight), "management_case": base}
    return out
