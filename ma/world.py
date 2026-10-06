"""The synthetic world: an acquirer, a target and its data room, a licensed database of comparable deals, and the true processes
behind all of them. Nothing here is a real company, contract, deal or vulnerability.

engine.py sees only what the data room and the deal database show. TRUTH, the true customer overlap, the true capability of every
application and the deal's hidden outcome are used only to score the models.
"""
import numpy as np

HORIZON_M = 36            # synergies are judged at their run-rate three years after close

# The true process behind every deal's synergy realisation (realised run-rate / announced run-rate, three years after close).
# The model in engine.py is linear in the same features; the two kinks below (over-ambitious cost targets, very large targets) it
# does not know about.
TRUTH = {
    "cost": {"b0": -0.12, "related": 0.15, "aggr_kink": 0.08, "aggr": -3.0, "size_kink": 0.30, "size": -0.8, "fast": 0.06, "sd": 0.22},
    "rev": {"b0": -0.50, "related": 0.45, "overlap": -1.4, "fast": -0.35, "sd": 0.50},
    "rho": 0.35,                       # correlation of the cost and revenue surprises within a deal (the same integration team)
    "meas_sd": 0.08,                   # noise in what acquirers report
    "ramp": {"cost": 0.8, "rev": 1.5},  # share of run-rate reached at month m: (m/36)^k; revenue comes later
    "erp": {"bias_mu": 0.30, "bias_sd": 0.28,                          # ERP programmes: actual / planned duration is lognormal
            "p_incident": {"fast": 0.35, "staged": 0.08},              # an order-to-cash disruption at a cutover
            "weeks_mu": float(np.log(4.0)), "weeks_sd": 0.6},          # and how many weeks it lasts
}

ACQUIRER = "Aldermoor Group"
DEAL = {"external_ref": "KESTREL", "name": "Project Kestrel", "target_name": "Corvane Systems", "acquirer_revenue": 2.4e9, "price": 1.2e9,
        "standalone_value": 9.4e8, "relatedness": 0.7, "proposed_approach": "fast",
        "management_case": {"cost_run_rate": 30e6, "revenue_run_rate": 45e6, "revenue_margin": 0.40, "it_share": 0.35}}

SYL = ["al", "bran", "cor", "del", "fen", "gar", "hal", "ist", "jor", "kel", "lom", "mar", "nor", "ost", "pel", "quin", "ros",
       "sel", "tam", "ul", "ver", "wen", "yar", "zel", "bec", "dun", "esk", "frey", "gil", "hav", "lin", "mor", "pry", "rav", "tor", "vik"]
INDUSTRY = ["Mechanical", "Metals", "Fluid Controls", "Logistics", "Foods", "Packaging", "Plastics", "Electric", "Pumps & Valves",
            "Fabrication", "Instruments", "Coatings", "Chemicals", "Robotics", "Energy", "Freight", "Hydraulics", "Castings", "Controls",
            "Materials", "Components", "Systems", "Technologies", "Engineering", "Services", "Manufacturing", "Distribution",
            "Industries", "International", "Tool & Die"]
SUFFIX = ["Inc", "LLC", "Ltd", "Corp", "Co", "", "Inc", "LLC"]
SUFFIX_VARIANT = {"Inc": ["Inc.", "Incorporated"], "LLC": ["L.L.C."], "Ltd": ["Limited", "Ltd."], "Corp": ["Corporation", "Corp."], "Co": ["Company", "Co."], "": ["Inc"]}
ABBREV = {"Manufacturing": "Mfg", "International": "Intl", "Engineering": "Eng", "Technologies": "Tech", "Services": "Svcs",
          "Distribution": "Dist", "Industries": "Inds", "Systems": "Sys", "Components": "Comp"}

CAPS = {   # true capability -> functional tags
    "erp": ["general ledger", "accounts payable", "accounts receivable", "fixed assets", "procurement", "inventory"],
    "crm": ["accounts", "opportunities", "pipeline", "contacts", "forecasting", "campaigns"],
    "hcm": ["employee records", "org chart", "performance", "recruiting", "onboarding", "time off"],
    "payroll": ["payroll", "tax filing", "pay slips", "deductions", "garnishments", "time off"],
    "billing": ["invoicing", "subscriptions", "usage rating", "dunning", "revenue recognition", "credit notes"],
    "field_service": ["work orders", "dispatch", "technician scheduling", "spare parts", "mobile app", "service contracts"],
    "plm": ["bill of materials", "engineering change", "cad vault", "product specs", "compliance docs", "revision control"],
    "data_warehouse": ["etl", "data models", "sql", "data lake", "lineage", "job scheduling"],
    "bi": ["dashboards", "reports", "kpis", "self-service", "alerts", "data models"],
    "itsm": ["incidents", "service desk", "change requests", "asset register", "sla", "knowledge base"],
    "identity": ["single sign-on", "mfa", "directory", "provisioning", "access reviews", "password reset"],
    "collab": ["email", "chat", "video calls", "documents", "calendar", "intranet"],
    "expense": ["expense reports", "receipts", "corporate cards", "mileage", "approvals", "per diem"],
    "treasury": ["cash positioning", "bank connectivity", "fx", "payments", "liquidity forecast", "hedging"],
    "ecommerce": ["storefront", "catalog", "cart", "checkout", "promotions", "order capture"],
    "cpq": ["configurator", "price books", "quotes", "discount approvals", "proposals", "contract generation"],
}
PRODUCTS = {   # invented products per capability; two companies on the same product is the easy overlap
    "erp": ["Ledgerline ERP", "Corvid Financials", "Atlas One"], "crm": ["Pinnacle CRM", "Relata", "Northpass Sales"],
    "hcm": ["Peoplegrid", "Staffwise"], "payroll": ["Paystream", "Ledgerline Payroll"], "billing": ["Invoya", "Recurra"],
    "field_service": ["Fieldmark", "Dispatchly"], "plm": ["Revisio", "Partwise PLM"], "data_warehouse": ["Quarry DW", "Tessel"],
    "bi": ["Lumen BI", "Chartwell"], "itsm": ["Servio", "Ticketry"], "identity": ["Keystone ID", "Gatekeep"],
    "collab": ["Teamhall", "Mailmoor"], "expense": ["Spendly", "Claimit"], "treasury": ["Cashwise", "Treasuro"],
    "ecommerce": ["Shopfront", "Cartell"], "cpq": ["Quotient CPQ", "Configura"],
}
ERP_LINKS = ["crm", "billing", "payroll", "field_service", "plm", "data_warehouse", "ecommerce", "expense", "treasury", "cpq", "hcm"]
OTHER_LINKS = [("crm", "cpq"), ("hcm", "payroll"), ("data_warehouse", "bi"), ("field_service", "crm"), ("billing", "crm")]
ENTITIES = [("North America", 0.55), ("Europe", 0.30), ("Asia-Pacific", 0.15)]

COC = ["In the event of a Change of Control of Supplier, Customer may terminate this Agreement upon thirty (30) days' written notice.",
       "Customer may terminate this Agreement if Supplier undergoes a change in control, merger or sale of substantially all of its assets.",
       "This Agreement may not be assigned, including by operation of law or by merger, without Customer's prior written consent.",
       "Should Supplier be acquired by, or merged into, any third party, Customer shall have the right to end this Agreement without penalty.",
       "If more than fifty percent (50%) of the voting securities of Supplier are transferred, Customer may elect to terminate."]
TFC = ["Customer may terminate this Agreement for convenience on ninety (90) days' written notice.",
       "Customer may end this Agreement at any time, for any reason, on sixty days' notice.",
       "Either party may terminate without cause upon one hundred twenty days' prior written notice."]
BOILER = ["Either party may terminate this Agreement for material breach not cured within thirty days of notice.",
          "Customer may assign this Agreement to an affiliate without consent.",
          "Supplier shall notify Customer of any change of address for notices.",
          "Each party's aggregate liability is limited to the fees paid in the twelve months preceding the claim.",
          "Each party shall keep the other's confidential information secret for five years after termination.",
          "Supplier warrants that the services will be performed in a professional and workmanlike manner.",
          "Neither party is liable for delay caused by events beyond its reasonable control."]
FINDINGS = [   # (severity, cvss, internet-facing, text)
    ("Critical", 9.8, True, "VPN gateway two firmware versions behind; a public remote-code-execution exploit exists."),
    ("Critical", 9.1, True, "Device telemetry API accepts unsigned firmware update requests from fielded IoT gateways."),
    ("High", 8.1, True, "Customer support portal allows password reuse and has no multi-factor authentication for administrators."),
    ("High", 8.0, False, "Third-party remote access tool installed on plant-floor engineering workstations."),
    ("High", 7.8, False, "Shared local administrator password across 340 field technician laptops."),
    ("High", 7.5, False, "Unsupported database version on the billing server; vendor patches ended in 2024."),
    ("High", 7.4, True, "Customer data exported to an unmanaged cloud storage bucket for analytics."),
    ("Medium", 6.5, False, "Seventeen former employees' accounts still active in the code repository."),
    ("Medium", 6.4, False, "No network segmentation between office and test-lab networks."),
    ("Medium", 6.0, False, "ERP backups are not tested; the last restore test was 26 months ago."),
    ("Medium", 5.9, False, "Secrets committed to a public code repository in 2025, since rotated."),
    ("Low", 3.7, False, "TLS 1.0 still enabled on the legacy licence server."),
    ("Low", 3.5, False, "Perimeter firewall logs kept for 14 days."),
    ("Low", 3.0, False, "Phishing simulation click rate of 19%."),
]


def ramp(m, k):
    return np.minimum(1.0, np.asarray(m, float) / HORIZON_M) ** k


def true_log_r(f, z):
    """f: feature dict of arrays; z: (n, 2) standard normals. -> true log realisation ratios (cost, revenue)."""
    c, v = TRUTH["cost"], TRUTH["rev"]
    u = c["sd"] * z[:, 0]
    w = v["sd"] * (TRUTH["rho"] * z[:, 0] + np.sqrt(1 - TRUTH["rho"] ** 2) * z[:, 1])
    lc = (c["b0"] + c["related"] * f["related"] + c["aggr"] * np.maximum(0, f["cost_pct"] - c["aggr_kink"])
          + c["size"] * np.maximum(0, f["rel_size"] - c["size_kink"]) + c["fast"] * f["fast"] + u)
    lr = v["b0"] + v["related"] * f["related"] + v["overlap"] * f["overlap"] + v["fast"] * f["fast"] + w
    return lc, lr


def features(r, n):
    return {"related": r.uniform(0.1, 1.0, n), "cost_pct": r.uniform(0.03, 0.18, n), "rev_pct": r.uniform(0.03, 0.25, n),
            "rel_size": np.clip(np.exp(r.normal(np.log(0.12), 0.7, n)), 0.02, 0.9), "fast": (r.random(n) < 0.4).astype(float),
            "overlap": r.uniform(0.0, 0.45, n)}


def comparables(seed, n=240):
    """The licensed deal database: what acquirers disclosed about past deals. Deals under three years old are still ramping; deals
    that fell behind often stopped disclosing after their first year. ERP programmes still running are known only to be longer
    than they have run so far, and whether they had a cutover incident is not yet known."""
    r = np.random.default_rng([seed, 1])
    f = features(r, n)
    lc, lr = true_log_r(f, r.standard_normal((n, 2)))
    months = r.integers(3, 121, n)
    weak = (np.exp(lc) < 0.7) | (np.exp(lr) < 0.3)
    silent = (months > 12) & weak & (r.random(n) < 0.75)
    k, sd = TRUTH["ramp"], TRUTH["meas_sd"]
    no_rev = r.random(n) < 0.2                       # some acquirers never put a number on revenue synergies
    first = np.minimum(months, 12)                   # every deal's first-anniversary (or only) report
    second = np.where((months > 12) & ~silent, months, 0)      # and its latest, unless it went quiet
    reports = []
    for i in range(n):
        reps = []
        for m in (first[i], second[i]):
            if m:
                e = r.normal(0, sd, 2)
                reps.append({"month": int(m), "cost": round(float(np.exp(lc[i]) * ramp(m, k["cost"]) * np.exp(e[0])), 4),
                             "rev": None if no_rev[i] else round(float(np.exp(lr[i]) * ramp(m, k["rev"]) * np.exp(e[1])), 4)})
        reports.append(reps)
    e = TRUTH["erp"]
    has_erp = r.random(n) < 0.6
    planned = np.round(r.uniform(26, 78, n))
    actual = planned * np.exp(r.normal(e["bias_mu"], e["bias_sd"], n))
    elapsed = np.maximum(0, (months - 2) * 4.345)
    done = elapsed >= actual
    p = np.where(f["fast"] == 1, e["p_incident"]["fast"], e["p_incident"]["staged"])
    incident = r.random(n) < p
    weeks = np.exp(r.normal(e["weeks_mu"], e["weeks_sd"], n))
    rows = []
    for i in range(n):
        rows.append({"ref": f"CD-{seed % 1000:03d}-{i + 1:04d}", "months_since_close": int(months[i]),
                     "still_disclosing": bool(not silent[i]), **{kk: round(float(vv[i]), 4) for kk, vv in f.items()},
                     "reports": reports[i],
                     "erp": None if not has_erp[i] else {"planned_weeks": int(planned[i]), "approach": "fast" if f["fast"][i] else "staged", "done": bool(done[i]),
                                                          "weeks": round(float(actual[i] if done[i] else elapsed[i]), 1),
                                                          "incident": bool(incident[i]) if done[i] else None,
                                                          "incident_weeks": round(float(weeks[i]), 1) if done[i] and incident[i] else None}})
    return rows, {"lc": lc, "lr": lr, "features": f}


def _names(r, n):
    seen, out = set(), []
    while len(out) < n:
        first = (r.choice(SYL) + r.choice(SYL) + (r.choice(SYL) if r.random() < 0.3 else "")).capitalize()
        name = f"{first} {r.choice(INDUSTRY)}"
        if name not in seen:
            seen.add(name)
            out.append((name, str(r.choice(SUFFIX))))
    return out


def _variant(r, name, suffix):
    """How the acquirer's CRM spells the same customer."""
    kind = r.choice(["exact", "exact", "suffix", "upper", "abbrev", "typo", "the", "amp"])
    base = name
    if kind == "abbrev" and any(w in name for w in ABBREV):
        w = next(w for w in ABBREV if w in name)
        base = name.replace(w, ABBREV[w])
    elif kind == "typo":
        word = name.split(" ")[1]
        j = int(r.integers(1, len(word) - 1))
        base = name.replace(word, word[:j - 1] + word[j] + word[j - 1] + word[j + 1:]) if word[j] != word[j - 1] else name
    elif kind == "amp" and "&" in name:
        base = name.replace("&", "and")
    elif kind == "the":
        base = "The " + name
    elif kind in ("abbrev", "amp"):
        kind = "suffix"
    sfx = str(r.choice(SUFFIX_VARIANT[suffix])) if kind == "suffix" else suffix
    full = f"{base} {sfx}".strip()
    return (full.upper() if kind == "upper" else full), str(kind)


def data_room(seed):
    """Everything the seller's virtual data room and the acquirer's own systems hold, plus the hidden truth for scoring."""
    r = np.random.default_rng([seed, 2])
    D = DEAL
    # financials: twelve quarters, growing, the last four summing to about $310M
    q = np.arange(12)
    rev = 62e6 * (1.09 ** (q / 4)) * np.exp(r.normal(0, 0.02, 12))
    rev *= 310e6 / rev[-4:].sum()
    gm = 0.58 + r.normal(0, 0.008, 12)
    opex = {"sales_marketing": 0.18, "research_development": 0.14, "general_admin": 0.09, "field_operations": 0.03}
    fin = []
    for i in range(12):
        lines = {"revenue": rev[i], "gross_profit": rev[i] * gm[i], **{k: rev[i] * (v + r.normal(0, 0.004)) for k, v in opex.items()}}
        lines["ebitda"] = lines["gross_profit"] - sum(lines[k] for k in opex)
        fin += [{"period": f"FY{2024 + i // 4}-Q{i % 4 + 1}", "metric": k, "value": round(float(v))} for k, v in lines.items()]
    # customers: heavy-tailed, the largest set between 12% and 20% of revenue
    n_cust = 420
    w = np.sort(np.exp(r.normal(0, 1.5, n_cust)))[::-1]
    w[0] = 0
    w = w / w.sum() * (1 - (top := r.uniform(0.12, 0.20)))
    w[0] = top
    tnames = _names(r, n_cust)
    overlap = r.random(n_cust) < 0.30
    customers = [{"ref": f"TC-{i + 1:04d}", "name": f"{n} {s}".strip(), "revenue": round(float(w[i] * 310e6)), "true_overlap": bool(overlap[i])} for i, (n, s) in enumerate(tnames)]
    # the acquirer's customer master: the shared customers spelled its own way, sister entities, and everyone else
    acq, truth_pairs = [], {}
    for i, (n, s) in enumerate(tnames):
        if overlap[i]:
            v, kind = _variant(r, n, s)
            truth_pairs[len(acq)] = (i, kind)
            acq.append(v)
    sisters = [i for i in np.flatnonzero(~overlap) if r.random() < 0.05]
    for i in sisters:      # a different legal entity of the same group: same name, another suffix (not the same customer)
        n, s = tnames[i]
        acq.append(f"{n} {[x for x in ('Ltd', 'Inc', 'LLC', 'Corp') if x != s][int(r.integers(0, 3))]}")
    others = _names(np.random.default_rng([seed, 3]), 2600)
    tset = {n for n, _ in tnames}
    acq += [f"{n} {s}".strip() for n, s in others if n not in tset][: 2600 - len(acq)]
    perm = r.permutation(len(acq))
    acq = [acq[j] for j in perm]
    inv = {int(new): old for new, old in enumerate(perm)}
    acq_truth = {new: truth_pairs[old] for new, old in inv.items() if old in truth_pairs}
    # contracts with the forty largest customers
    docs, clause_truth = [], []
    for rank in range(40):
        c = customers[rank]
        coc = r.random() < 0.38
        tfc = r.random() < 0.22
        ci = int(r.integers(0, len(COC))) if coc else None
        ti = int(r.integers(0, len(TFC))) if tfc else None
        body = [f"This Agreement commences on the Effective Date and continues for {int(r.integers(2, 6))} years.",
                "Thereafter it renews automatically for successive one-year terms unless either party gives ninety days' notice.",
                f"Invoices are payable within {int(r.choice([30, 45, 60]))} days."]
        body += list(r.choice(BOILER, size=4, replace=False))
        if coc:
            body.insert(int(r.integers(2, len(body))), COC[ci])
        if tfc:
            body.insert(int(r.integers(2, len(body))), TFC[ti])
        body.append(f"This Agreement is governed by the laws of the State of {r.choice(['Delaware', 'New York', 'Texas', 'Illinois'])}.")
        text = f"MASTER SERVICES AGREEMENT between {D['target_name']} (\"Supplier\") and {c['name']} (\"Customer\").\n" + "\n".join(f"{k + 1}. {s}" for k, s in enumerate(body))
        docs.append({"ref": f"VDR-2.{rank + 1:02d}", "kind": "customer_contract", "title": f"MSA - {c['name']}", "customer_ref": c["ref"], "text": text})
        clause_truth.append({"doc": f"VDR-2.{rank + 1:02d}", "coc": ci, "tfc": ti})
    # the cyber assessment
    pick = sorted(r.choice(len(FINDINGS), size=12, replace=False))
    lines, ftruth = [], []
    for k, j in enumerate(pick):
        sev, cvss, net, txt = FINDINGS[j]
        open_ = r.random() < (0.75 if sev in ("Critical", "High") else 0.5)
        day = f"2025-{int(r.integers(3, 13)):02d}-{int(r.integers(1, 29)):02d}" if open_ else f"2026-{int(r.integers(1, 9)):02d}-{int(r.integers(1, 29)):02d}"
        lines.append(f"Finding F-{k + 1:02d} — severity {sev} (CVSS {cvss}). {'Internet-facing. ' if net else ''}{txt} Status: {'open since' if open_ else 'remediated'} {day}.")
        ftruth.append({"id": f"F-{k + 1:02d}", "severity": sev, "open": bool(open_), "internet_facing": net})
    docs.append({"ref": "VDR-7.01", "kind": "cyber_assessment", "title": "Independent cyber security assessment, summary of findings", "customer_ref": None,
                 "text": f"Assessment of {D['target_name']} by an independent firm, fieldwork completed 2026-08.\n" + "\n".join(lines)})
    # application portfolios and their interfaces
    apps, edges = [], []
    rs = np.random.default_rng([seed, 4])
    for company, p_have, p_second in (("acquirer", 1.0, 0.35), ("target", 0.85, 0.0)):
        mine = {}
        for cap in CAPS:
            if company == "target" and cap == "erp" or rs.random() < p_have:
                for _ in range(1 + (rs.random() < p_second)):
                    tags = [t for t in CAPS[cap] if rs.random() < 0.7] or CAPS[cap][:2]
                    if rs.random() < 0.35:
                        other = str(rs.choice([c for c in CAPS if c != cap]))
                        tags.append(str(rs.choice(CAPS[other])))
                    cost = float(np.exp(rs.normal(np.log(2.4e6 if cap == "erp" else 4e5), 0.4)))
                    a = {"ref": f"{'A' if company == 'acquirer' else 'T'}-APP-{len(apps) + 1:03d}", "company": company, "product": str(rs.choice(PRODUCTS[cap])),
                         "tags": sorted(set(tags)), "annual_cost": round(cost, -3), "users": int(np.exp(rs.normal(np.log(1500 if company == "acquirer" else 400), 0.6))), "true_capability": cap}
                    apps.append(a)
                    mine.setdefault(cap, []).append(a["ref"])
        for cap in ERP_LINKS:
            for e in mine.get(cap, []):
                edges.append((mine["erp"][0], e, "interface"))
        for x, y in OTHER_LINKS:
            if x in mine and y in mine:
                edges.append((mine[x][0], mine[y][0], "interface"))
    truth = {"customer_overlap_share": float(w[overlap].sum()), "acquirer_matches": acq_truth, "clauses": clause_truth, "findings": ftruth,
             "sisters": len(sisters)}
    return {"financials": fin, "customers": customers, "acquirer_customers": acq, "documents": docs, "apps": apps, "edges": edges, "truth": truth}


def hidden_outcome(seed):
    """The deal's own fate, drawn once: the standard normals behind its true realisation under either approach."""
    return np.random.default_rng([seed, 9]).standard_normal(2)


def realisation_reports(seed, approach, quarters, f, mgmt_cost, mgmt_rev):
    """Quarterly synergy run-rate reports after close, from the hidden outcome. f: the deal's features with its true overlap."""
    rw = np.random.default_rng([seed, 10])
    ff = {k: np.array([v], float) for k, v in f.items()}
    ff["fast"] = np.array([1.0 if approach == "fast" else 0.0])
    lc, lr = true_log_r(ff, hidden_outcome(seed)[None, :])
    out = []
    for q in range(1, quarters + 1):
        m = 3 * q
        e = rw.normal(0, TRUTH["meas_sd"], 2)
        out.append({"quarter": q, "month": m, "cost_run_rate": float(mgmt_cost * np.exp(lc[0]) * ramp(m, TRUTH["ramp"]["cost"]) * np.exp(e[0])),
                    "rev_run_rate": float(mgmt_rev * np.exp(lr[0]) * ramp(m, TRUTH["ramp"]["rev"]) * np.exp(e[1]))})
    return out, {"cost_ratio": float(np.exp(lc[0])), "rev_ratio": float(np.exp(lr[0]))}
