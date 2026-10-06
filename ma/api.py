"""Public API (modular monolith). Blueprint services map to: target intelligence (deals, financials), diligence ingest (the data
room as a background job, documents one by one), entity graph (companies and same-as edges from customer matching), risk
extractor (contract and cyber parsers, concentration), synergy model (fitted to the deal database), system mapping (application
clusters and interfaces), integration planner (fast against staged ERP, as a background job) and the decision log.

    uvicorn ma.api:app          python -m core.jobs ma.api      # the worker
"""
import functools
import hashlib
import json
import time
import uuid

import numpy as np
from fastapi import Depends, Header
from fastapi.encoders import jsonable_encoder
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from core import audit, db, jobs
from core.app import Ctx, Problem, create_app, run

from . import engine as E, world as W

READ = {"deal:read", "jobs:read"}
WORK = READ | {"deal:write", "models:fit", "scenarios:run", "decisions:propose"}
PERMISSIONS = {"viewer": READ, "analyst": WORK, "deal_lead": WORK | {"data:generate", "decisions:approve", "audit:read"}}
app = create_app("ma-integration-intelligence", PERMISSIONS)
auth = app.state.auth
IdemKey = Header(None, alias="Idempotency-Key")
APPROACHES = ("fast", "staged")


def worker_ctx(job):
    return Ctx(job["tenant_id"], uuid.UUID(job["payload"]["actor_id"]), "worker", "system")


def money(v):
    return round(float(v))


def r3(v):
    return round(float(v), 3)


@functools.lru_cache(maxsize=4)
def room(seed):
    """The synthetic data room for a seed (the generator is deterministic). Its truth is read only to score the models."""
    return W.data_room(seed)


def deal_row(c, deal_id):
    d = c.execute("SELECT * FROM deal WHERE deal_id = %s", [deal_id]).fetchone()
    if not d:
        raise Problem(404, "deal_not_found")
    return d


def loaded(c, d):
    if not c.execute("SELECT 1 FROM company WHERE deal_id = %s AND status = 'target'", [d["deal_id"]]).fetchone():
        raise Problem(409, "no_data_room", "import the data room first: POST /v1/diligence/import")


def ltm(c, deal_id):
    """Revenue and EBITDA over the last four quarters."""
    rows = c.execute("""SELECT metric, sum(value) AS v FROM (SELECT * FROM financial_metric WHERE deal_id = %s AND period IN
                          (SELECT DISTINCT period FROM financial_metric WHERE deal_id = %s ORDER BY period DESC LIMIT 4)) x GROUP BY metric""", [deal_id, deal_id]).fetchall()
    return {r["metric"]: float(r["v"]) for r in rows}


def overlap_share(c, deal_id):
    row = c.execute("""SELECT coalesce(sum(t.revenue) FILTER (WHERE EXISTS (SELECT 1 FROM entity_edge e WHERE e.src = t.company_id)), 0) AS shared, sum(t.revenue) AS total
                         FROM company t WHERE t.deal_id = %s AND t.status = 'target_customer'""", [deal_id]).fetchone()
    return float(row["shared"]) / float(row["total"])


def features(c, d):
    f = ltm(c, d["deal_id"])
    return E.deal_features(d, f["revenue"], f["ebitda"], overlap_share(c, d["deal_id"])), f["revenue"]


def truth_features(d, f):
    return {**f, "overlap": room(d["seed"])["truth"]["customer_overlap_share"]} if d["seed"] is not None else None


def models(c):
    rows = {r["model_name"]: r for r in c.execute("SELECT DISTINCT ON (model_name) * FROM model_artifact ORDER BY model_name, created_at DESC")}
    if "synergy_realisation" not in rows or "erp_programmes" not in rows:
        raise Problem(409, "no_models", "fit the models first: POST /v1/synergies/model")
    return rows


def evidence(c, t, deal_id, subject_type, subject_id, document_id=None, locator=None, snippet=None, model_id=None):
    c.execute("INSERT INTO evidence (evidence_id, tenant_id, deal_id, subject_type, subject_id, document_id, locator, snippet, model_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
              [uuid.uuid7(), t, deal_id, subject_type, subject_id, document_id, locator, snippet, model_id])


# ---------------------------------------------------------------- target intelligence

class ManagementCase(BaseModel):
    cost_run_rate: float = Field(gt=0, description="announced run-rate cost synergies, $ a year, three years after close")
    revenue_run_rate: float = Field(ge=0, description="announced run-rate revenue synergies, $ of revenue a year")
    revenue_margin: float = Field(0.4, gt=0, le=1, description="EBITDA margin on the extra revenue")
    it_share: float = Field(0.35, ge=0, le=1, description="share of the cost synergies that needs the ERP consolidated first")


class DealIn(BaseModel):
    external_ref: str = Field(min_length=1, max_length=40)
    name: str = Field(max_length=120)
    target_name: str = Field(min_length=1, max_length=120)
    acquirer_revenue: float = Field(gt=0)
    price: float = Field(gt=0)
    standalone_value: float = Field(gt=0)
    relatedness: float = Field(ge=0, le=1, description="how close the two businesses are, 0 to 1")
    proposed_approach: str = Field("fast", pattern="^(fast|staged)$")
    management_case: ManagementCase


@app.post("/v1/deals", status_code=201, tags=["target intelligence"], summary="Open a deal: the target, the price, what the target is worth alone, and management's synergy case")
def create_deal(body: DealIn, ctx: Ctx = Depends(auth("deal:write")), idem: str | None = IdemKey):
    if body.price <= body.standalone_value:
        raise Problem(422, "no_premium", "the price is not above the standalone value: nothing for synergies to pay for")

    def work(c):
        if c.execute("SELECT 1 FROM deal WHERE external_ref = %s", [body.external_ref]).fetchone():
            raise Problem(409, "deal_exists")
        deal_id = uuid.uuid5(ctx.tenant_id, f"deal-{body.external_ref}")
        c.execute("""INSERT INTO deal (deal_id, tenant_id, external_ref, name, metadata, target_name, price, standalone_value, acquirer_revenue, management_case, relatedness)
                     VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                  [deal_id, ctx.tenant_id, body.external_ref, body.name, Jsonb({"proposed_approach": body.proposed_approach}), body.target_name, body.price, body.standalone_value,
                   body.acquirer_revenue, Jsonb(body.management_case.model_dump()), body.relatedness])
        audit.record(c, ctx, "deal.created", "deal", deal_id, {"target": body.target_name, "price": body.price})
        return 201, {"deal_id": deal_id, "premium": money(body.price - body.standalone_value)}
    return run(ctx, idem, body, work)


@app.get("/v1/deals/{deal_id}", tags=["target intelligence"], summary="(+) The target: financials, customers, documents, systems, and the deal's terms")
def get_deal(deal_id: uuid.UUID, ctx: Ctx = Depends(auth("deal:read"))):
    with db.tx(ctx.tenant_id) as c:
        d = deal_row(c, deal_id)
        loaded(c, d)
        fin = c.execute("SELECT period, metric, value FROM financial_metric WHERE deal_id = %s ORDER BY period", [deal_id]).fetchall()
        n = c.execute("""SELECT (SELECT count(*) FROM company WHERE deal_id = %(d)s AND status = 'target_customer') AS customers,
                                (SELECT count(*) FROM company WHERE deal_id = %(d)s AND status = 'acquirer_customer') AS acquirer_customers,
                                (SELECT count(*) FROM document WHERE deal_id = %(d)s) AS documents,
                                (SELECT count(*) FROM system WHERE deal_id = %(d)s AND owner = 'target') AS target_systems,
                                (SELECT count(*) FROM system WHERE deal_id = %(d)s AND owner = 'acquirer') AS acquirer_systems,
                                (SELECT count(*) FROM comparable_deal) AS comparable_deals""", {"d": deal_id}).fetchone()
        l4 = ltm(c, deal_id)
    q = {}
    for r in fin:
        q.setdefault(r["period"], {})[r["metric"]] = money(r["value"])
    periods = sorted(q)
    growth = l4["revenue"] / sum(q[p]["revenue"] for p in periods[-8:-4]) - 1
    return jsonable_encoder({"deal_id": deal_id, "name": d["name"], "target": d["target_name"], "price": money(d["price"]), "standalone_value": money(d["standalone_value"]),
                             "premium": money(d["price"] - d["standalone_value"]), "management_case": d["management_case"], "proposed_approach": d["metadata"].get("proposed_approach"),
                             "ltm": {"revenue": money(l4["revenue"]), "ebitda": money(l4["ebitda"]), "ebitda_margin": r3(l4["ebitda"] / l4["revenue"]), "growth": r3(growth),
                                     "gross_margin": r3(l4["gross_profit"] / l4["revenue"])},
                             "quarters": [{"period": p, **q[p]} for p in periods], "counts": n})


# ---------------------------------------------------------------- diligence ingest

class Generate(BaseModel):
    seed: int = Field(31, ge=0, le=10 ** 6)


class Doc(BaseModel):
    ref: str = Field(min_length=1, max_length=40)
    kind: str = Field(pattern="^(customer_contract|side_letter|cyber_assessment)$")
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=20, max_length=200_000)
    customer: str | None = Field(None, max_length=120, description="the counterparty, matched to the target's customers by name")


class ImportIn(BaseModel):
    deal_id: uuid.UUID
    generate: Generate | None = None
    documents: list[Doc] | None = Field(None, max_length=200)


@app.post("/v1/diligence/import", status_code=202, tags=["diligence ingest"], summary="Load the data room. `generate` loads the synthetic one (financials, both customer masters, contracts, the cyber assessment, both application portfolios, the deal database) in a background job; `documents` adds documents one by one, each parsed on arrival")
def import_diligence(body: ImportIn, ctx: Ctx = Depends(auth("deal:write")), idem: str | None = IdemKey):
    if (body.generate is None) == (body.documents is None):
        raise Problem(422, "generate_or_documents", "send exactly one of generate or documents")
    if body.generate and "data:generate" not in PERMISSIONS[ctx.role]:
        raise Problem(403, "forbidden", "loading a synthetic data room needs the deal lead")

    def work(c):
        d = deal_row(c, body.deal_id)
        if body.generate:
            if c.execute("SELECT 1 FROM company WHERE deal_id = %s AND status = 'target'", [d["deal_id"]]).fetchone():
                raise Problem(409, "data_room_loaded")
            return 202, jobs.enqueue(c, ctx, "diligence.generate", {"deal_id": str(d["deal_id"]), "seed": body.generate.seed})
        loaded(c, d)
        out = []
        for doc in body.documents:
            if c.execute("SELECT 1 FROM document WHERE deal_id = %s AND ref = %s", [d["deal_id"], doc.ref]).fetchone():
                out.append({"ref": doc.ref, "status": "refused", "why": "a document with this reference is already in the data room"})
                continue
            cust = None
            if doc.customer:
                cust = c.execute("SELECT company_id, name, revenue FROM company WHERE deal_id = %s AND status = 'target_customer' AND lower(name) = lower(%s)", [d["deal_id"], doc.customer]).fetchone()
                if not cust:
                    out.append({"ref": doc.ref, "status": "refused", "why": f"no customer called '{doc.customer}' in the target's customer master"})
                    continue
            did = add_document(c, ctx.tenant_id, d["deal_id"], doc.ref, doc.kind, doc.title, doc.text, cust["company_id"] if cust else None)
            risks = extract(c, ctx.tenant_id, d["deal_id"], did, doc.ref, doc.kind, doc.text, cust)
            out.append({"ref": doc.ref, "status": "parsed", "risks": risks})
        audit.record(c, ctx, "diligence.documents_added", "deal", d["deal_id"], {"parsed": sum(o["status"] == "parsed" for o in out), "refused": sum(o["status"] == "refused" for o in out)})
        return 201, {"deal_id": d["deal_id"], "documents": out}
    return run(ctx, idem, body, work)


def add_document(c, t, deal_id, ref, kind, title, text, company_id):
    did = uuid.uuid5(deal_id, f"doc-{ref}")
    c.execute("INSERT INTO document (document_id, tenant_id, deal_id, ref, kind, title, body, sha256, company_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
              [did, t, deal_id, ref, kind, title, text, hashlib.sha256(text.encode()).hexdigest(), company_id])
    return did


def extract(c, t, deal_id, did, ref, kind, text, cust):
    """Run the parser for a document's kind and write a risk, with its evidence, for every clause or finding it reports."""
    out = []
    if kind in ("customer_contract", "side_letter"):
        total = float(c.execute("SELECT sum(revenue) AS s FROM company WHERE deal_id = %s AND status = 'target_customer'", [deal_id]).fetchone()["s"])
        rev = float(cust["revenue"]) if cust else 0.0
        hits = E.parse_contract(text if kind == "customer_contract" else "\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(x for x in text.replace("\n", " ").split(". ") if x)))
        for key, label in (("coc", "change_of_control"), ("tfc", "termination_for_convenience")):
            for no, sentence, why in hits[key]:
                rid = uuid.uuid7()
                sev = "high" if label == "change_of_control" and rev / total >= 0.02 else "medium"
                conf = 0.9 if why.startswith("names") or why.startswith("terminable for convenience") else 0.75
                title = f"{'Change of control' if label == 'change_of_control' else 'Termination for convenience'}: {cust['name'] if cust else ref}"
                c.execute("INSERT INTO risk (risk_id, tenant_id, deal_id, kind, severity, title, exposure_usd, confidence, detail) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                          [rid, t, deal_id, label, sev, title, rev, conf, Jsonb({"rule": why, "revenue_share": rev / total if total else 0, "document": ref, "clause": no, "customer": cust["name"] if cust else None})])
                evidence(c, t, deal_id, "risk", rid, did, f"{ref} clause {no}", sentence)
                out.append({"kind": label, "severity": sev, "clause": no, "rule": why, "revenue": money(rev)})
    elif kind == "cyber_assessment":
        for f in E.parse_cyber(text):
            if not f["open"] or f["severity"] not in ("Critical", "High"):
                continue
            rid = uuid.uuid7()
            sev = "high" if f["severity"] == "Critical" or f["internet_facing"] else "medium"
            c.execute("INSERT INTO risk (risk_id, tenant_id, deal_id, kind, severity, title, exposure_usd, confidence, detail) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                      [rid, t, deal_id, "cyber", sev, f"{f['id']} {f['severity']}: {f['finding']}", E.REMEDIATION[f["severity"]], 0.95, Jsonb(f)])
            evidence(c, t, deal_id, "risk", rid, did, f"{ref} {f['id']}", f["finding"])
            out.append({"kind": "cyber", "severity": sev, "finding": f["id"]})
    return out


@jobs.handler("diligence.generate")
def generate_job(c, job):
    t, p = job["tenant_id"], job["payload"]
    deal_id, seed = uuid.UUID(p["deal_id"]), p["seed"]
    d = deal_row(c, deal_id)
    r = room(seed)
    c.execute("UPDATE deal SET seed = %s WHERE deal_id = %s", [seed, deal_id])
    cid = lambda ref: uuid.uuid5(deal_id, f"co-{ref}")     # noqa: E731
    db.load(c, "company", ["company_id", "tenant_id", "deal_id", "status", "attributes", "name", "revenue"],
            [(cid("acquirer"), t, deal_id, "acquirer", Jsonb({"revenue": float(d["acquirer_revenue"])}), W.ACQUIRER, None),
             (cid("target"), t, deal_id, "target", Jsonb({}), d["target_name"], None)]
            + [(cid(x["ref"]), t, deal_id, "target_customer", Jsonb({"ref": x["ref"]}), x["name"], x["revenue"]) for x in r["customers"]]
            + [(cid(f"AC-{j + 1:05d}"), t, deal_id, "acquirer_customer", Jsonb({"ref": f"AC-{j + 1:05d}"}), n, None) for j, n in enumerate(r["acquirer_customers"])])
    db.load(c, "financial_metric", ["tenant_id", "deal_id", "period", "metric", "value"], [(t, deal_id, f["period"], f["metric"], f["value"]) for f in r["financials"]])
    # the entity graph: the target's customers the acquirer already serves
    match = E.match_customers([x["name"] for x in r["customers"]], r["acquirer_customers"])
    db.load(c, "entity_edge", ["tenant_id", "deal_id", "src", "dst", "relation", "confidence", "method"],
            [(t, deal_id, cid(r["customers"][i]["ref"]), cid(f"AC-{j + 1:05d}"), "same_as", s, m) for i, (j, s, m) in match.items()])
    # application portfolios, their interfaces, and clusters of overlapping capability
    labels = E.cluster_apps(r["apps"])
    sid = lambda ref: uuid.uuid5(deal_id, f"sys-{ref}")     # noqa: E731
    db.load(c, "system", ["system_id", "tenant_id", "deal_id", "owner", "ref", "product", "tags", "annual_cost", "users", "cluster"],
            [(sid(a["ref"]), t, deal_id, a["company"], a["ref"], a["product"], a["tags"], a["annual_cost"], a["users"], int(labels[i])) for i, a in enumerate(r["apps"])])
    db.load(c, "dependency", ["tenant_id", "deal_id", "src", "dst", "kind"], [(t, deal_id, sid(x), sid(y), k) for x, y, k in r["edges"]])
    # the deal database (tenant-wide reference data)
    rows, _ = W.comparables(seed)
    c.execute("CREATE TEMP TABLE _cd (LIKE comparable_deal) ON COMMIT DROP")
    db.copy(c, "_cd", ["tenant_id", "ref", "months_since_close", "still_disclosing", "features", "reports", "erp"],
            [(t, x["ref"], x["months_since_close"], x["still_disclosing"], json.dumps({k: x[k] for k in ("related", "cost_pct", "rev_pct", "rel_size", "fast", "overlap")}),
              json.dumps(x["reports"]), json.dumps(x["erp"]) if x["erp"] else None) for x in rows])
    c.execute("INSERT INTO comparable_deal SELECT * FROM _cd ON CONFLICT DO NOTHING")
    c.execute("DROP TABLE _cd")
    # documents, parsed on arrival
    for doc in r["documents"]:
        cust = c.execute("SELECT company_id, name, revenue FROM company WHERE company_id = %s", [cid(doc["customer_ref"])]).fetchone() if doc["customer_ref"] else None
        did = add_document(c, t, deal_id, doc["ref"], doc["kind"], doc["title"], doc["text"], cust["company_id"] if cust else None)
        extract(c, t, deal_id, did, doc["ref"], doc["kind"], doc["text"], cust)
    # concentration and overlap, from the customer master
    revs = np.array([x["revenue"] for x in r["customers"]], float)
    conc = E.concentration(revs)
    top = r["customers"][0]
    rid = uuid.uuid7()
    c.execute("INSERT INTO risk (risk_id, tenant_id, deal_id, kind, severity, title, exposure_usd, confidence, detail) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
              [rid, t, deal_id, "concentration", "high" if conc["top1"] >= 0.10 or conc["top10"] >= 0.5 else "medium", f"Customer concentration: {top['name']} is {100 * conc['top1']:.1f}% of revenue",
               top["revenue"], 1.0, Jsonb(conc)])
    evidence(c, t, deal_id, "risk", rid, None, "customer master", f"{top['ref']} {top['name']}: ${top['revenue']:,} of ${revs.sum():,.0f}")
    share = float(revs[list(match)].sum() / revs.sum())
    rid = uuid.uuid7()
    c.execute("INSERT INTO risk (risk_id, tenant_id, deal_id, kind, severity, title, exposure_usd, confidence, detail) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
              [rid, t, deal_id, "customer_overlap", "medium" if share > 0.25 else "low", f"{100 * share:.0f}% of revenue is from customers the acquirer already serves",
               float(revs[list(match)].sum()), None, Jsonb({"share": share, "matched": len(match)})])
    evidence(c, t, deal_id, "risk", rid, None, "entity graph", f"{len(match)} same-as edges between the two customer masters")
    audit.record(c, worker_ctx(job), "diligence.generated", "deal", deal_id, {"seed": seed, "customers": len(r["customers"]), "documents": len(r["documents"]), "systems": len(r["apps"]), "comparables": len(rows)})
    return {"deal_id": deal_id, "customers": len(r["customers"]), "acquirer_customers": len(r["acquirer_customers"]), "documents": len(r["documents"]),
            "systems": len(r["apps"]), "comparable_deals": len(rows), "same_as_edges": len(match)}


# ---------------------------------------------------------------- risk extractor

@app.get("/v1/deals/{deal_id}/risks", tags=["risk extractor"], summary="Every risk with its evidence: concentration, change-of-control and convenience clauses, open cyber findings, the customer overlap; how the entity graph and the parsers did")
def risks(deal_id: uuid.UUID, ctx: Ctx = Depends(auth("deal:read"))):
    with db.tx(ctx.tenant_id) as c:
        d = deal_row(c, deal_id)
        loaded(c, d)
        rows = c.execute("""SELECT r.*, coalesce(json_agg(json_build_object('locator', e.locator, 'snippet', e.snippet, 'document', doc.ref)) FILTER (WHERE e.evidence_id IS NOT NULL), '[]') AS evidence
                              FROM risk r LEFT JOIN evidence e ON e.subject_id = r.risk_id LEFT JOIN document doc USING (document_id)
                             WHERE r.deal_id = %s GROUP BY r.risk_id ORDER BY r.kind, r.exposure_usd DESC NULLS LAST""", [deal_id]).fetchall()
        edges = c.execute("""SELECT t.name AS target_name, a.name AS acquirer_name, t.revenue, e.confidence, e.method FROM entity_edge e JOIN company t ON t.company_id = e.src JOIN company a ON a.company_id = e.dst
                              WHERE e.deal_id = %s ORDER BY t.revenue DESC""", [deal_id]).fetchall()
        exact = c.execute("""SELECT count(*) AS n, coalesce(sum(t.revenue), 0) AS rev FROM company t WHERE t.deal_id = %(d)s AND t.status = 'target_customer'
                               AND EXISTS (SELECT 1 FROM company a WHERE a.deal_id = %(d)s AND a.status = 'acquirer_customer' AND a.name = t.name)""", {"d": deal_id}).fetchone()
        total = float(c.execute("SELECT sum(revenue) AS s FROM company WHERE deal_id = %s AND status = 'target_customer'", [deal_id]).fetchone()["s"])
        contracts = c.execute("SELECT count(*) AS n FROM document WHERE deal_id = %s AND kind = 'customer_contract'", [deal_id]).fetchone()["n"]
    by = lambda k: [x for x in rows if x["kind"] == k]     # noqa: E731
    coc, cyber = by("change_of_control"), by("cyber")
    coc_by = {x["detail"].get("customer") or x["detail"]["document"]: x for x in coc}        # a customer with a clause in two documents counts once
    coc_rev = sum(float(x["exposure_usd"]) for x in coc_by.values())
    out = {"deal_id": deal_id, "risks": [{"risk_id": x["risk_id"], "kind": x["kind"], "severity": x["severity"], "title": x["title"], "exposure_usd": money(x["exposure_usd"]) if x["exposure_usd"] is not None else None,
                                          "confidence": x["confidence"], "detail": x["detail"], "evidence": x["evidence"]} for x in rows],
           "summary": {"concentration": by("concentration")[0]["detail"] if by("concentration") else None, "contracts_read": contracts,
                       "change_of_control": {"customers": len(coc_by), "documents": len({x["detail"]["document"] for x in coc}), "revenue": money(coc_rev), "share": r3(coc_rev / total)},
                       "termination_for_convenience": len(by("termination_for_convenience")),
                       "cyber_open_critical_high": len(cyber), "cyber_internet_facing": sum(bool(x["detail"]["internet_facing"]) for x in cyber),
                       "cyber_remediation_estimate": money(sum(float(x["exposure_usd"]) for x in cyber))},
           "entity_graph": {"same_as_edges": len(edges), "overlap_share": r3(sum(float(e["revenue"]) for e in edges) / total), "by_method": {m: sum(e["method"] == m for e in edges) for m in ("normalised name", "fuzzy")},
                            "exact_name_baseline": {"matches": exact["n"], "overlap_share": r3(float(exact["rev"]) / total)},
                            "examples": [{"target": e["target_name"], "acquirer": e["acquirer_name"], "confidence": e["confidence"], "method": e["method"]} for e in edges if e["target_name"] != e["acquirer_name"]][:6]},
           "how": "Contracts and side letters: sentence rules for change-of-control and convenience clauses, written by hand (no language model). The cyber assessment: one regular expression per finding line. Customers: names normalised (case, punctuation, legal suffixes, common abbreviations, a leading 'The') and matched exactly, else to the most similar name sharing the first word if similarity is at least 0.9."}
    if d["seed"] is not None:
        tr = room(d["seed"])["truth"]
        names = {x["name"]: x for x in room(d["seed"])["customers"]}
        tp = sum(names.get(e["target_name"], {}).get("true_overlap", False) for e in edges)
        true_n = sum(x["true_overlap"] for x in room(d["seed"])["customers"])
        truth_coc = [x for x in tr["clauses"] if x["coc"] is not None]
        found = {x["detail"]["document"] for x in coc} & {x["doc"] for x in tr["clauses"]}      # the generated contracts only
        out["simulation_truth"] = {"what": "what the generator put in the data room; the parsers and the matcher never see it",
                                   "overlap_share": r3(tr["customer_overlap_share"]), "shared_customers": true_n, "sister_entities": tr["sisters"],
                                   "matcher": {"precision": r3(tp / max(1, len(edges))), "recall": r3(tp / max(1, true_n))},
                                   "change_of_control": {"contracts": len(truth_coc), "found": sum(x["doc"] in found for x in truth_coc), "false_alarms": len(found - {x["doc"] for x in truth_coc}),
                                                         "missed_wording": sorted({W.COC[x["coc"]] for x in truth_coc if x["doc"] not in found})},
                                   "cyber_open_critical_high": sum(f["open"] and f["severity"] in ("Critical", "High") for f in tr["findings"])}
    return jsonable_encoder(out)


# ---------------------------------------------------------------- synergy model

class ModelIn(BaseModel):
    deal_id: uuid.UUID
    draws: int = Field(20000, ge=1000, le=100000)


GRID = np.linspace(0, 2.0, 41)       # realisation ratio, for the histograms


def hist(a):
    h, _ = np.histogram(np.clip(a, 0, 2.0 - 1e-9), GRID)
    return (h / len(a)).round(4).tolist()


@app.post("/v1/synergies/model", status_code=201, tags=["synergy model"], summary="Fit synergy realisation and ERP programme models to the deal database, store them as versioned models, and give this deal's synergies as distributions for both integration approaches")
def model_synergies(body: ModelIn, ctx: Ctx = Depends(auth("models:fit")), idem: str | None = IdemKey):
    def work(c):
        d = deal_row(c, body.deal_id)
        loaded(c, d)
        rows = [{"ref": x["ref"], "months_since_close": x["months_since_close"], "still_disclosing": x["still_disclosing"], **x["features"], "reports": x["reports"], "erp": x["erp"]}
                for x in c.execute("SELECT * FROM comparable_deal ORDER BY ref")]
        if len(rows) < 30:
            raise Problem(409, "too_few_comparables", f"{len(rows)} comparable deals; the fit needs 30")
        t0 = time.perf_counter()
        mod, erp = E.fit_synergy(rows), E.fit_erp(rows)
        secs = time.perf_counter() - t0
        ver = f"{E.VERSION}-{int(time.time() * 1000)}"
        ids = {}
        for name, params, metrics, inputs in (("synergy_realisation", mod, {"deals": mod["n"], "silent": mod["silent"], "young": mod["young"], "fit_seconds": round(secs, 2)}, ["related", "announced share", "relative size", "fast", "customer overlap", "months since close"]),
                                              ("erp_programmes", erp, {"programmes": erp["programmes"], "running": erp["running"]}, ["planned weeks", "approach"])):
            ids[name] = uuid.uuid7()
            c.execute("INSERT INTO model_artifact (model_id, tenant_id, model_name, version, artifact_uri, feature_schema, metrics, params) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                      [ids[name], ctx.tenant_id, name, ver, f"db://model_artifact/{name}/{ver}", Jsonb({"inputs": inputs}), Jsonb(metrics), Jsonb(params)])
        f, revenue = features(c, d)
        tf = truth_features(d, f)
        mc = d["management_case"]
        dist = {}
        for ap in APPROACHES:
            rc, rr = E.predict(mod, f, ap == "fast", body.draws, np.random.default_rng(11))
            dist[ap] = {"cost": {"ratio": E.q(rc), "run_rate": [money(x * mc["cost_run_rate"]) for x in E.q(rc)], "mean": r3(rc.mean()), "hist": hist(rc)},
                        "revenue": {"ratio": E.q(rr), "run_rate": [money(x * mc["revenue_run_rate"]) for x in E.q(rr)], "mean": r3(rr.mean()), "hist": hist(rr)},
                        "p_cost_meets_case": r3((rc >= 1).mean()), "p_revenue_meets_case": r3((rr >= 1).mean())}
            for kind, a in (("cost", rc), ("revenue", rr)):
                base = mc["cost_run_rate"] if kind == "cost" else mc["revenue_run_rate"]
                p10, p50, p90 = (x * base for x in E.q(a))
                c.execute("INSERT INTO synergy (synergy_id, tenant_id, deal_id, kind, basis, approach, p10, p50, p90, model_id) VALUES (%s,%s,%s,%s,'model',%s,%s,%s,%s,%s)",
                          [uuid.uuid7(), ctx.tenant_id, d["deal_id"], kind, ap, p10, p50, p90, ids["synergy_realisation"]])
        for kind, base, comp in (("cost", mc["cost_run_rate"], mod["comps"]["cost"]), ("revenue", mc["revenue_run_rate"], mod["comps"]["rev"])):
            c.execute("INSERT INTO synergy (synergy_id, tenant_id, deal_id, kind, basis, p50) VALUES (%s,%s,%s,%s,'management_case',%s), (%s,%s,%s,%s,'comparables',%s)",
                      [uuid.uuid7(), ctx.tenant_id, d["deal_id"], kind, base, uuid.uuid7(), ctx.tenant_id, d["deal_id"], kind, base * comp])
        audit.record(c, ctx, "models.fitted", "model_artifact", ver, {"deals": mod["n"], "programmes": erp["programmes"]})
        coef = lambda m: dict(zip(["intercept"] + E.FEATURES, [r3(b) for b in m["beta"]]))     # noqa: E731
        T = W.TRUTH
        out = {"version": ver, "fit_seconds": round(secs, 2), "features": {k: r3(v) for k, v in f.items()}, "management_case": mc, "proposed_approach": d["metadata"].get("proposed_approach"),
               "database": {"deals": mod["n"], "young": mod["young"], "went_quiet": mod["silent"], "reports": mod["cost"]["reports"], "with_revenue_figures": mod["rev"]["n"]},
               "fit": {"cost": {"coef": coef(mod["cost"]), "ramp_k": r3(mod["cost"]["k"]), "deal_sd": r3(mod["cost"]["sd"]), "report_sd": r3(mod["cost"]["report_sd"])},
                       "revenue": {"coef": coef(mod["rev"]), "ramp_k": r3(mod["rev"]["k"]), "deal_sd": r3(mod["rev"]["sd"]), "report_sd": r3(mod["rev"]["report_sd"])}, "rho": r3(mod["rho"])},
               "baselines": {"management_case": 1.0, "comparables": {"cost": r3(mod["comps"]["cost"]), "revenue": r3(mod["comps"]["rev"]), "deals": mod["comps"]["n"],
                                                                      "what": "the average latest disclosure of deals at least three years old that still disclose"},
                             "latest_as_is": {"cost": r3(mod["naive_latest"]["cost"]), "revenue": r3(mod["naive_latest"]["rev"])}},
               "distributions": dist, "grid": GRID.round(2).tolist(),
               "erp": {"programmes": erp["programmes"], "running": erp["running"], "lateness_median": r3(np.exp(erp["bias_mu"])), "lateness_finished_only": r3(np.exp(erp["finished_only_mu"])),
                       "incident_rate": {a: r3(erp["incident"][a]["a"] / (erp["incident"][a]["a"] + erp["incident"][a]["b"])) for a in APPROACHES},
                       "incidents": {a: [erp["incident"][a]["incidents"], erp["incident"][a]["programmes"]] for a in APPROACHES}, "incident_weeks_median": r3(np.exp(erp["weeks_mu"]))},
               "how": "Every deal in the database counts at each disclosure: log(reported / announced) = features . beta + k log(months / 36) + the deal's own effect + reporting noise, by maximum likelihood over the first-anniversary and latest reports. Deals that went quiet after their first year still contribute that year. Draws for this deal add coefficient uncertainty and the deal effect. ERP lateness: lognormal, with programmes still running as right-censored."}
        if tf:
            out["simulation_truth"] = {"what": "the process the database was generated from; the fit never sees it", "overlap": r3(tf["overlap"]),
                                       "cost": {"related": T["cost"]["related"], "fast": T["cost"]["fast"], "ramp_k": T["ramp"]["cost"], "deal_sd": T["cost"]["sd"],
                                                "note": f"announced share enters as a kink above {T['cost']['aggr_kink']:.0%}, relative size above {T['cost']['size_kink']:.0%}; the model is linear in both"},
                                       "revenue": {"related": T["rev"]["related"], "fast": T["rev"]["fast"], "overlap": T["rev"]["overlap"], "ramp_k": T["ramp"]["rev"], "deal_sd": T["rev"]["sd"]},
                                       "report_sd": T["meas_sd"], "lateness_median": r3(np.exp(T["erp"]["bias_mu"])), "incident_rate": T["erp"]["p_incident"], "incident_weeks_median": r3(np.exp(T["erp"]["weeks_mu"]))}
            for ap in APPROACHES:
                tc, trr = E.truth_draws(tf, ap == "fast", body.draws, np.random.default_rng(12))
                out["simulation_truth"][ap] = {"cost": E.q(tc), "revenue": E.q(trr), "cost_hist": hist(tc), "revenue_hist": hist(trr)}
        return 201, out
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- system mapping

@app.get("/v1/systems/dependencies", tags=["system mapping"], summary="Both application portfolios, clusters of overlapping capability, what could be retired and what it saves, and the interfaces the target's ERP carries")
def dependencies(deal_id: uuid.UUID, ctx: Ctx = Depends(auth("deal:read"))):
    with db.tx(ctx.tenant_id) as c:
        d = deal_row(c, deal_id)
        loaded(c, d)
        sys_ = c.execute("SELECT * FROM system WHERE deal_id = %s ORDER BY owner, ref", [deal_id]).fetchall()
        deps = c.execute("""SELECT s.ref AS src, s.owner, t.ref AS dst, t.product AS dst_product FROM dependency x JOIN system s ON s.system_id = x.src JOIN system t ON t.system_id = x.dst
                             WHERE x.deal_id = %s""", [deal_id]).fetchall()
    by_cluster = {}
    for s in sys_:
        by_cluster.setdefault(s["cluster"], []).append(s)
    erp_target = target_erp(sys_, deps)
    overlaps, savings = [], 0.0
    for k, members in sorted(by_cluster.items()):
        acq, tgt = [m for m in members if m["owner"] == "acquirer"], [m for m in members if m["owner"] == "target"]
        if not acq or not tgt:
            continue
        shared = sorted(set.intersection(*[set(m["tags"]) for m in members])) or sorted(set(acq[0]["tags"]) & set(tgt[0]["tags"]))
        retire = tgt            # the acquirer's platform survives: the integration thesis is to move the target onto it
        save = sum(float(m["annual_cost"]) for m in retire)
        savings += save
        overlaps.append({"cluster": k, "acquirer": [m["product"] for m in acq], "target": [m["product"] for m in tgt], "shared_tags": shared[:6],
                         "retire": [m["ref"] for m in retire], "annual_saving": money(save), "same_product": any(a["product"] == t["product"] for a in acq for t in tgt)})
    out = {"deal_id": deal_id, "systems": {o: [{"ref": s["ref"], "product": s["product"], "tags": s["tags"], "annual_cost": money(s["annual_cost"]), "users": s["users"], "cluster": s["cluster"]} for s in sys_ if s["owner"] == o] for o in ("acquirer", "target")},
           "overlaps": overlaps, "it_savings_identified": money(savings), "it_synergy_in_case": money(d["management_case"]["cost_run_rate"] * d["management_case"]["it_share"]),
           "target_erp": erp_target, "same_product_baseline": sum(o["same_product"] for o in overlaps),
           "dependencies": [{"src": x["src"], "dst": x["dst"], "owner": x["owner"]} for x in deps],
           "how": "Average-linkage clusters on the Jaccard distance between each application's functional tags, cut at 0.75; a cluster holding applications of both companies is an overlap. The baseline is matching on product name."}
    if d["seed"] is not None:
        apps = room(d["seed"])["apps"]
        cap = {a["ref"]: a["true_capability"] for a in apps}
        pairs = {(a["ref"], b["ref"]) for a in sys_ for b in sys_ if a["owner"] == "acquirer" and b["owner"] == "target"}
        truth = {p for p in pairs if cap[p[0]] == cap[p[1]]}
        found = {(a["ref"], b["ref"]) for a in sys_ for b in sys_ if a["owner"] == "acquirer" and b["owner"] == "target" and a["cluster"] == b["cluster"]}
        prod = {(a["ref"], b["ref"]) for a in sys_ for b in sys_ if a["owner"] == "acquirer" and b["owner"] == "target" and a["product"] == b["product"]}
        out["simulation_truth"] = {"what": "each application's true capability, which the tags only hint at", "overlapping_pairs": len(truth),
                                   "clusters": {"precision": r3(len(found & truth) / max(1, len(found))), "recall": r3(len(found & truth) / max(1, len(truth)))},
                                   "same_product": {"precision": r3(len(prod & truth) / max(1, len(prod))), "recall": r3(len(prod & truth) / max(1, len(truth)))}}
    return jsonable_encoder(out)


def target_erp(sys_, deps):
    """The target's ERP: its system with the most interfaces, and the systems those interfaces reach."""
    tsys = {s["ref"]: s for s in sys_ if s["owner"] == "target"}
    count = {}
    for x in deps:
        if x["owner"] == "target":
            count[x["src"]] = count.get(x["src"], 0) + 1
    if not count:
        return None
    erp = max(count, key=count.get)
    return {"ref": erp, "product": tsys[erp]["product"], "interfaces": [{"ref": x["dst"], "product": x["dst_product"], "users": tsys[x["dst"]]["users"]} for x in deps if x["src"] == erp]}


# ---------------------------------------------------------------- integration planner

class ScenarioIn(BaseModel):
    deal_id: uuid.UUID
    draws: int = Field(10000, ge=1000, le=50000)
    seed: int = Field(2026, ge=0, le=10 ** 6)


@app.post("/v1/integration/scenario", status_code=202, tags=["integration planner"], summary="Fast against staged ERP integration in a background job: the plan for each, the team size chosen by enumeration, and the value of each as a distribution against the premium")
def scenario(body: ScenarioIn, ctx: Ctx = Depends(auth("scenarios:run")), idem: str | None = IdemKey):
    def work(c):
        d = deal_row(c, body.deal_id)
        loaded(c, d)
        m = models(c)
        sid = uuid.uuid7()
        c.execute("INSERT INTO integration_scenario (scenario_id, tenant_id, deal_id, status, params, created_by) VALUES (%s,%s,%s,'queued',%s,%s)",
                  [sid, ctx.tenant_id, d["deal_id"], Jsonb({**body.model_dump(mode="json"), "models": {k: v["version"] for k, v in m.items()}}), ctx.actor_id])
        return 202, {**jobs.enqueue(c, ctx, "integration.scenario", {"scenario_id": str(sid), **body.model_dump(mode="json")}), "scenario_id": sid}
    return run(ctx, idem, body, work)


def summary_of(v, premium):
    return {"mean": money(v.mean()), "p10": money(np.quantile(v, 0.1)), "p50": money(np.median(v)), "p90": money(np.quantile(v, 0.9)), "p_covers_premium": r3((v >= premium).mean())}


@jobs.handler("integration.scenario")
def scenario_job(c, job):
    p = job["payload"]
    sid, t = uuid.UUID(p["scenario_id"]), job["tenant_id"]
    d = deal_row(c, uuid.UUID(p["deal_id"]))
    m = models(c)
    mod, erp = m["synergy_realisation"]["params"], m["erp_programmes"]["params"]
    sys_ = c.execute("SELECT * FROM system WHERE deal_id = %s", [d["deal_id"]]).fetchall()
    deps = c.execute("""SELECT s.ref AS src, s.owner, t.ref AS dst, t.product AS dst_product FROM dependency x JOIN system s ON s.system_id = x.src JOIN system t ON t.system_id = x.dst
                         WHERE x.deal_id = %s""", [d["deal_id"]]).fetchall()
    te = target_erp(sys_, deps)
    ws = E.workstreams([(x["ref"], x["product"], x["users"]) for x in te["interfaces"]])
    f, revenue = features(c, d)
    tf = truth_features(d, f)
    deal = {"management_case": d["management_case"]}
    premium = float(d["price"] - d["standalone_value"])
    t0 = time.perf_counter()
    opts, values, edges = {}, {}, np.linspace(0, 700e6, 36)
    for ap in APPROACHES:
        lanes, scores = E.best_lanes(ws, mod, erp, f, ap, deal, revenue)
        o = E.evaluate_option(ws, mod, erp, f, ap, lanes, p["draws"], p["seed"], deal, revenue, truth_f=tf)
        s, pl = o["sim"], o["plan"]
        mcase = E.management_case(ws, mod, ap, lanes, deal, revenue)
        values[ap] = s["value"]
        opts[ap] = {"lanes": lanes, "lane_scores": {str(k): money(v) for k, v in scores.items()}, "planned_weeks": round(pl["done"], 1), "waves": pl["waves"],
                    "done_weeks": [round(float(x), 1) for x in np.quantile(s["done_weeks"], [0.5, 0.9])], "p_disruption": r3(s["disrupted"].mean()),
                    "expected_disruption_loss": money(s["loss"].mean()), "integration_cost": money(s["cost"].mean()), "value": summary_of(s["value"], premium),
                    "hist": (np.histogram(np.clip(s["value"], 0, 700e6 - 1), edges)[0] / len(s["value"])).round(4).tolist(),
                    "management_case_value": money(mcase["value"][0]), "cutovers": [[round(w, 1), share] for w, share in pl["cutovers"]],
                    "tasks": [{"wave": w, "key": k, "name": n, "start": round(a, 1), "finish": round(b, 1)} for w, k, n, a, b in pl["tasks"]]}
        if tf:
            tr = o["truth"]
            opts[ap]["simulation_truth"] = {**summary_of(tr["value"], premium), "done_weeks": round(float(np.median(tr["done_weeks"])), 1), "p_disruption": r3(tr["disrupted"].mean()),
                                            "hist": (np.histogram(np.clip(tr["value"], 0, 700e6 - 1), edges)[0] / len(tr["value"])).round(4).tolist()}
        db.load(c, "integration_workstream", ["tenant_id", "scenario_id", "approach", "wave", "key", "name", "planned_start_week", "planned_finish_week"],
                [(t, sid, ap, w, k, n, a, b) for w, k, n, a, b in pl["tasks"]])
        ms = [(f"Cutover {i + 1}" if ap == "staged" else "Cutover", wk, wk * float(np.exp(erp["bias_mu"])), wk * float(np.exp(erp["bias_mu"] + 1.2816 * erp["bias_sd"]))) for i, (wk, _) in enumerate(pl["cutovers"])]
        ms.append(("Legacy ERP decommissioned", pl["done"], float(np.median(s["done_weeks"])), float(np.quantile(s["done_weeks"], 0.9))))
        db.load(c, "milestone", ["tenant_id", "scenario_id", "approach", "name", "planned_week", "p50_week", "p90_week"], [(t, sid, ap, *x) for x in ms])
    diff = values["fast"] - values["staged"]
    best = max(APPROACHES, key=lambda a: opts[a]["value"]["mean"])
    spread = float(np.quantile(values[best], 0.9) - np.quantile(values[best], 0.1))
    metrics = {"options": opts, "premium": money(premium), "value_edges": edges.tolist(), "seconds": round(time.perf_counter() - t0, 1),
               "comparison": {"recommended": best, "fast_minus_staged_mean": money(diff.mean()), "p_fast_beats_staged": r3((diff > 0).mean()),
                              "difference_vs_spread": r3(abs(diff.mean()) / spread), "close_call": bool(abs(diff.mean()) < 0.05 * spread)},
               "target_erp": {"product": te["product"], "interfaces": len(te["interfaces"])},
               "assumptions": {**E.COSTS, "discount": E.DISCOUNT, "multiple": E.MULTIPLE, "horizon_weeks": E.WEEKS}}
    if tf:
        metrics["comparison"]["truth_recommends"] = max(APPROACHES, key=lambda a: opts[a]["simulation_truth"]["mean"])
        metrics["comparison"]["truth_fast_minus_staged_mean"] = opts["fast"]["simulation_truth"]["mean"] - opts["staged"]["simulation_truth"]["mean"]
    metrics = jsonable_encoder(metrics)
    c.execute("UPDATE integration_scenario SET status = 'completed', metrics = %s WHERE scenario_id = %s", [Jsonb(metrics), sid])
    audit.record(c, worker_ctx(job), "integration.scenario_completed", "integration_scenario", sid, {"recommended": best, "fast_mean": opts["fast"]["value"]["mean"], "staged_mean": opts["staged"]["value"]["mean"]})
    return {"scenario_id": sid, **metrics}


# ---------------------------------------------------------------- decision log

class DecisionIn(BaseModel):
    deal_id: uuid.UUID
    scenario_id: uuid.UUID
    decision_type: str = Field("erp_integration_approach", pattern="^erp_integration_approach$")
    choice: str = Field(pattern="^(fast|staged)$")
    rationale: str = Field(min_length=10, max_length=2000)


@app.post("/v1/decisions", status_code=201, tags=["decision log"], summary="Propose a decision with its evidence, expected impact and policy checks; it takes effect only when a second person approves it")
def propose(body: DecisionIn, ctx: Ctx = Depends(auth("decisions:propose")), idem: str | None = IdemKey):
    def work(c):
        d = deal_row(c, body.deal_id)
        s = c.execute("SELECT * FROM integration_scenario WHERE scenario_id = %s AND deal_id = %s", [body.scenario_id, d["deal_id"]]).fetchone()
        if not s or s["status"] != "completed":
            raise Problem(409, "scenario_not_ready")
        m, opt = s["metrics"], s["metrics"]["options"][body.choice]
        policy = {"four_eyes": "pending", "follows_recommendation": body.choice == m["comparison"]["recommended"], "covers_premium_p50": opt["value"]["p50"] >= m["premium"]}
        if not policy["follows_recommendation"] and not m["comparison"]["close_call"] and len(body.rationale) < 80:
            raise Problem(422, "override_needs_reason", "choosing the option with the lower expected value needs a rationale of at least 80 characters")
        did = uuid.uuid7()
        expected = {"value": opt["value"], "done_weeks": opt["done_weeks"], "p_disruption": opt["p_disruption"], "lanes": opt["lanes"], "management_case_value": opt["management_case_value"], "premium": m["premium"]}
        c.execute("""INSERT INTO decision (decision_id, tenant_id, deal_id, decision_type, choice, scenario_id, status, rationale, inputs_hash, model_ids, expected, policy_state, proposed_by)
                     VALUES (%s,%s,%s,%s,%s,%s,'proposed',%s,%s,%s,%s,%s,%s)""",
                  [did, ctx.tenant_id, d["deal_id"], body.decision_type, body.choice, body.scenario_id, body.rationale, hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest(),
                   Jsonb(s["params"]["models"]), Jsonb(expected), Jsonb(policy), ctx.actor_id])
        for k, v in s["params"]["models"].items():
            mid = c.execute("SELECT model_id FROM model_artifact WHERE model_name = %s AND version = %s", [k, v]).fetchone()["model_id"]
            evidence(c, ctx.tenant_id, d["deal_id"], "decision", did, None, f"model {k} {v}", None, mid)
        evidence(c, ctx.tenant_id, d["deal_id"], "decision", did, None, f"integration_scenario {body.scenario_id}", f"{body.choice}: mean value ${opt['value']['mean']:,}, P(covers premium) {opt['value']['p_covers_premium']}")
        audit.record(c, ctx, "decision.proposed", "decision", did, {"choice": body.choice, "expected_value": opt["value"]["mean"]})
        return 201, {"decision_id": did, "status": "proposed", "choice": body.choice, "expected": expected, "policy": policy, "recommended": m["comparison"]["recommended"], "close_call": m["comparison"]["close_call"]}
    return run(ctx, idem, body, work)


class ApproveIn(BaseModel):
    note: str = Field(min_length=3, max_length=500)


@app.post("/v1/decisions/{decision_id}/approve", tags=["decision log"], summary="(+) Approve a proposed decision; never by the person who proposed it")
def approve(decision_id: uuid.UUID, body: ApproveIn, ctx: Ctx = Depends(auth("decisions:propose")), idem: str | None = IdemKey):
    def work(c):
        x = c.execute("SELECT * FROM decision WHERE decision_id = %s", [decision_id]).fetchone()
        if not x:
            raise Problem(404, "decision_not_found")
        if x["proposed_by"] == ctx.actor_id:
            raise Problem(409, "four_eyes", "the person who proposed a decision cannot approve it")
        if "decisions:approve" not in PERMISSIONS[ctx.role]:
            raise Problem(403, "forbidden", "approval needs the deal lead")
        if x["status"] == "approved":
            raise Problem(409, "already_approved")
        c.execute("UPDATE decision SET status = 'approved', approved_by = %s, approved_at = now(), policy_state = policy_state || '{\"four_eyes\": \"passed\"}' WHERE decision_id = %s", [ctx.actor_id, decision_id])
        c.execute("UPDATE model_artifact SET approved = true WHERE version IN (SELECT jsonb_array_elements_text(jsonb_path_query_array(%s::jsonb, '$.*')))", [json.dumps(x["model_ids"])])
        audit.record(c, ctx, "decision.approved", "decision", decision_id, {"note": body.note, "choice": x["choice"]})
        return 200, {"decision_id": decision_id, "status": "approved", "choice": x["choice"], "approved_by": ctx.actor_name}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- realisation tracking

class TrackIn(BaseModel):
    deal_id: uuid.UUID
    quarters: int = Field(4, ge=1, le=12)


@app.post("/v1/synergies/track", status_code=201, tags=["synergy model"], summary="(+) Quarterly synergy run-rate reports after close (delivered by the simulator) turned into an updated forecast; a shortfall opens a review task instead of changing anything")
def track(body: TrackIn, ctx: Ctx = Depends(auth("deal:write")), idem: str | None = IdemKey):
    def work(c):
        d = deal_row(c, body.deal_id)
        dec = c.execute("SELECT * FROM decision WHERE deal_id = %s AND decision_type = 'erp_integration_approach' AND status = 'approved' ORDER BY approved_at DESC LIMIT 1", [d["deal_id"]]).fetchone()
        if not dec:
            raise Problem(409, "no_approved_decision", "approve an integration approach first")
        if d["seed"] is None:
            raise Problem(409, "no_reports", "realisation reports come from the synthetic data room")
        m = models(c)
        mod = m["synergy_realisation"]["params"]
        f, _ = features(c, d)
        tf = truth_features(d, f)
        mc = d["management_case"]
        reports, hidden = W.realisation_reports(d["seed"], dec["choice"], body.quarters, tf, mc["cost_run_rate"], mc["revenue_run_rate"])
        db.many(c, "INSERT INTO realisation_report (tenant_id, deal_id, quarter, month, cost_run_rate, revenue_run_rate) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                [(ctx.tenant_id, d["deal_id"], r["quarter"], r["month"], r["cost_run_rate"], r["rev_run_rate"]) for r in reports])
        tr = E.track(mod, f, dec["choice"], reports, mc)
        out, tasks = {}, []
        for kind, label in (("cost", "cost"), ("rev", "revenue")):
            x = tr[kind]
            status = "on track" if x["p_meets_case"] >= 0.5 else "behind" if x["p_below_70pct"] >= 0.5 else "at risk"
            p10, p50, p90 = (v * x["management_case"] for v in x["posterior"])
            c.execute("INSERT INTO synergy (synergy_id, tenant_id, deal_id, kind, basis, approach, p10, p50, p90, model_id) VALUES (%s,%s,%s,%s,'tracked',%s,%s,%s,%s,%s)",
                      [uuid.uuid7(), ctx.tenant_id, d["deal_id"], label, dec["choice"], p10, p50, p90, m["synergy_realisation"]["model_id"]])
            out[label] = {"status": status, "forecast_run_rate": [money(p10), money(p50), money(p90)], "prior_run_rate": [money(v * x["management_case"]) for v in x["prior"]],
                          "p_meets_case": r3(x["p_meets_case"]), "p_below_70pct": r3(x["p_below_70pct"]), "straight_line": money(x["straight_line"] * x["management_case"]), "management_case": money(x["management_case"]), "ramp_k": r3(mod[kind]["k"])}
            if status == "behind":
                rid = uuid.uuid7()
                c.execute("INSERT INTO risk (risk_id, tenant_id, deal_id, kind, severity, title, exposure_usd, confidence, detail, status) VALUES (%s,%s,%s,'realisation_shortfall','high',%s,%s,%s,%s,'review')",
                          [rid, ctx.tenant_id, d["deal_id"], f"{label.capitalize()} synergies forecast at {p50 / x['management_case']:.0%} of the case after {body.quarters} quarters",
                           x["management_case"] - p50, x["p_below_70pct"], Jsonb({"kind": label, "forecast": [p10, p50, p90]})])
                evidence(c, ctx.tenant_id, d["deal_id"], "risk", rid, None, f"realisation reports Q1-Q{body.quarters}", f"latest {label} run-rate ${reports[-1]['cost_run_rate' if kind == 'cost' else 'rev_run_rate']:,.0f}", m["synergy_realisation"]["model_id"])
                tasks.append({"risk_id": rid, "title": f"Review the {label} synergy plan", "assigned_to": "deal_lead"})
        audit.record(c, ctx, "synergies.tracked", "deal", d["deal_id"], {"quarters": body.quarters, **{k: v["status"] for k, v in out.items()}})
        return 201, {"deal_id": d["deal_id"], "approach": dec["choice"], "reports": [{**r, "cost_run_rate": money(r["cost_run_rate"]), "rev_run_rate": money(r["rev_run_rate"])} for r in reports],
                     **out, "review_tasks": tasks,
                     "how": "A normal update of the deal's log realisation ratio: the model's predictive for the approach taken is the prior; each report divided by the fitted ramp measures the final ratio, with the fitted reporting noise plus the ramp's uncertainty. The PMO's straight line extends the latest report to month 36 in proportion.",
                     "simulation_truth": {"what": "the deal's hidden fate, drawn once by the generator", "cost_run_rate": money(hidden["cost_ratio"] * mc["cost_run_rate"]), "revenue_run_rate": money(hidden["rev_ratio"] * mc["revenue_run_rate"])}}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- the executive summary (templated from evidence, no language model)

@app.get("/v1/deals/{deal_id}/summary", tags=["decision log"], summary="(+) The executive summary: every sentence built from stored risks, synergies, scenarios and decisions, and citing them")
def summary(deal_id: uuid.UUID, ctx: Ctx = Depends(auth("deal:read"))):
    with db.tx(ctx.tenant_id) as c:
        d = deal_row(c, deal_id)
        loaded(c, d)
        l4 = ltm(c, deal_id)
        rk = c.execute("SELECT risk_id, kind, severity, title, exposure_usd, detail FROM risk WHERE deal_id = %s", [deal_id]).fetchall()
        docs = {r["subject_id"]: r["ref"] for r in c.execute("SELECT e.subject_id, doc.ref FROM evidence e JOIN document doc USING (document_id) WHERE e.deal_id = %s", [deal_id])}
        syn = c.execute("SELECT DISTINCT ON (kind, basis) * FROM synergy WHERE deal_id = %s AND (approach IS NULL OR approach = %s OR basis = 'tracked') ORDER BY kind, basis, created_at DESC",
                        [deal_id, d["metadata"].get("proposed_approach")]).fetchall()
        dec = c.execute("SELECT * FROM decision WHERE deal_id = %s ORDER BY created_at DESC LIMIT 1", [deal_id]).fetchone()
        mv = c.execute("SELECT DISTINCT ON (model_name) model_id, model_name, version FROM model_artifact ORDER BY model_name, created_at DESC").fetchall()
    cite = lambda kind, rows: [{"type": kind, "id": str(r["risk_id"] if kind == "risk" else r["synergy_id"]), "ref": docs.get(r.get("risk_id"))} for r in rows]     # noqa: E731
    by = lambda k: [r for r in rk if r["kind"] == k]     # noqa: E731
    out = [{"text": f"{d['target_name']}: ${l4['revenue'] / 1e6:,.0f}M revenue over the last four quarters at a {l4['ebitda'] / l4['revenue']:.0%} EBITDA margin; the price of ${float(d['price']) / 1e9:.2f}B carries a ${float(d['price'] - d['standalone_value']) / 1e6:,.0f}M premium over standalone value.", "evidence": [{"type": "financial_metric", "id": "ltm", "ref": None}]}]
    if by("concentration"):
        x = by("concentration")[0]
        out.append({"text": f"The largest customer is {x['detail']['top1']:.1%} of revenue and the ten largest {x['detail']['top10']:.0%}.", "evidence": cite("risk", [x])})
    coc = by("change_of_control")
    if coc:
        cust = {r["detail"].get("customer") or r["detail"]["document"]: float(r["exposure_usd"]) for r in coc}
        out.append({"text": f"{len(cust)} customers can end their contracts on a change of control, covering ${sum(cust.values()) / 1e6:,.1f}M of annual revenue ({sum(cust.values()) / l4['revenue']:.0%}); the rules that found them miss wording that never names the event, so this is a floor.", "evidence": cite("risk", coc)})
    cy = by("cyber")
    if cy:
        out.append({"text": f"{len(cy)} critical or high cyber findings are open, {sum(bool(r['detail']['internet_facing']) for r in cy)} of them internet-facing; remediation is estimated at ${sum(float(r['exposure_usd']) for r in cy) / 1e6:,.1f}M on stated planning rates.", "evidence": cite("risk", cy)})
    if by("customer_overlap"):
        x = by("customer_overlap")[0]
        out.append({"text": f"{x['detail']['share']:.0%} of revenue comes from customers the acquirer already serves, which the deal database says pulls revenue synergies down.", "evidence": cite("risk", [x])})
    s = {(r["kind"], r["basis"]): r for r in syn}
    if ("cost", "model") in s:
        cm, rm = s[("cost", "model")], s[("revenue", "model")]
        out.append({"text": f"Against a case of ${float(s[('cost', 'management_case')]['p50']) / 1e6:,.0f}M cost and ${float(s[('revenue', 'management_case')]['p50']) / 1e6:,.0f}M revenue synergies, the model fitted to the deal database expects ${float(cm['p50']) / 1e6:,.1f}M (80%: {float(cm['p10']) / 1e6:,.1f}–{float(cm['p90']) / 1e6:,.1f}) and ${float(rm['p50']) / 1e6:,.1f}M ({float(rm['p10']) / 1e6:,.1f}–{float(rm['p90']) / 1e6:,.1f}).",
                    "evidence": cite("synergy", [cm, rm]) + [{"type": "model", "id": str(m["model_id"]), "ref": m["version"]} for m in mv if m["model_name"] == "synergy_realisation"]})
    if dec:
        e = dec["expected"]
        out.append({"text": f"ERP integration: {dec['choice']} ({dec['status']}); expected synergy value ${e['value']['mean'] / 1e6:,.0f}M (80%: {e['value']['p10'] / 1e6:,.0f}–{e['value']['p90'] / 1e6:,.0f}) against a ${e['premium'] / 1e6:,.0f}M premium, a {e['value']['p_covers_premium']:.0%} chance of covering it; the plan as written says ${e['management_case_value'] / 1e6:,.0f}M.",
                    "evidence": [{"type": "decision", "id": str(dec["decision_id"]), "ref": None}, {"type": "integration_scenario", "id": str(dec["scenario_id"]), "ref": None}]})
    if ("cost", "tracked") in s:
        tc, trv = s[("cost", "tracked")], s[("revenue", "tracked")]
        out.append({"text": f"After the latest quarterly reports, cost synergies are forecast at ${float(tc['p50']) / 1e6:,.1f}M and revenue synergies at ${float(trv['p50']) / 1e6:,.1f}M a year.", "evidence": cite("synergy", [tc, trv])})
    review = by("realisation_shortfall")
    if review:
        out.append({"text": f"{len(review)} review task{'s' if len(review) > 1 else ''} open: {'; '.join(r['title'] for r in review)}.", "evidence": cite("risk", review)})
    return jsonable_encoder({"deal_id": deal_id, "sentences": out, "how": "Each sentence is a template filled from stored rows and cites them; there is no language model."})
