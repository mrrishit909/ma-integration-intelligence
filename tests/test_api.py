"""Integration, end-to-end and security tests against a real Postgres."""
import psycopg
import pytest

from core import db, jobs, scenario

from .conftest import bearer

LEAD, ANALYST, VIEWER, OTHER = bearer("aldermoor", "deal_lead"), bearer("aldermoor", "analyst"), bearer("aldermoor", "viewer"), bearer("other", "deal_lead")
DEAL = {"external_ref": "X", "name": "X", "target_name": "T", "acquirer_revenue": 1e9, "price": 5e8, "standalone_value": 4e8, "relatedness": 0.5,
        "management_case": {"cost_run_rate": 1e7, "revenue_run_rate": 1e7}}


def test_nothing_runs_before_a_deal_and_its_data_room(client):
    assert client.post("/v1/deals", headers=ANALYST, json={**DEAL, "price": 3e8}).json()["code"] == "no_premium"
    deal_id = client.post("/v1/deals", headers=ANALYST, json=DEAL).json()["deal_id"]
    assert client.get(f"/v1/deals/{deal_id}/risks", headers=VIEWER).json()["code"] == "no_data_room"
    assert client.post("/v1/synergies/model", headers=ANALYST, json={"deal_id": deal_id}).json()["code"] == "no_data_room"
    assert client.post("/v1/diligence/import", headers=ANALYST, json={"deal_id": deal_id, "generate": {"seed": 1}}).status_code == 403
    assert client.post("/v1/diligence/import", headers=LEAD, json={"deal_id": deal_id}).json()["code"] == "generate_or_documents"
    assert client.get("/v1/deals/00000000-0000-0000-0000-000000000000", headers=VIEWER).status_code == 404


@pytest.fixture(scope="module")
def demo(client):
    results, _ = scenario.run(client, drain=jobs.drain)
    return results


def one(sql, params=(), tenant=None):
    with db.tx(tenant) as c:
        return c.execute(sql, params).fetchone()


def test_demo_journey(demo):
    room = demo["room"]["result"]
    assert room["customers"] == 420 and room["documents"] == 41 and room["comparable_deals"] == 240 and one("SELECT count(*) AS n FROM entity_edge")["n"] == room["same_as_edges"]
    docs = {d["ref"]: d for d in demo["docs"]["documents"]}
    assert docs["VDR-2.41"]["risks"][0]["kind"] == "change_of_control" and docs["VDR-2.42"]["risks"] == [] and docs["VDR-2.43"]["status"] == "refused"
    assert abs(demo["target"]["ltm"]["revenue"] - 310e6) < 1e5 and demo["target"]["premium"] == 260_000_000
    r = demo["risks"]
    tr = r["simulation_truth"]
    assert r["summary"]["concentration"]["top1"] > 0.1 and r["summary"]["cyber_open_critical_high"] == tr["cyber_open_critical_high"]
    assert tr["change_of_control"]["false_alarms"] == 0 and tr["change_of_control"]["found"] < tr["change_of_control"]["contracts"]
    assert r["entity_graph"]["exact_name_baseline"]["overlap_share"] < 0.5 * tr["overlap_share"] < r["entity_graph"]["overlap_share"]
    assert all(x["evidence"] for x in r["risks"])                                          # every risk says where it came from
    s = demo["syn"]
    assert one("SELECT count(*) AS n FROM model_artifact")["n"] == 2 and s["database"]["went_quiet"] > 0
    assert s["distributions"]["fast"]["revenue"]["ratio"][1] < s["distributions"]["staged"]["revenue"]["ratio"][1] < s["baselines"]["comparables"]["revenue"] < 1
    sy = demo["systems"]
    assert sy["simulation_truth"]["clusters"]["recall"] > sy["simulation_truth"]["same_product"]["recall"] and sy["target_erp"]["interfaces"]
    sc = demo["scen"]["result"]
    for ap in ("fast", "staged"):
        o = sc["options"][ap]
        assert o["management_case_value"] > o["value"]["p90"] and o["done_weeks"][0] > o["planned_weeks"] and str(o["lanes"]) in o["lane_scores"]
        assert max(o["lane_scores"].values()) == o["lane_scores"][str(o["lanes"])]
    assert sc["options"]["staged"]["p_disruption"] < sc["options"]["fast"]["p_disruption"] and one("SELECT count(*) AS n FROM milestone")["n"] == 2 + 4
    assert demo["self"]["code"] == "four_eyes" and demo["approve"]["status"] == "approved" and one("SELECT count(*) AS n FROM model_artifact WHERE approved")["n"] == 2
    t = demo["track"]
    assert len(t["reports"]) == 4 and one("SELECT count(*) AS n FROM realisation_report")["n"] == 4
    assert t["revenue"]["status"] == "behind" and t["review_tasks"] and one("SELECT count(*) AS n FROM risk WHERE status = 'review'")["n"] == 1
    lo, _, hi = t["revenue"]["forecast_run_rate"]
    assert lo <= t["simulation_truth"]["revenue_run_rate"] <= hi
    assert all(x["evidence"] for x in demo["summary"]["sentences"]) and len(demo["summary"]["sentences"]) >= 8
    assert demo["audit"]["chain_valid"] and {"deal.created", "diligence.generated", "models.fitted", "integration.scenario_completed", "decision.proposed", "decision.approved", "synergies.tracked"} <= {e["action"] for e in demo["audit"]["events"]}


def test_validation_replay_and_approval_rules(client, demo):
    deal_id, dec = demo["deal"]["deal_id"], demo["propose"]["decision_id"]
    doc = {"ref": "VDR-2.41", "kind": "side_letter", "title": "again", "text": "Customer may terminate if Supplier is acquired by anyone."}
    assert client.post("/v1/diligence/import", headers=ANALYST, json={"deal_id": deal_id, "documents": [doc]}).json()["documents"][0]["status"] == "refused"
    assert client.post("/v1/diligence/import", headers=ANALYST, json={"deal_id": deal_id, "documents": [{**doc, "text": "short"}]}).status_code == 422
    assert client.post(f"/v1/decisions/{dec}/approve", headers=LEAD, json={"note": "again"}).json()["code"] == "already_approved"
    assert client.post(f"/v1/decisions/{dec}/approve", headers=VIEWER, json={"note": "me"}).status_code == 403
    h = {**ANALYST, "Idempotency-Key": "track-replay"}
    a, b = client.post("/v1/synergies/track", headers=h, json={"deal_id": deal_id, "quarters": 2}), client.post("/v1/synergies/track", headers=h, json={"deal_id": deal_id, "quarters": 2})
    assert a.json() == b.json() and b.headers.get("Idempotent-Replay") == "true"
    assert client.post("/v1/synergies/track", headers=h, json={"deal_id": deal_id, "quarters": 3}).json()["code"] == "idempotency_key_reused"


def test_tenant_isolation_and_append_only_audit(client, demo):
    deal_id = demo["deal"]["deal_id"]
    assert client.get(f"/v1/deals/{deal_id}", headers=OTHER).status_code == 404
    assert client.get(f"/v1/deals/{deal_id}/risks", headers=OTHER).status_code == 404
    assert client.get(f"/v1/systems/dependencies?deal_id={deal_id}", headers=OTHER).status_code == 404
    with db.tx() as c:
        other = c.execute("SELECT DISTINCT tenant_id FROM api_tokens WHERE actor_name LIKE '%%Pellucid%%'").fetchone()["tenant_id"]
    for table in ("document", "risk", "company", "comparable_deal", "decision"):
        assert one(f"SELECT count(*) AS n FROM {table}", tenant=other)["n"] == 0
    with pytest.raises(psycopg.errors.RaiseException), db.tx() as c:
        c.execute("UPDATE audit_events SET action = 'edited'")
