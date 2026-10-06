"""The parsers, the matcher, the fits, the schedule and the value model, without a database."""
import numpy as np

from ma import engine as E, world as W


def test_matcher_reads_spelling_variants_but_not_other_companies():
    assert E.normalise("The Halden Pumps & Valves Inc.") == E.normalise("HALDEN PUMPS AND VALVES INCORPORATED") == "halden pumps and valves"
    assert E.normalise("Brask Mfg LLC") == E.normalise("Brask Manufacturing")
    m = E.match_customers(["Halden Mechanical Inc", "Brask Metals LLC", "Corvel Foods"], ["Halden Mechanicla Inc", "Brask Plastics LLC", "Corvel Freight"])
    assert set(m) == {0} and m[0][2] == "fuzzy"             # a typo is caught; a different company with the same first word is not
    assert E.match_exact(["Halden Mechanical Inc"], ["HALDEN MECHANICAL INC"]) == {}


def test_contract_rules_and_their_known_miss():
    for k, sentence in enumerate(W.COC):
        found = E.parse_contract(f"title\n1. {sentence}")["coc"]
        assert bool(found) == (k != 4)                          # the wording that never names the event is missed, and the evaluation says so
    for sentence in W.BOILER:
        assert E.parse_contract(f"title\n1. {sentence}") == {"coc": [], "tfc": []}
    assert all(E.parse_contract(f"title\n1. {s}")["tfc"] for s in W.TFC)


def test_cyber_parser_reads_every_finding():
    r = W.data_room(41)
    found = E.parse_cyber(r["documents"][-1]["text"])
    assert [(f["id"], f["severity"], f["open"], f["internet_facing"]) for f in found] == [(t["id"], t["severity"], t["open"], t["internet_facing"]) for t in r["truth"]["findings"]]


def test_fits_recover_the_ramp_and_the_survivorship_bias_shows():
    rows, _ = W.comparables(41)
    m = E.fit_synergy(rows)
    assert abs(m["cost"]["k"] - W.TRUTH["ramp"]["cost"]) < 0.05 and abs(m["rev"]["k"] - W.TRUTH["ramp"]["rev"]) < 0.1
    assert abs(m["cost"]["report_sd"] - W.TRUTH["meas_sd"]) < 0.02 and abs(m["rev"]["sd"] - W.TRUTH["rev"]["sd"]) < 0.08
    assert m["comps"]["rev"] > m["naive_latest"]["rev"] + 0.1             # survivors high, unfinished deals taken as final low
    e = E.fit_erp(rows)
    assert abs(np.exp(e["bias_mu"]) - np.exp(W.TRUTH["erp"]["bias_mu"])) < 0.08 and e["running"] > 0


def test_schedule_respects_dependencies_and_team_size():
    ws = E.workstreams([(f"T{i}", f"App {i}", 10 ** (i % 4 + 1)) for i in range(9)])
    assert all(E.lanes_of(ws["interfaces"], k + 1) <= E.lanes_of(ws["interfaces"], k) for k in range(1, 8))
    fast, staged = E.plan(ws, "fast", 3), E.plan(ws, "staged", 3)
    start = {k: (a, b) for _, k, _, a, b in fast["tasks"]}
    assert start["migrate"][0] >= max(start["coa"][1], start["clean"][1]) and start["test"][0] >= start["interfaces"][1]
    assert len(staged["cutovers"]) == 3 and abs(sum(s for _, s in staged["cutovers"]) - 1) < 1e-9
    assert staged["done"] > fast["done"] and [w for w, _ in staged["cutovers"]] == sorted(w for w, _ in staged["cutovers"])


def test_value_model_moves_the_right_way():
    ws = E.workstreams([(f"T{i}", f"App {i}", 1000) for i in range(8)])
    p = E.plan(ws, "fast", 3)
    deal, one, none = W.DEAL, np.ones(3), np.zeros((3, 1))
    base = E.simulate(p, one, one, one, none.astype(bool), none, 0.8, 1.5, "fast", 3, deal, 310e6)["value"]
    late = E.simulate(p, one, one, np.array([1.0, 1.5, 2.0]), none.astype(bool), none, 0.8, 1.5, "fast", 3, deal, 310e6)["value"]
    hit = E.simulate(p, one, one, one, np.ones((3, 1), bool), np.full((3, 1), 4.0), 0.8, 1.5, "fast", 3, deal, 310e6)["value"]
    low = E.simulate(p, one * 0.5, one * 0.5, one, none.astype(bool), none, 0.8, 1.5, "fast", 3, deal, 310e6)["value"]
    assert late[0] > late[1] > late[2] and (hit < base).all() and (low < base).all()


def test_tracker_follows_the_reports():
    rows, _ = W.comparables(41)
    m = E.fit_synergy(rows)
    f = {"related": 0.7, "cost_pct": 0.11, "rev_pct": 0.15, "rel_size": 0.13, "overlap": 0.3}
    mc = W.DEAL["management_case"]
    strong = [{"month": 3 * q, "cost_run_rate": mc["cost_run_rate"] * 1.2 * W.ramp(3 * q, 0.8), "rev_run_rate": mc["revenue_run_rate"] * 0.3 * W.ramp(3 * q, 1.5)} for q in range(1, 5)]
    t = E.track(m, f, "staged", strong, mc)
    assert abs(t["cost"]["posterior"][1] - 1.2) < 0.08 and abs(t["rev"]["posterior"][1] - 0.3) < 0.05
    assert t["cost"]["posterior"][2] - t["cost"]["posterior"][0] < t["cost"]["prior"][2] - t["cost"]["prior"][0]
