import os

import pandas as pd
import pytest

from secureagent_iot.data import GROUP_COL, feature_columns, make_toy, split, validate_and_report
from secureagent_iot.evaluation import run_attack_benchmark, run_response_benchmark
from secureagent_iot.llm_adapter import LLMUnavailable, ToolCallingAgent, to_raw_proposal
from secureagent_iot.agents import view_of
from secureagent_iot.schemas import ProposalRejected, parse_proposal
from secureagent_iot.simulator import FAMILIES, containment_worthy, realize

MEASURED = ["decision_latency_us"]  # wall-clock measurements are not expected to repeat


def test_templates_meet_target(templates, registry, cfg):
    assert len(templates) >= 60
    assert len({t.scenario_id for t in templates}) == len(templates)
    assert {t.family for t in templates} == set(FAMILIES)
    assert {t.uncertainty for t in templates} >= {"low", "medium", "high"}
    assert {registry[t.device_id].criticality for t in templates} == {"low", "medium", "high"}
    assert any(containment_worthy(t, registry, cfg["spec"]) for t in templates)


def test_realization_is_seed_deterministic(templates):
    a, b, c = realize(templates[5], 42), realize(templates[5], 42), realize(templates[5], 43)
    assert a.p_attack == b.p_attack and a.observations == b.observations
    assert a.p_attack != c.p_attack


def test_response_benchmark_reproducible(cfg):
    a = run_response_benchmark(cfg, [42]).drop(columns=MEASURED)
    b = run_response_benchmark(cfg, [42]).drop(columns=MEASURED)
    pd.testing.assert_frame_equal(a, b)
    c = run_response_benchmark(cfg, [43]).drop(columns=MEASURED)
    assert not a["p_attack"].equals(c["p_attack"])
    # identical scenario IDs for every system
    sets = a.groupby("system")["scenario_id"].apply(frozenset)
    assert sets.nunique() == 1


def test_attack_benchmark_reproducible(cfg):
    kw = dict(systems=["B1_unguarded", "B2_guarded"], controls=False, compliance=0.5)
    a = run_attack_benchmark(cfg, [42], **kw).drop(columns=MEASURED)
    b = run_attack_benchmark(cfg, [42], **kw).drop(columns=MEASURED)
    pd.testing.assert_frame_equal(a, b)
    # paired design: both systems face the same compliance draw
    p = a.pivot(index=["attack_id", "scenario_id"], columns="system", values="proposal_success")
    assert (p["B1_unguarded"] == p["B2_guarded"]).all()


def test_toy_split_has_no_group_or_row_leakage():
    df, report = validate_and_report(make_toy(seed=1))
    train, test, info = split(df, 0.2, seed=1)
    assert info["group_overlap"] == 0 and info["cross_split_identical_feature_rows"] == 0
    feats, _ = feature_columns(train)
    assert "label" not in feats and GROUP_COL not in feats and "y" not in feats and "family" not in feats
    assert report["missing_values_total"] > 0  # the imputer has something to do


def test_llm_adapter_refuses_without_credentials(monkeypatch, templates, registry):
    for k in ("LLM_PROVIDER", "LLM_MODEL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(LLMUnavailable):
        ToolCallingAgent()
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    with pytest.raises(LLMUnavailable):
        ToolCallingAgent()
    # model output is untrusted: malformed or over-scoped tool arguments are rejected downstream
    view = view_of(realize(templates[0], 1))
    with pytest.raises(ProposalRejected):
        parse_proposal(to_raw_proposal("just isolate everything", view), registry)
    with pytest.raises(ProposalRejected):
        parse_proposal(to_raw_proposal({"device_id": view.device_id, "action_type": "restrict_egress", "duration_seconds": 60,
                                        "evidence_refs": ["x"], "reason": "r", "run_shell": "id"}, view), registry)


class _PlumbingDouble:
    """TEST DOUBLE for wiring only. It is not an LLM and produces no reported result."""
    provider, model = "test-double", "none"

    def __init__(self):
        self.calls = []

    def propose(self, view, feedback=None, memory=None):
        self.seen_memory = memory
        self.calls.append({"model": "none", "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0})
        return {"incident_id": view.incident_id, "agent_id": "responder", "device_id": view.device_id,
                "action_type": "restrict_egress", "duration_seconds": 300 if feedback else 4000,
                "evidence_refs": [o["evidence_id"] for o in view.observations], "reason": view.untrusted_text,
                "memory_note": "note from " + view.incident_id}


def test_llm_code_path_is_wired_and_still_guarded(cfg):
    df = run_response_benchmark(cfg, [42], llm=_PlumbingDouble())
    assert set(df.system) == {"B0_rules", "B1_unguarded", "B2_guarded"}
    b2 = df[df.system == "B2_guarded"]
    assert not b2.spec_violation.any() and (b2.retries == 1).mean() > 0.9
    assert df[df.system == "B1_unguarded"].spec_violation.all()
    atk = run_attack_benchmark(cfg, [42], llm=_PlumbingDouble(), controls=False)
    assert {"single", "memory_notify", "memory_release"} <= set(atk["mode"])   # memory variants run on the LLM path too
    assert not atk[atk.system == "B2_guarded"].any_violation.any()


def test_llm_flag_blocks_cleanly_without_credentials(monkeypatch, tmp_path):
    import importlib.util, pathlib
    for k in ("LLM_PROVIDER", "LLM_MODEL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_response_benchmark.py"
    spec = importlib.util.spec_from_file_location("rrb", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--llm", "--out", str(tmp_path / "raw")]) == 2
    assert not (tmp_path / "raw").exists()


def test_llm_memory_notes_are_carried_and_never_reach_the_schema(cfg, registry, templates):
    from secureagent_iot.agents import AgenticSystem
    from secureagent_iot.config import rng_for
    from secureagent_iot.simulator import Testbed
    double = _PlumbingDouble()
    system = AgenticSystem("B2_guarded", cfg, guarded=True, llm=double)
    tb = Testbed(cfg, registry, "t")
    for t in templates[:2]:
        sc = realize(t, 1)
        tb.register(sc)
        res = system.handle(sc, tb, rng_for("t"))
        assert res["proposal_valid"]            # memory_note was stripped before strict parsing
    assert len(system.memory) == 2 and double.seen_memory and "note from" in double.seen_memory[0]


def test_detector_sampler_replaces_parametric_probabilities(cfg):
    from sklearn.dummy import DummyClassifier
    from secureagent_iot.simulator import DetectorSampler, build_templates
    df, _ = validate_and_report(make_toy(seed=3))
    feats, _ = feature_columns(df)
    model = DummyClassifier(strategy="prior").fit(df[feats].fillna(0), df["y"])
    sampler = DetectorSampler(model, df.fillna(0), feats, "toy-sanity")
    t = build_templates()[0]
    assert realize(t, 5, sampler).p_attack == pytest.approx(round(float(df["y"].mean()), 4))
    assert realize(t, 5, sampler).p_attack != realize(t, 5).p_attack
    out = run_response_benchmark(cfg, [5], systems=["B2_guarded"], sampler=sampler)
    assert not out.spec_violation.any()
