"""The SMT proof is part of the test suite: it must hold for the shipped configuration,
the transcription must agree with the real engine, and each ablation must break safety."""
import pytest

from secureagent_iot import formal as F
from secureagent_iot.config import load_config
from secureagent_iot.simulator import build_registry


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_all_theorems_proved(cfg):
    rows = F.prove(cfg)
    assert all(r["result"] == "proved" for r in rows), [r for r in rows if r["result"] != "proved"]


def test_transcription_matches_engine(cfg):
    r = F.differential_test(cfg, build_registry(), n=5000, seed=3)
    assert r["mismatches"] == 0


@pytest.mark.parametrize("name", ["no_provenance", "no_criticality", "no_ttl_cap", "no_budget"])
def test_ablation_breaks_safety(cfg, name):
    rows = F.prove(cfg, F.ABLATIONS[name])
    s1 = [r for r in rows if r["theorem"].startswith("S1:")][0]
    assert s1["result"] == "counterexample"


def test_floor_ablation_breaks_liveness_only(cfg):
    rows = {r["theorem"][:2]: r["result"] for r in F.prove(cfg, F.ABLATIONS["no_floor"])}
    assert rows["S1"] == "proved" and rows["A2"] == "counterexample"
