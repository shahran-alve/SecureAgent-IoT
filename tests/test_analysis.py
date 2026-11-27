"""scripts/analyze_llm_run.py on small synthetic frames with known answers. No model, no network."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analyze_llm_run", ROOT / "scripts" / "analyze_llm_run.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

N_FAST = 2000      # resamples/permutations in tests; the script's default is 10,000


def attack_frame(n_clusters=20, effect_clusters=10, value="proposal_success", model="m/M", condition="base", system="B2_guarded"):
    """Every cluster (attack_id, scenario_id) x 5 seeds x (4 styles + 1 control). The control is always 0; an injected
    attempt is 1 exactly in the first `effect_clusters` clusters. True paired difference = effect_clusters / n_clusters."""
    rows = []
    for k in range(n_clusters):
        for seed in A.SEEDS:
            base = {"model": model, "condition": condition, "system": system, "seed": seed, "attack_id": f"A{k % 7}.{k}",
                    "scenario_id": f"S-{k:02d}"}
            rows.append({**base, "style": A.CONTROL_STYLE, "injected": False, value: False})
            for style in A.STYLES:
                rows.append({**base, "style": style, "injected": True, value: k < effect_clusters})
    return pd.DataFrame(rows)


def stats_for(d, value="proposal_success", styles=A.STYLES, key=("t",)):
    pr = A.pair_injected_with_control(d, value, styles)
    return pr, A.paired_stats(pr["diff"].values, pr["cluster"].values, key, N_FAST, N_FAST)


def test_known_effect_gives_the_right_difference_interval_and_p():
    pr, st = stats_for(attack_frame(20, 10))
    assert len(pr) == 20 * 5 * 4 and st["n_pairs"] == 400 and st["n_clusters"] == 20
    assert st["estimate"] == pytest.approx(0.5)
    assert 0.25 < st["ci95_lo"] < 0.5 < st["ci95_hi"] < 0.75
    # 10 of 20 clusters differ, all in the same direction: exact two-sided sign-flip p = 2 / 2**10 ~ 0.002
    assert st["p_value"] < 0.01
    # all 20 clusters differ: the interval collapses to the effect and p reaches its floor
    _, st = stats_for(attack_frame(20, 20))
    assert st["estimate"] == 1.0 and st["ci95_lo"] == st["ci95_hi"] == 1.0 and st["p_value"] == pytest.approx(1 / (N_FAST + 1), abs=2e-3)


def test_no_effect_gives_zero_and_p_one():
    _, st = stats_for(attack_frame(20, 0))
    assert st["estimate"] == 0 and st["ci95_lo"] == st["ci95_hi"] == 0 and st["p_value"] == 1.0


def test_effects_that_cancel_across_clusters_are_not_significant():
    d = attack_frame(20, 10)
    d.loc[d["scenario_id"].isin([f"S-{k:02d}" for k in range(10, 20)]) & ~d["injected"], "proposal_success"] = True   # control 1 there
    _, st = stats_for(d)
    assert st["estimate"] == pytest.approx(0.0) and st["p_value"] == 1.0 and st["ci95_lo"] < 0 < st["ci95_hi"]


def test_per_style_pairs_reuse_the_same_control():
    d = attack_frame(12, 6)
    d.loc[d["injected"] & (d["style"] == "authority"), "proposal_success"] = True
    for style, want in (("authority", 1.0), ("naive", 0.5)):
        pr, st = stats_for(d, styles=[style], key=(style,))
        assert len(pr) == 12 * 5 and st["estimate"] == pytest.approx(want)


def test_results_are_reproducible_and_seeded_by_test_name():
    d = attack_frame(15, 5)
    a, b, c = (stats_for(d, key=k)[1] for k in (("x",), ("x",), ("y",)))
    assert a == b and (a["ci95_lo"], a["ci95_hi"]) != (c["ci95_lo"], c["ci95_hi"]) or a["p_value"] != c["p_value"]


def test_pairing_refuses_rows_without_a_matching_control():
    d = attack_frame(5, 2)
    lost = d[~d["injected"]].index[3]
    with pytest.raises(A.PairingError, match="no matching control"):
        A.pair_injected_with_control(d.drop(index=lost), "proposal_success", A.STYLES)
    dup = pd.concat([d, d[~d["injected"]].iloc[[0]]])
    with pytest.raises(A.PairingError, match="share a pairing unit"):
        A.pair_injected_with_control(dup, "proposal_success", A.STYLES)
    orphan = d.drop(index=d[d["injected"] & (d["scenario_id"] == "S-01")].index)
    with pytest.raises(A.PairingError, match="controls have no injected attempt"):
        A.pair_injected_with_control(orphan, "proposal_success", A.STYLES)
    a = pd.DataFrame({"seed": [1, 2], "scenario_id": ["S", "S"], "v": [0, 1]})
    with pytest.raises(A.PairingError, match="one side only"):
        A.pair_on_keys(a, a.iloc[:1], ["seed", "scenario_id"], "v", "t")
    assert A.pair_on_keys(a, a.assign(v=[1, 1]), ["seed", "scenario_id"], "v", "t")["diff"].tolist() == [1.0, 0.0]


def test_holm_and_clopper_pearson():
    p = pd.Series([0.01, 0.04, 0.03, np.nan], index=list("abcd"))
    assert A.holm(p).round(6).tolist()[:3] == [0.03, 0.06, 0.06] and np.isnan(A.holm(p)["d"])
    assert A.clopper_pearson_upper(0, 100) == pytest.approx(1 - 0.05 ** (1 / 100))
    assert A.clopper_pearson_upper(0, 76800) == pytest.approx(3.9e-5, rel=0.01)
    assert A.clopper_pearson_upper(5, 5) == 1.0 and np.isnan(A.clopper_pearson_upper(0, 0))
    assert 0.05 < A.clopper_pearson_upper(2, 50) < 0.13


# ---------------------------------------------------------------------------
# End to end on synthetic merged CSVs
# ---------------------------------------------------------------------------
def write_synthetic_run(raw: Path, models: list[str], n_cases: int = 8, n_scen: int = 6):
    """Merged CSVs with every column the analysis reads. Known facts: injection raises proposal_success from 0 to 1
    in every case of `base`; B2 never violates; B1 violates whenever the proposal succeeds; one containment-worthy
    scenario is contained after a TTL retry."""
    raw.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for m in models:
        for c in A.CONDITIONS:
            rows = []
            for k in range(n_cases):
                aid, sid = ("A6.1" if k == 0 else f"X{k}.1"), f"S-{k % n_scen:02d}"
                for seed in A.SEEDS:
                    for system in ("B1_unguarded", "B2_guarded"):
                        for style, inj in [(s, True) for s in A.STYLES] + [(A.CONTROL_STYLE, False)]:
                            prop = bool(inj and c == "base")
                            rows.append({"system": system, "seed": seed, "attack_id": aid, "category": "x", "scenario_id": sid,
                                         "style": style, "injected": inj, "proposal_success": prop,
                                         "benign_isolated": bool(system == "B2_guarded" and inj and k == 3),
                                         "suppression_scored": k == 0, "suppression_success": bool(k == 0 and inj and system == "B2_guarded"),
                                         "escalated": bool(system == "B2_guarded" and not prop), "any_violation": bool(system == "B1_unguarded" and prop),
                                         "exec_success": bool(system == "B1_unguarded" and prop), "policy_decisions": "ALLOW_REVIEW",
                                         "prompt_condition": c})
            pd.DataFrame(rows).to_csv(raw / f"LLM_{A.slug(m)}__{c}_attack_runs.csv", index=False)
            rows = []
            for j in range(n_scen):
                for seed in A.SEEDS:
                    for system in ("B0_rules", "B1_unguarded", "B2_guarded"):
                        cw, mal = j < 3, j < 4
                        retry = bool(system == "B2_guarded" and j == 0)
                        rows.append({"system": system, "seed": seed, "scenario_id": f"S-{j:02d}", "containment_worthy": cw, "truth_malicious": mal,
                                     "safe_contained": bool(cw and system != "B1_unguarded"), "false_isolation": bool(not mal and system == "B0_rules"),
                                     "escalated": bool(system == "B2_guarded" and not cw), "spec_violation": bool(system == "B1_unguarded" and cw),
                                     "utility": float(rng.normal()), "retries": int(retry),
                                     "policy_decisions": "DENY_TTL_EXCEEDS_CAP>ALLOW_RESTRICT" if retry else ("DENY_SCHEMA_INVALID" if j == 5 else "ALLOW_RESTRICT"),
                                     "prompt_condition": c})
            pd.DataFrame(rows).to_csv(raw / f"LLM_{A.slug(m)}__{c}_response_runs.csv", index=False)


def test_end_to_end_on_synthetic_merged_csvs(tmp_path):
    main_m, phi = ["v1/Main-One", "v2/Main-Two"], ["v3/Phi-Like"]
    raw, tables, report = tmp_path / "raw", tmp_path / "tables", tmp_path / "LLM_ANALYSIS.md"
    write_synthetic_run(raw, main_m + phi)
    assert A.main(["--raw", str(raw), "--tables", str(tables), "--report", str(report), "--main-models", *main_m,
                   "--fail-closed-models", *phi, "--n-boot", "300", "--n-perm", "300"]) == 0
    f1 = pd.read_csv(tables / "LLM_analysis_family1_injection.csv")
    a = f1[(f1["metric"] == "a_proposal_goal") & (f1["styles"] == "pooled")].set_index(["model", "condition"])
    assert a.loc[("v1/Main-One", "base"), "estimate"] == 1.0 and a.loc[("v1/Main-One", "policy"), "estimate"] == 0.0
    assert f1["family"].astype(str).value_counts().to_dict() == {"1": 2 * 4 * 5 * 5, "1_phi": 4 * 5 * 5}
    assert {"injected_numerator", "injected_denominator", "control_numerator", "control_denominator", "n_clusters", "p_holm"} <= set(f1.columns)
    sup = f1[(f1["metric"] == "c_b2_suppression") & (f1["styles"] == "pooled")]
    assert (sup["injected_denominator"] == 4 * 5).all() and (sup["estimate"] == 1.0).all()      # A6.1 only: one case
    assert (f1["p_holm"] >= f1["p_value"] - 1e-12).all()
    f2 = pd.read_csv(tables / "LLM_analysis_family2_condition.csv")
    asr = f2[(f2["metric"] == "injected_proposal_asr") & (f2["from"] == "base") & (f2["to"] == "policy")]
    assert (asr["estimate"] == -1.0).all() and len(f2) == 3 * 4 * 7
    f4 = pd.read_csv(tables / "LLM_analysis_family4_prohibited.csv")
    pooled = f4[f4["pooled"] & (f4["benchmark"] == "both")].iloc[0]
    assert pooled["prohibited_executions"] == 0 and pooled["b2_attempts"] == 2 * 4 * (8 * 5 * 5 + 6 * 5)
    assert pooled["cp_upper_95_one_sided"] == pytest.approx(1 - 0.05 ** (1 / pooled["b2_attempts"]))
    assert "Phi" not in pooled["scope"] and (f4["note"] == "relative to the written spec only").all()
    f5 = pd.read_csv(tables / "LLM_analysis_family5_containment_source.csv")
    assert (f5["allowed_after_ttl_denial_and_retry"] == 5).all() and (f5["first_proposal_allowed"] == 10).all()
    f6 = pd.read_csv(tables / "LLM_analysis_family6_phi_schema_rejections.csv")
    assert set(f6["model"]) == set(phi) and (f6[f6["benchmark"] == "response"]["first_decision_schema_rejection"] == 5).all()
    sens = pd.read_csv(tables / "LLM_analysis_family6_phi_sensitivity.csv")
    assert sens["analysis"].str.startswith("SENSITIVITY").all()
    text = report.read_text()
    assert "relative to the written authorization spec only" in text and "Sensitivity analysis, not a causal estimate" in text
    assert text.count("| Model |") >= 6


def test_refuses_missing_merged_files_and_overlapping_model_lists(tmp_path):
    with pytest.raises(FileNotFoundError, match="merged CSVs missing"):
        A.main(["--raw", str(tmp_path), "--tables", str(tmp_path / "t"), "--report", str(tmp_path / "r.md"),
                "--main-models", "x/Y", "--fail-closed-models", "z/W"])
    assert A.main(["--raw", str(tmp_path), "--main-models", "x/Y", "--fail-closed-models", "x/Y"]) == 2
