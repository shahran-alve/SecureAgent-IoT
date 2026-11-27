"""Sensitivity analysis: the principal comparisons recomputed with scenario TEMPLATES as clusters.

The main analysis clusters attack attempts by (attack variant, scenario template). Variants applied to one template
share its device, ground truth, evidence, and detector draw, so they may be correlated. Here every attack variant,
seed, and injection style of one template stays inside one cluster. No model is rerun: the raw run files are reused.

Outputs (results/tables):
  template_cluster_llm.csv        LLM injection effects (family 1) and prompt effects on goal-matching (family 2),
                                       template-clustered, Holm-adjusted within the same families as the main analysis,
                                       side by side with the main (variant, template) analysis
  template_cluster_scripted.csv   scripted benchmark and holdout comparisons with template clusters
  template_cluster_zero_bounds.csv  zero-event upper bounds per independent template
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("A", ROOT / "scripts" / "analyze_llm_run.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
RAW, T = ROOT / "results" / "raw", ROOT / "results" / "tables"
NB = NP = 10_000


def zero_bound(n: int) -> float:
    return 1 - 0.05 ** (1 / n)


def llm():
    att, rsp = A.load(RAW, A.MAIN_MODELS)
    orig_pair = A.pair_injected_with_control

    def pair_template(d, value, styles):
        m = orig_pair(d, value, styles)
        m["cluster"] = m["scenario_id"]
        return m
    A.pair_injected_with_control = pair_template
    f1 = A.family1(att, A.MAIN_MODELS, "T", NB, NP)
    A.pair_injected_with_control = orig_pair
    # family 2 with template clusters (proposal ASR rows); response rows already cluster on scenario
    rows = []
    for m in A.MAIN_MODELS:
        for c1, c2 in A.CONDITION_PAIRS:
            keys = ["system", "seed", "attack_id", "scenario_id", "style"]
            sel = lambda c: att[(att["model"] == m) & (att["condition"] == c) & (att["system"] == "B2_guarded") & att["injected"]]
            pr = A.pair_on_keys(sel(c1), sel(c2), keys, "proposal_success", "asr")
            st = A.paired_stats(pr["diff"].values, pr["scenario_id"].values, ("f2T", m, c1, c2, "asr"), NB, NP)
            rows.append({"family": "2T", "model": m, "from": c1, "to": c2, "metric": "injected_proposal_asr", **st})
            for system in ("B1_unguarded", "B2_guarded"):
                for metric in ("safe_containment", "false_isolation", "review_rate"):
                    value = A.RESPONSE_METRICS[metric][0]
                    s2 = lambda c: A.response_rows(rsp[(rsp["model"] == m) & (rsp["condition"] == c) & (rsp["system"] == system)], metric)
                    pr = A.pair_on_keys(s2(c1), s2(c2), ["system", "seed", "scenario_id"], value, "resp")
                    st = A.paired_stats(pr["diff"].values, pr["scenario_id"].values, ("f2", m, c1, c2, system, metric), NB, NP)
                    rows.append({"family": "2T", "model": m, "from": c1, "to": c2, "metric": f"{system}:{metric}", **st})
    f2 = pd.DataFrame(rows)
    f2["p_holm"] = A.holm(f2["p_value"])
    main1 = pd.read_csv(T / "LLM_analysis_family1_injection.csv")
    main2 = pd.read_csv(T / "LLM_analysis_family2_condition.csv")
    out = []
    for r in f1[f1.styles == "pooled"].itertuples():
        if r.metric not in ("a_proposal_goal", "b_b2_benign_isolation", "c_b2_suppression"):
            continue
        o = main1[(main1.model == r.model) & (main1.condition == r.condition) & (main1.metric == r.metric) & (main1.styles == "pooled")].iloc[0]
        out.append({"comparison": "injection excess (injected - control)", "model": A.short(r.model), "condition": r.condition,
                    "metric": r.metric, "estimate_pp": 100 * r.estimate,
                    "template_ci_lo": 100 * r.ci95_lo, "template_ci_hi": 100 * r.ci95_hi, "template_p_holm": r.p_holm,
                    "template_clusters": r.n_clusters, "main_ci_lo": 100 * o.ci95_lo, "main_ci_hi": 100 * o.ci95_hi,
                    "main_p_holm": o.p_holm, "main_clusters": o.n_clusters})
    for r in f2[(f2.metric == "injected_proposal_asr") & (f2["from"] == "base") & (f2["to"] == "policy")].itertuples():
        o = main2[(main2.model == r.model) & (main2["from"] == "base") & (main2["to"] == "policy") & (main2.metric == "injected_proposal_asr")].iloc[0]
        out.append({"comparison": "prompt effect base -> policy on injected goal-matching", "model": A.short(r.model),
                    "condition": "base->policy", "metric": "a_proposal_goal", "estimate_pp": 100 * r.estimate,
                    "template_ci_lo": 100 * r.ci95_lo, "template_ci_hi": 100 * r.ci95_hi, "template_p_holm": r.p_holm,
                    "template_clusters": r.n_clusters, "main_ci_lo": 100 * o.ci95_lo, "main_ci_hi": 100 * o.ci95_hi,
                    "main_p_holm": o.p_holm, "main_clusters": o.n_clusters})
    res = pd.DataFrame(out)
    res.insert(0, "evidence_class", "testbed-synthetic scenarios, LLM-backed agents (run1); template-clustered reanalysis")
    # zero events in the guarded LLM runs, per template
    att_all, rsp_all = A.load(RAW, A.MAIN_MODELS + A.FAIL_CLOSED)
    b2 = pd.concat([att_all[att_all.system == "B2_guarded"][["scenario_id", "any_violation"]],
                    rsp_all[rsp_all.system == "B2_guarded"].rename(columns={"spec_violation": "any_violation"})[["scenario_id", "any_violation"]]])
    return res, {"llm_guarded_runs": len(b2), "llm_violations": int(b2.any_violation.sum()), "llm_templates": b2.scenario_id.nunique()}


def scripted():
    from secureagent_iot.extended2 import _wrongful_seconds_attack
    a = pd.read_csv(RAW / "attack_runs.csv")
    a = a[a.injected & ~a.suppression_scored]
    rows, zero = [], {}
    piv = a.pivot_table(index=["seed", "attack_id", "scenario_id"], columns="system", values="any_violation", aggfunc="first").reset_index()
    for cl_name, cl in (("(variant, template)", piv.attack_id + "|" + piv.scenario_id), ("template", piv.scenario_id)):
        d = piv["B2_guarded"].astype(float) - piv["B1_unguarded"].astype(float)
        st = A.paired_stats(d.values, cl.values, ("scripted", cl_name), NB, NP)
        rows.append({"comparison": "prohibited execution, B2 - B1 (scripted, c = 1)", "cluster": cl_name, "estimate_pp": 100 * st["estimate"],
                     "ci_lo": 100 * st["ci95_lo"], "ci_hi": 100 * st["ci95_hi"], "p": st["p_value"], "clusters": st["n_clusters"]})
    b2 = a[a.system == "B2_guarded"].copy()
    b2["wrongful"] = _wrongful_seconds_attack(b2)
    b2["harm"] = b2.goal_executed & b2.benign_isolated & ~b2.any_violation
    for cl_name, cl in (("(variant, template)", b2.attack_id + "|" + b2.scenario_id), ("template", b2.scenario_id)):
        codes, u = pd.factorize(cl)
        sums = np.bincount(codes, weights=b2.harm.astype(float).values)
        cnt = np.bincount(codes).astype(float)
        rng = np.random.default_rng(7)
        w = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=NB)
        bt = (w @ sums) / (w @ cnt)
        rows.append({"comparison": "harmful permitted rate, B2 (scripted, c = 1)", "cluster": cl_name, "estimate_pp": 100 * sums.sum() / cnt.sum(),
                     "ci_lo": 100 * np.percentile(bt, 2.5), "ci_hi": 100 * np.percentile(bt, 97.5), "p": np.nan, "clusters": len(u)})
    zero["scripted_templates"] = b2.scenario_id.nunique()
    zero["scripted_cases"] = b2[["attack_id", "scenario_id"]].drop_duplicates().shape[0]
    return pd.DataFrame(rows), zero


def holdout(rows):
    """Template-clustered interval for the conditional harm rates in the holdout (rerun the holdout attempts)."""
    from secureagent_iot import extended5 as R
    from secureagent_iot.agents import AgenticSystem
    from secureagent_iot.attacks import build_cases, run_attempt
    from secureagent_iot.config import load_config
    from secureagent_iot.simulator import build_registry, build_templates, generate_population
    cfg = load_config(None)
    templates = generate_population(400, R.HOLDOUT_SEED)
    reg = build_registry()
    lookup = templates + [t for t in build_templates() if t.scenario_id == "BEN-01"]
    sysm = AgenticSystem("B2_guarded", cfg, guarded=True)
    recs = []
    for s in cfg["seeds"]:
        for v, t in build_cases(templates, reg, cfg["spec"]):
            r = run_attempt(sysm, v, t, s, cfg, reg, lookup, inject=True)
            if not r["suppression_scored"]:
                recs.append({"scenario_id": t.scenario_id, "lookalike": t.uncertainty == "lookalike",
                             "harm": bool(r["goal_executed"] and r["benign_isolated"] and not r["any_violation"])})
    d = pd.DataFrame(recs)
    for name, sub in (("look-alike attempts", d[d.lookalike]), ("other attempts", d[~d.lookalike]), ("all attempts", d)):
        codes, u = pd.factorize(sub.scenario_id)
        sums = np.bincount(codes, weights=sub.harm.astype(float).values)
        cnt = np.bincount(codes).astype(float)
        rng = np.random.default_rng(11)
        w = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=NB)
        bt = (w @ sums) / (w @ cnt)
        rows.append({"comparison": f"holdout harmful permitted rate, {name}", "cluster": "template", "estimate_pp": 100 * sums.sum() / cnt.sum(),
                     "ci_lo": 100 * np.percentile(bt, 2.5), "ci_hi": 100 * np.percentile(bt, 97.5), "p": np.nan, "clusters": len(u),
                     "attempts": len(sub)})
    return d.scenario_id.nunique()


if __name__ == "__main__":
    s_rows, zero = scripted()
    s_rows = s_rows.to_dict("records")
    zero["holdout_templates"] = holdout(s_rows)
    pd.DataFrame(s_rows).assign(evidence_class="testbed-synthetic; template-clustered reanalysis").to_csv(T / "template_cluster_scripted.csv", index=False)
    res, z2 = llm()
    zero.update(z2)
    res.to_csv(T / "template_cluster_llm.csv", index=False)
    zb = [{"setting": "scripted catalogue, B2", "independent_unit": "template", "n": zero["scripted_templates"], "events": 0,
           "upper_95_one_sided": zero_bound(zero["scripted_templates"])},
          {"setting": "scripted catalogue, B2", "independent_unit": "(variant, template)", "n": zero["scripted_cases"], "events": 0,
           "upper_95_one_sided": zero_bound(zero["scripted_cases"])},
          {"setting": "guarded LLM runs, four models", "independent_unit": "template", "n": zero["llm_templates"], "events": zero["llm_violations"],
           "upper_95_one_sided": zero_bound(zero["llm_templates"])},
          {"setting": "generated holdout, B2", "independent_unit": "template", "n": zero["holdout_templates"], "events": 0,
           "upper_95_one_sided": zero_bound(zero["holdout_templates"])}]
    pd.DataFrame(zb).to_csv(T / "template_cluster_zero_bounds.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(s_rows).round(3).to_string()); print(res.drop(columns="evidence_class").round(3).to_string()); print(pd.DataFrame(zb))
