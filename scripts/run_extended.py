#!/usr/bin/env python
"""Extended evaluations E1-E23 (see secureagent_iot/extended*.py). TESTBED-SYNTHETIC, scripted agents, no LLM."""
import argparse
import time
from pathlib import Path

import pandas as pd

from secureagent_iot import extended as X
from secureagent_iot import extended2 as Y
from secureagent_iot import extended3 as Z
from secureagent_iot.config import load_config, set_global_seed
from secureagent_iot.evaluation import save


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--only", default="", help="comma-separated subset of: population,profiles,quality,thresholds,mutation,queue,scale")
    ap.add_argument("--quick", action="store_true", help="smaller sizes (for tests)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    seeds = list(cfg["seeds"])
    set_global_seed(seeds[0])
    out, T = Path(a.out), Path(a.out).parent / "tables"
    ALL = ["population", "profiles", "quality", "thresholds", "mutation", "queue", "scale", "patterns", "campaign", "seeds",
           "breakdown", "coalesce", "audit", "trust", "repeat", "matrix", "formal", "adaptive", "harmbound", "reviewer"]
    want = set(filter(None, a.only.split(","))) or set(ALL)
    q = a.quick

    def step(name, fn):
        if name in want:
            t0 = time.time(); fn(); print(f"[testbed-synthetic] {name}: done in {time.time() - t0:.1f}s")

    def population():
        df, summ, mix = X.population_replication(cfg, n=200 if q else 2000, seeds=tuple(seeds[:2] if q else seeds))
        save(df, out / "ext_population_runs.csv"); save(summ, T / "ext_population_summary.csv"); save(mix, T / "ext_population_mix.csv")

    def profiles():
        c, at = X.agent_profiles(cfg, seeds[:1] if q else seeds)
        save(c, T / "ext_profiles_clean.csv"); save(at, T / "ext_profiles_attack.csv")

    def quality():
        save(X.detector_quality_sweep(cfg, n=100 if q else 600, seeds=tuple(seeds[:1] if q else seeds[:3])), T / "ext_detector_quality.csv")

    def thresholds():
        save(X.threshold_sensitivity(cfg, n=100 if q else 600, seeds=tuple(seeds[:1] if q else seeds[:3])), T / "ext_threshold_sensitivity.csv")

    def mutation():
        m = X.mutation_testing(cfg, seed=42)
        save(m, out / "ext_mutants.csv"); save(X.summarize_mutation(m), T / "ext_mutation_summary.csv")

    def queue():
        clean, att = pd.read_csv(out / "response_runs.csv"), pd.read_csv(out / "attack_runs.csv")
        save(X.review_queue_simulation(clean, att, reps=30 if q else 500), T / "ext_review_queue_sim.csv")

    def scale():
        save(X.scalability(cfg, registry_sizes=(100, 1000) if q else (100, 1_000, 10_000, 100_000),
                           n_decisions=2000 if q else 20_000, audit_sizes=(1000,) if q else (1_000, 10_000, 100_000)),
             T / "ext_scalability.csv")

    def patterns():
        summ, clean, atk = Y.design_pattern_comparison(cfg, seeds[:1] if q else seeds)
        save(summ, T / "ext_patterns.csv")
        expo, dist = Y.exposure_by_pattern(clean, atk, reps=20 if q else 300)
        save(expo, T / "ext_exposure.csv"); save(dist, T / "ext_exposure_cdf.csv")

    def campaign():
        sweep = Y.filter_recall_sweep(cfg, seeds[:1] if q else seeds)
        save(sweep, T / "ext_filter_recall.csv")
        cs = pd.read_csv(T / "attack_compliance_sweep.csv").set_index(["compliance", "system"])
        ours = {f"B1 unguarded, compliance {c}": float(cs.loc[(c, "B1_unguarded")].exec_asr) for c in (0.25, 1.0)}
        fr = sweep[sweep.system == "P4_agent_filter"].set_index("recall").exec_asr
        ours.update({f"B1 + injection filter, recall {r}": float(fr.loc[r]) for r in (0.9, 0.99)})
        ours["B2 guarded (0 observed; exact 95% upper bound)"] = 1 - 0.05 ** (1 / 224)
        d1 = Y.campaign_risk(ours); d1["source"] = "this testbed"
        d2 = Y.campaign_risk({lab: rate for lab, rate, _ in Y.REPORTED_PRIOR}); d2["source"] = "reported in prior work (different setting)"
        save(pd.concat([d1, d2], ignore_index=True), T / "ext_campaign_risk.csv")

    def seeds_conv():
        save(Y.seed_convergence(cfg, 6 if q else 30), T / "ext_seed_convergence.csv")

    def breakdown():
        runs, atk = pd.read_csv(out / "response_runs.csv"), pd.read_csv(out / "attack_runs.csv")
        save(Y.family_breakdown(runs), T / "ext_family_breakdown.csv"); save(Y.reason_codes(runs, atk), T / "ext_reason_codes.csv")

    def coalesce():
        clean, att = pd.read_csv(out / "response_runs.csv"), pd.read_csv(out / "attack_runs.csv")
        save(Y.coalescing_vs_compromised(clean, att, reps=20 if q else 300), T / "ext_coalescing.csv")

    def audit():
        save(Y.audit_tamper(200 if q else 1000, 30 if q else 300), T / "ext_audit_tamper.csv")

    def trust():
        save(Y.trust_assumption_stress(cfg, seeds[:1] if q else seeds), T / "ext_trust_stress.csv")

    def repeat():
        r, curve = Z.repeated_isolation(cfg, seeds[:1] if q else seeds)
        save(r, T / "ext_repeat_isolation.csv"); save(curve, T / "ext_repeat_isolation_curve.csv")
        aruns = pd.read_csv(out / "attack_runs.csv")
        save(Z.headline_endpoints(aruns, r), T / "ext_headline_endpoints.csv")
        cat, crit = Z.residual_breakdown(aruns)
        save(cat, T / "ext_residual_by_category.csv"); save(crit, T / "ext_residual_by_criticality.csv")

    def matrix():
        save(Z.decision_matrix(cfg), T / "ext_decision_matrix.csv")

    def adaptive():
        save(Z.adaptive_campaigns(cfg, seeds[:1] if q else seeds), T / "ext_adaptive_campaigns.csv")

    def reviewer():
        clean, att = pd.read_csv(out / "response_runs.csv"), pd.read_csv(out / "attack_runs.csv")
        save(Z.reviewer_sensitivity(clean, att, reps=20 if q else 200), T / "ext_reviewer_sensitivity.csv")

    def harmbound():
        from secureagent_iot import harmbound as HB
        bounds, rnd = HB.harm_bounds(cfg, n_random=60 if q else 500)
        save(bounds, T / "ext_harm_bound.csv"); save(rnd, T / "ext_harm_bound_random.csv")
        save(HB.smt_tile_lemma(cfg), T / "ext_harm_bound_smt.csv")
        save(HB.adaptive_vs_bound(cfg, pd.read_csv(T / "ext_adaptive_campaigns.csv")), T / "ext_harm_bound_adaptive.csv")

    def formal():
        proofs, diff = Z.formal_results(cfg, 20000 if q else 100000)
        save(proofs, T / "ext_formal_proofs.csv"); save(diff, T / "ext_formal_differential.csv")

    for name, fn in (("adaptive", adaptive), ("harmbound", harmbound), ("reviewer", reviewer), ("formal", formal), ("repeat", repeat), ("matrix", matrix), ("patterns", patterns), ("campaign", campaign), ("seeds", seeds_conv), ("breakdown", breakdown),
                     ("coalesce", coalesce), ("audit", audit), ("trust", trust)):
        step(name, fn)
    for name, fn in (("population", population), ("profiles", profiles), ("quality", quality), ("thresholds", thresholds),
                     ("mutation", mutation), ("queue", queue), ("scale", scale)):
        step(name, fn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
