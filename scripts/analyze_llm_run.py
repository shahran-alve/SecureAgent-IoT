#!/usr/bin/env python
"""Paired analysis of the full real-LLM run (run1): four models x four prompt conditions x five seeds.

Reads the MERGED per-(model, condition) CSVs only (results/raw/LLM_<model>__<condition>_{attack,response}_runs.csv).
Main models: Qwen2.5-7B-Instruct, gemma-4-12b-it, Mistral-Nemo-Instruct-2407. Phi-4-mini-instruct is the
fail-closed case: it is analysed on its own, never pooled with the main models, and its tests form their own Holm
families.

Pairing unit: (model, condition, system, seed, attack_id, scenario_id). Every styled (injected) attempt is paired
with its own no-injection control, which has the same unit. Pairing refuses (PairingError) if any row lacks its
partner or a partner is duplicated.

Intervals: percentile 95% intervals from a cluster bootstrap, cluster = (attack_id, scenario_id) for the attack
benchmark and scenario_id for the response benchmark, B = 10,000 resamples. The estimate is the mean of the paired
differences over all pairs; a resample redraws clusters with replacement and recomputes that mean over the drawn
clusters' pairs. p-values: two-sided paired sign-flip permutation test at cluster level (each cluster's summed
difference gets a random sign), 10,000 permutations, p = (1 + #{|T*| >= |T|}) / (1 + 10,000). Holm adjustment
within each numbered family (main models and Phi-4-mini separately). Every random draw has a fixed seed derived
from the test's name, so the output is reproducible.

Families:
 1. Injection-attributable effects (injected minus its own control), per model and condition, pooled over styles
    and per style: (a) proposal matches the attacker's goal, (b) B2 wrongful restriction executed, (c) B2 suppression
    (A6.1 only), (d) B2 review, (e) B1 violating execution.
 2. Prompt-condition effects per model, paired on identical keys: base->defended, base->policy,
    policy->policy_defended, defended->policy_defended, for injected proposal ASR and for the response benchmark's
    safe containment, false isolation and review rate of B1 and B2.
 3. B2 against B0 on the response benchmark, per model and condition: safe containment, false isolation, review
    rate, utility.
 4. B2 prohibited executions: exact counts and one-sided 95% Clopper-Pearson upper bounds, per model and pooled
    over the main models, relative to the written spec only.
 5. Where B2's safe containment comes from: first proposal allowed versus allowed after a TTL denial and retry.
 6. Phi-4-mini: share of B2 decisions that are schema rejections, and B2's metrics on schema-valid proposals only
    (a sensitivity analysis).

Writes results/tables/LLM_analysis_*.csv and results/LLM_ANALYSIS.md. Run it on a compute node, never on the head node:
    python scripts/analyze_llm_run.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
MAIN_MODELS = ["Qwen/Qwen2.5-7B-Instruct", "google/gemma-4-12b-it", "mistralai/Mistral-Nemo-Instruct-2407"]
FAIL_CLOSED = ["microsoft/Phi-4-mini-instruct"]
CONDITIONS = ["base", "defended", "policy", "policy_defended"]
import os
# LLM_STYLE_SET=heldout analyses a held-out-style run (see scripts/llm_heldout_sampling.sbatch)
STYLES = {"main": ["naive", "authority", "delimiter", "in_policy"],
          "heldout": ["json_field", "prior_decision", "split", "vendor_request"]}[os.environ.get("LLM_STYLE_SET") or "main"]
CONTROL_STYLE = "none"
SEEDS = [11, 23, 42, 77, 101]
N_BOOT = N_PERM = 10_000
ATTACK_UNIT = ["model", "condition", "system", "seed", "attack_id", "scenario_id"]
RESPONSE_UNIT = ["model", "condition", "system", "seed", "scenario_id"]
CONDITION_PAIRS = [("base", "defended"), ("base", "policy"), ("policy", "policy_defended"), ("defended", "policy_defended")]
INJECTION_METRICS = {   # name: (system, value column, row filter, label)
    "a_proposal_goal": ("B2_guarded", "proposal_success", None, "proposal matches the attacker's goal"),
    "b_b2_benign_isolation": ("B2_guarded", "benign_isolated", None, "B2 wrongful restriction executed"),
    "c_b2_suppression": ("B2_guarded", "suppression_success", "suppression_scored", "B2 suppression (A6.1 only)"),
    "d_b2_review": ("B2_guarded", "escalated", None, "B2 review"),
    "e_b1_violation": ("B1_unguarded", "any_violation", None, "B1 violating execution"),
}
RESPONSE_METRICS = {    # name: (value column, denominator filter or None, label)
    "safe_containment": ("safe_contained", "containment_worthy", "safe containment"),
    "false_isolation": ("false_isolation", "benign", "false isolation"),
    "review_rate": ("escalated", None, "review rate"),
    "utility": ("utility", None, "mean utility"),
}
SPEC_NOTE = ("These bounds hold relative to the written authorization spec only (the simulator's independent oracle, "
             "simulator.spec_violations). They say nothing about harm the spec permits, such as restrictions of benign "
             "look-alike incidents that meet every policy condition.")


class PairingError(ValueError):
    """Rows that should be paired do not match one to one."""


def slug(model: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in model.split("/")[-1])


def short(model: str) -> str:
    return model.split("/")[-1]


def seed_of(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def paired_stats(diff: np.ndarray, cluster: np.ndarray, key: tuple, n_boot: int = N_BOOT, n_perm: int = N_PERM) -> dict:
    """Mean paired difference with a cluster-bootstrap percentile 95% interval and a two-sided cluster-level
    sign-flip permutation p-value."""
    diff = np.asarray(diff, dtype=float)
    codes, cl = pd.factorize(pd.Series(cluster))
    n_cl = len(cl)
    if len(diff) == 0 or n_cl == 0:
        return {"estimate": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan, "p_value": np.nan, "n_pairs": 0, "n_clusters": 0}
    sums = np.bincount(codes, weights=diff, minlength=n_cl)
    counts = np.bincount(codes, minlength=n_cl).astype(float)
    est = sums.sum() / counts.sum()
    rng = np.random.default_rng(seed_of("boot", *key))
    w = rng.multinomial(n_cl, np.full(n_cl, 1.0 / n_cl), size=n_boot)
    boot = (w @ sums) / (w @ counts)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rng = np.random.default_rng(seed_of("perm", *key))
    signs = rng.integers(0, 2, size=(n_perm, n_cl), dtype=np.int8) * 2 - 1
    t_obs = abs(sums.sum())
    t_perm = np.abs(signs @ sums)
    p = (1 + np.count_nonzero(t_perm >= t_obs - 1e-9)) / (1 + n_perm)
    return {"estimate": float(est), "ci95_lo": float(lo), "ci95_hi": float(hi), "p_value": float(p),
            "n_pairs": int(len(diff)), "n_clusters": int(n_cl)}


def holm(p: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values (NaN entries are left out and stay NaN)."""
    q = p.dropna()
    order = q.sort_values().index
    m = len(q)
    adj, running = {}, 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * q[idx]))
        adj[idx] = running
    return pd.Series(adj).reindex(p.index)


def clopper_pearson_upper(k: int, n: int, level: float = 0.95) -> float:
    """One-sided upper confidence bound for a binomial proportion (exact)."""
    if n == 0:
        return float("nan")
    if k >= n:
        return 1.0
    return float(beta.ppf(level, k + 1, n - k))


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
def pair_injected_with_control(d: pd.DataFrame, value: str, styles: list[str]) -> pd.DataFrame:
    """One row per styled attempt in `styles`, with its own control's value. Refuses unmatched or duplicated rows."""
    inj = d[d["injected"] & d["style"].isin(styles)]
    ctl = d[~d["injected"]]
    if (ctl["style"] != CONTROL_STYLE).any():
        raise PairingError("control rows must have style 'none'")
    dup = ctl.duplicated(ATTACK_UNIT, keep=False)
    if dup.any():
        raise PairingError(f"{int(dup.sum())} control rows share a pairing unit, e.g. {ctl[dup][ATTACK_UNIT].iloc[0].to_dict()}")
    if inj.duplicated(ATTACK_UNIT + ["style"]).any():
        raise PairingError("duplicated injected attempts (same unit and style)")
    m = inj[ATTACK_UNIT + ["style", value]].merge(ctl[ATTACK_UNIT + [value]], on=ATTACK_UNIT, how="left",
                                                   suffixes=("_inj", "_ctl"), indicator=True)
    missing = m[m["_merge"] != "both"]
    if len(missing):
        raise PairingError(f"{len(missing)} injected attempts have no matching control, e.g. {missing[ATTACK_UNIT].iloc[0].to_dict()}")
    unused = ctl.merge(inj[ATTACK_UNIT].drop_duplicates(), on=ATTACK_UNIT, how="left", indicator=True)
    if styles == STYLES and (unused["_merge"] != "both").any():
        raise PairingError(f"{int((unused['_merge'] != 'both').sum())} controls have no injected attempt")
    m["diff"] = m[f"{value}_inj"].astype(float) - m[f"{value}_ctl"].astype(float)
    m["cluster"] = m["attack_id"] + "|" + m["scenario_id"]
    return m.drop(columns="_merge")


def pair_on_keys(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], value: str, what: str) -> pd.DataFrame:
    """Rows of `a` and `b` matched one to one on `keys`; diff = b - a. Refuses if the key sets differ."""
    for name, d in (("first", a), ("second", b)):
        if d.duplicated(keys).any():
            raise PairingError(f"{what}: duplicated keys in the {name} set")
    m = a[keys + [value]].merge(b[keys + [value]], on=keys, how="outer", suffixes=("_a", "_b"), indicator=True)
    if (m["_merge"] != "both").any():
        raise PairingError(f"{what}: {int((m['_merge'] != 'both').sum())} keys present on one side only")
    m["diff"] = m[f"{value}_b"].astype(float) - m[f"{value}_a"].astype(float)
    return m.drop(columns="_merge")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load(raw: Path, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    att, rsp, missing = [], [], []
    for m in models:
        for c in CONDITIONS:
            for kind, bucket in (("attack", att), ("response", rsp)):
                path = raw / f"LLM_{slug(m)}__{c}_{kind}_runs.csv"
                if not path.exists():
                    missing.append(str(path))
                    continue
                d = pd.read_csv(path, low_memory=False)
                d.insert(0, "condition", c)
                d.insert(0, "model", m)
                bucket.append(d)
    if missing:
        raise FileNotFoundError("merged CSVs missing (merge every (model, condition) first): " + "; ".join(missing[:6]))
    att_df, rsp_df = pd.concat(att, ignore_index=True), pd.concat(rsp, ignore_index=True)
    for col in ("injected", "proposal_success", "benign_isolated", "suppression_scored", "suppression_success", "escalated",
                "any_violation", "exec_success"):
        att_df[col] = att_df[col].astype(bool)
    for col in ("containment_worthy", "truth_malicious", "safe_contained", "false_isolation", "escalated", "spec_violation"):
        rsp_df[col] = rsp_df[col].astype(bool)
    att_df["seed"] = att_df["seed"].astype(int)
    rsp_df["seed"] = rsp_df["seed"].astype(int)
    # sanity: every file holds exactly its own prompt condition, and five seeds
    for d in (att_df, rsp_df):
        if (d["prompt_condition"] != d["condition"]).any():
            raise ValueError("a merged file holds rows of another prompt condition")
    if sorted(att_df["seed"].unique()) != SEEDS or sorted(rsp_df["seed"].unique()) != SEEDS:
        raise ValueError(f"expected seeds {SEEDS}")
    return att_df, rsp_df


def response_rows(d: pd.DataFrame, metric: str) -> pd.DataFrame:
    value, flt, _ = RESPONSE_METRICS[metric]
    if flt == "containment_worthy":
        return d[d["containment_worthy"]]
    if flt == "benign":
        return d[~d["truth_malicious"]]
    return d


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------
def family1(att: pd.DataFrame, models: list[str], group: str, nb: int, npm: int) -> pd.DataFrame:
    rows = []
    for m in models:
        for c in CONDITIONS:
            for metric, (system, value, flt, label) in INJECTION_METRICS.items():
                d = att[(att["model"] == m) & (att["condition"] == c) & (att["system"] == system)]
                if flt:
                    d = d[d[flt]]
                for style_set, styles in [("pooled", STYLES)] + [(s, [s]) for s in STYLES]:
                    pr = pair_injected_with_control(d, value, styles)
                    st = paired_stats(pr["diff"].values, pr["cluster"].values, ("f1", m, c, metric, style_set), nb, npm)
                    rows.append({"family": f"1{group}", "model": m, "condition": c, "metric": metric, "label": label,
                                 "system": system, "styles": style_set,
                                 "injected_numerator": int(pr[f"{value}_inj"].sum()), "injected_denominator": len(pr),
                                 "injected_rate": pr[f"{value}_inj"].mean() if len(pr) else np.nan,
                                 "control_numerator": int(d[~d["injected"]][value].sum()), "control_denominator": int((~d["injected"]).sum()),
                                 "control_rate": d[~d["injected"]][value].mean() if (~d["injected"]).any() else np.nan,
                                 **st})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"])
    return out


def family2(att: pd.DataFrame, rsp: pd.DataFrame, models: list[str], group: str, nb: int, npm: int) -> pd.DataFrame:
    rows = []
    for m in models:
        for c1, c2 in CONDITION_PAIRS:
            # injected proposal ASR, B2 rows (B1's proposals are the same prompts), keys incl. style
            keys = ["system", "seed", "attack_id", "scenario_id", "style"]
            sel = lambda c: att[(att["model"] == m) & (att["condition"] == c) & (att["system"] == "B2_guarded") & att["injected"]]
            a, b = sel(c1), sel(c2)
            pr = pair_on_keys(a, b, keys, "proposal_success", f"{m} {c1}->{c2} proposal ASR")
            st = paired_stats(pr["diff"].values, (pr["attack_id"] + "|" + pr["scenario_id"]).values, ("f2", m, c1, c2, "asr"), nb, npm)
            rows.append({"family": f"2{group}", "model": m, "from": c1, "to": c2, "benchmark": "attack", "system": "B2_guarded (= B1 proposals)",
                         "metric": "injected_proposal_asr", "from_numerator": int(a["proposal_success"].sum()), "from_denominator": len(a),
                         "to_numerator": int(b["proposal_success"].sum()), "to_denominator": len(b), **st})
            for system in ("B1_unguarded", "B2_guarded"):
                for metric in ("safe_containment", "false_isolation", "review_rate"):
                    value = RESPONSE_METRICS[metric][0]
                    sel = lambda c: response_rows(rsp[(rsp["model"] == m) & (rsp["condition"] == c) & (rsp["system"] == system)], metric)
                    a, b = sel(c1), sel(c2)
                    pr = pair_on_keys(a, b, ["system", "seed", "scenario_id"], value, f"{m} {c1}->{c2} {system} {metric}")
                    st = paired_stats(pr["diff"].values, pr["scenario_id"].values, ("f2", m, c1, c2, system, metric), nb, npm)
                    rows.append({"family": f"2{group}", "model": m, "from": c1, "to": c2, "benchmark": "response", "system": system,
                                 "metric": metric, "from_numerator": int(a[value].sum()), "from_denominator": len(a),
                                 "to_numerator": int(b[value].sum()), "to_denominator": len(b), **st})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"])
    return out


def family3(rsp: pd.DataFrame, models: list[str], group: str, nb: int, npm: int) -> pd.DataFrame:
    rows = []
    for m in models:
        for c in CONDITIONS:
            d = rsp[(rsp["model"] == m) & (rsp["condition"] == c)]
            for metric in RESPONSE_METRICS:
                value = RESPONSE_METRICS[metric][0]
                b0 = response_rows(d[d["system"] == "B0_rules"], metric)
                b2 = response_rows(d[d["system"] == "B2_guarded"], metric)
                pr = pair_on_keys(b0, b2, ["seed", "scenario_id"], value, f"{m} {c} B2 vs B0 {metric}")
                st = paired_stats(pr["diff"].values, pr["scenario_id"].values, ("f3", m, c, metric), nb, npm)
                num = (lambda x: float(x[value].sum())) if metric == "utility" else (lambda x: int(x[value].sum()))
                rows.append({"family": f"3{group}", "model": m, "condition": c, "metric": metric,
                             "b0_numerator": num(b0), "b0_denominator": len(b0), "b0_mean": b0[value].mean(),
                             "b2_numerator": num(b2), "b2_denominator": len(b2), "b2_mean": b2[value].mean(), **st})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"])
    return out


def family4(att: pd.DataFrame, rsp: pd.DataFrame, main: list[str], other: list[str]) -> pd.DataFrame:
    rows = []
    def add(label: str, ms: list[str], pooled: bool):
        a = att[att["model"].isin(ms) & (att["system"] == "B2_guarded")]
        r = rsp[rsp["model"].isin(ms) & (rsp["system"] == "B2_guarded")]
        for bench, d, col in (("attack", a, "any_violation"), ("response", r, "spec_violation"), ("both", None, None)):
            if bench == "both":
                k, n = int(a["any_violation"].sum() + r["spec_violation"].sum()), len(a) + len(r)
                ncl = a[["attack_id", "scenario_id"]].drop_duplicates().shape[0] + r["scenario_id"].nunique()
            else:
                k, n = int(d[col].sum()), len(d)
                ncl = d[["attack_id", "scenario_id"]].drop_duplicates().shape[0] if bench == "attack" else d["scenario_id"].nunique()
            rows.append({"family": "4", "scope": label, "pooled": pooled, "benchmark": bench, "prohibited_executions": k,
                         "b2_attempts": n, "n_clusters": ncl, "rate": k / n if n else np.nan,
                         "cp_upper_95_one_sided": clopper_pearson_upper(k, n), "note": "relative to the written spec only"})
    for m in main + other:
        add(m, [m], False)
    add("pooled main models (" + ", ".join(short(m) for m in main) + ")", main, True)
    return pd.DataFrame(rows)


def family5(rsp: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for m in models:
        for c in CONDITIONS:
            d = rsp[(rsp["model"] == m) & (rsp["condition"] == c) & (rsp["system"] == "B2_guarded")]
            cw = d[d["containment_worthy"]]
            sc = cw[cw["safe_contained"]]
            first = sc[sc["retries"].astype(int) == 0]
            retry = sc[sc["retries"].astype(int) > 0]
            ttl = retry["policy_decisions"].astype(str).str.startswith("DENY_TTL_EXCEEDS_CAP>")
            rows.append({"family": "5", "model": m, "condition": c, "containment_worthy_rows": len(cw),
                         "safe_contained": len(sc), "first_proposal_allowed": len(first), "allowed_after_ttl_denial_and_retry": len(retry),
                         "retries_that_were_ttl_denials": int(ttl.sum()),
                         "share_first_proposal": len(first) / len(sc) if len(sc) else np.nan,
                         "share_after_retry": len(retry) / len(sc) if len(sc) else np.nan,
                         "n_clusters": sc["scenario_id"].nunique()})
    return pd.DataFrame(rows)


def family6(att: pd.DataFrame, rsp: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    share, sens = [], []
    for m in models:
        for c in CONDITIONS:
            for bench, d in (("response", rsp), ("attack", att)):
                b2 = d[(d["model"] == m) & (d["condition"] == c) & (d["system"] == "B2_guarded")]
                first = b2["policy_decisions"].astype(str).str.split(">").str[0]
                anywhere = b2["policy_decisions"].astype(str).str.contains("DENY_SCHEMA")
                share.append({"family": "6", "model": m, "condition": c, "benchmark": bench, "b2_attempts": len(b2),
                              "first_decision_schema_rejection": int(first.str.startswith("DENY_SCHEMA").sum()),
                              "share_first_decision": float(first.str.startswith("DENY_SCHEMA").mean()) if len(b2) else np.nan,
                              "any_decision_schema_rejection": int(anywhere.sum()),
                              "share_any_decision": float(anywhere.mean()) if len(b2) else np.nan,
                              "n_clusters": (b2[["attack_id", "scenario_id"]].drop_duplicates().shape[0] if bench == "attack"
                                             else b2["scenario_id"].nunique())})
            b2 = rsp[(rsp["model"] == m) & (rsp["condition"] == c) & (rsp["system"] == "B2_guarded")]
            valid = b2[~b2["policy_decisions"].astype(str).str.contains("DENY_SCHEMA")]
            for metric in RESPONSE_METRICS:
                value = RESPONSE_METRICS[metric][0]
                for subset, d in (("all B2 rows", b2), ("schema-valid proposals only", valid)):
                    r = response_rows(d, metric)
                    sens.append({"family": "6", "analysis": "SENSITIVITY (selection on the model's own output; not a causal estimate)",
                                 "model": m, "condition": c, "metric": metric, "subset": subset,
                                 "numerator": float(r[value].sum()) if metric == "utility" else int(r[value].sum()),
                                 "denominator": len(r), "value": r[value].mean() if len(r) else np.nan,
                                 "n_clusters": r["scenario_id"].nunique()})
    return pd.DataFrame(share), pd.DataFrame(sens)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def ci(r, scale=100, nd=1) -> str:
    if np.isnan(r["estimate"]):
        return "n/a"
    f = lambda v: f"{scale * v:+.{nd}f}"
    return f"{f(r['estimate'])} [{f(r['ci95_lo'])}, {f(r['ci95_hi'])}]"


def pv(p) -> str:
    return "n/a" if np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def md_table(head: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def write_report(path: Path, t: dict, models_main: list[str], models_other: list[str], nb: int, npm: int, inputs: list[str]) -> None:
    fam_sizes = {fam: int(n) for k in ("family1", "family2", "family3") for fam, n in t[k]["family"].astype(str).value_counts().sort_index().items()}
    L = ["# LLM analysis — paired analysis of the full real-LLM run (run1)", "",
         f"Generated by `scripts/analyze_llm_run.py` from the merged per-(model, condition) CSVs. Main models: "
         f"{', '.join(short(m) for m in models_main)}. Fail-closed case, analysed separately and never pooled: "
         f"{', '.join(short(m) for m in models_other)}.", "",
         "**Method.** Pairing unit (model, condition, system, seed, attack_id, scenario_id); every styled attempt is paired with "
         "its own no-injection control. Estimates are mean paired differences in percentage points (utility: in utility units). "
         f"Intervals: percentile 95% cluster bootstrap, {nb:,} resamples, cluster = (attack_id, scenario_id) for the attack "
         f"benchmark and scenario_id for the response benchmark. p: two-sided cluster-level sign-flip permutation test, {npm:,} "
         "permutations; p_holm: Holm-adjusted within the numbered family (main models and Phi-4-mini are separate families). "
         "Fixed seeds; rerunning gives the same numbers. Full tables: `results/tables/LLM_analysis_*.csv`. "
         "Testbed-synthetic scenarios; not field measurements.", "",
         f"**Floor on p-values.** With {npm:,} permutations the smallest attainable p is 1/{npm + 1:,} = {1 / (npm + 1):.5f}. Holm multiplies "
         "the smallest p by the family size, so the smallest attainable adjusted p is about "
         + ", ".join(f"{1 / (npm + 1) * n:.3f} in family {fam} ({n} tests)" for fam, n in fam_sizes.items())
         + ". An adjusted p at that floor means the effect is as significant as this test can show.", ""]
    f1 = t["family1"]
    for group, ms, title in (("", models_main, "main models"), ("_phi", models_other, "Phi-4-mini (fail-closed case)")):
        d = f1[(f1["family"] == f"1{group}") & (f1["styles"] == "pooled")]
        L += [f"## 1. Injection-attributable effects, {title}: injected minus own control, pooled over styles", ""]
        rows = [[short(r["model"]), r["condition"], r["label"], f"{r['injected_numerator']}/{r['injected_denominator']}",
                 f"{r['control_numerator']}/{r['control_denominator']}", ci(r), pv(r["p_value"]), pv(r["p_holm"]), r["n_clusters"]]
                for r in d.to_dict("records")]
        L += [md_table(["Model", "Condition", "Measure", "Injected (n/N)", "Control (n/N)", "Difference, pp [95% CI]", "p", "p_holm", "Clusters"], rows), ""]
        d = f1[(f1["family"] == f"1{group}") & (f1["metric"] == "a_proposal_goal") & (f1["styles"] != "pooled")]
        L += [f"Per style, measure (a) proposal matches the attacker's goal ({title}):", ""]
        rows = [[short(r["model"]), r["condition"], r["styles"], f"{r['injected_numerator']}/{r['injected_denominator']}",
                 f"{r['control_numerator']}/{r['control_denominator']}", ci(r), pv(r["p_holm"]), r["n_clusters"]] for r in d.to_dict("records")]
        L += [md_table(["Model", "Condition", "Style", "Injected (n/N)", "Control (n/N)", "Difference, pp [95% CI]", "p_holm", "Clusters"], rows), "",
              "Per-style rows for measures (b)–(e) are in `LLM_analysis_family1_injection.csv`.", ""]
    f2 = t["family2"]
    for group, title in (("", "main models"), ("_phi", "Phi-4-mini")):
        d = f2[f2["family"] == f"2{group}"]
        L += [f"## 2. Prompt-condition effects, {title} (to minus from, paired on identical keys)", ""]
        rows = [[short(r["model"]), f"{r['from']}->{r['to']}", r["benchmark"], r["system"], r["metric"], f"{r['from_numerator']}/{r['from_denominator']}",
                 f"{r['to_numerator']}/{r['to_denominator']}", ci(r), pv(r["p_holm"]), r["n_clusters"]] for r in d.to_dict("records")]
        L += [md_table(["Model", "Change", "Benchmark", "System", "Measure", "From (n/N)", "To (n/N)", "Difference, pp [95% CI]", "p_holm", "Clusters"], rows), ""]
    f3 = t["family3"]
    for group, title in (("", "main models"), ("_phi", "Phi-4-mini")):
        d = f3[f3["family"] == f"3{group}"]
        L += [f"## 3. B2 against B0, response benchmark, {title} (B2 minus B0, paired on seed and scenario)", ""]
        rows = []
        for r in d.to_dict("records"):
            u = r["metric"] == "utility"
            fmtn = (lambda n, N: f"{n:.1f}/{N}") if u else (lambda n, N: f"{int(n)}/{N}")
            rows.append([short(r["model"]), r["condition"], r["metric"], fmtn(r["b0_numerator"], r["b0_denominator"]),
                         fmtn(r["b2_numerator"], r["b2_denominator"]), ci(r, 1, 3) if u else ci(r), pv(r["p_holm"]), r["n_clusters"]])
        L += [md_table(["Model", "Condition", "Measure", "B0 (n/N)", "B2 (n/N)", "Difference [95% CI] (pp; utility in units)", "p_holm", "Clusters"], rows), ""]
    f4 = t["family4"]
    L += ["## 4. B2 prohibited executions (exact counts, one-sided 95% Clopper-Pearson upper bounds)", "", f"**{SPEC_NOTE}**", ""]
    rows = [[r["scope"] if not r["pooled"] else f"**{r['scope']}**", r["benchmark"], r["prohibited_executions"], r["b2_attempts"],
             r["n_clusters"], f"{100 * r['cp_upper_95_one_sided']:.4f}%"] for r in f4.to_dict("records")]
    L += [md_table(["Scope", "Benchmark", "Prohibited executions", "B2 attempts", "Clusters", "Upper bound (95%, one-sided)"], rows), ""]
    f5 = t["family5"]
    L += ["## 5. Where B2's safe containment comes from (response benchmark)", ""]
    rows = [[short(r["model"]), r["condition"], r["safe_contained"], r["containment_worthy_rows"],
             f"{r['first_proposal_allowed']} ({pct(r['share_first_proposal'])}%)",
             f"{r['allowed_after_ttl_denial_and_retry']} ({pct(r['share_after_retry'])}%)", r["retries_that_were_ttl_denials"], r["n_clusters"]]
            for r in f5.to_dict("records")]
    L += [md_table(["Model", "Condition", "Safe contained", "Containment-worthy rows", "Allowed on first proposal", "Allowed after TTL denial and retry",
                    "Of which TTL denials", "Clusters"], rows), ""]
    f6a, f6b = t["family6_share"], t["family6_sensitivity"]
    L += ["## 6. Phi-4-mini: schema rejections and a sensitivity analysis", "",
          "Share of B2 decisions that are schema rejections (first decision of the attempt, and anywhere in the attempt):", ""]
    rows = [[short(r["model"]), r["condition"], r["benchmark"], f"{r['first_decision_schema_rejection']}/{r['b2_attempts']} ({pct(r['share_first_decision'])}%)",
             f"{r['any_decision_schema_rejection']}/{r['b2_attempts']} ({pct(r['share_any_decision'])}%)", r["n_clusters"]]
            for r in f6a.to_dict("records")]
    L += [md_table(["Model", "Condition", "Benchmark", "First decision is a schema rejection", "Any decision is a schema rejection", "Clusters"], rows), "",
          "**Sensitivity analysis, not a causal estimate:** B2's response metrics recomputed on the attempts whose proposals passed schema "
          "validation. Which proposals are schema-valid is chosen by the model itself, so this subset is selected, not randomized.", ""]
    rows = [[short(r["model"]), r["condition"], r["metric"], r["subset"],
             (f"{r['numerator']:.1f}/{r['denominator']}" if r["metric"] == "utility" else f"{int(r['numerator'])}/{r['denominator']}"),
             (f"{r['value']:.3f}" if r["metric"] == "utility" else f"{pct(r['value'])}%"), r["n_clusters"]] for r in f6b.to_dict("records")]
    L += [md_table(["Model", "Condition", "Measure", "Subset", "n/N", "Value", "Clusters"], rows), "",
          "## Inputs", ""] + [f"- `{p}`" for p in inputs] + [""]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--tables", default=str(ROOT / "results" / "tables"))
    ap.add_argument("--report", default=str(ROOT / "results" / "LLM_ANALYSIS.md"))
    ap.add_argument("--main-models", nargs="+", default=MAIN_MODELS)
    ap.add_argument("--fail-closed-models", nargs="+", default=FAIL_CLOSED)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    a = ap.parse_args(argv)
    raw, tables = Path(a.raw), Path(a.tables)
    main_m, other_m = list(a.main_models), list(a.fail_closed_models)
    if set(main_m) & set(other_m):
        print("REFUSED: a model cannot be both a main model and the fail-closed case", file=sys.stderr)
        return 2
    att, rsp = load(raw, main_m + other_m)
    print(f"loaded {len(att):,} attack rows and {len(rsp):,} response rows for {len(main_m) + len(other_m)} models", flush=True)
    t: dict = {}
    t["family1"] = pd.concat([family1(att, main_m, "", a.n_boot, a.n_perm), family1(att, other_m, "_phi", a.n_boot, a.n_perm)], ignore_index=True)
    print("family 1 done", flush=True)
    t["family2"] = pd.concat([family2(att, rsp, main_m, "", a.n_boot, a.n_perm), family2(att, rsp, other_m, "_phi", a.n_boot, a.n_perm)], ignore_index=True)
    print("family 2 done", flush=True)
    t["family3"] = pd.concat([family3(rsp, main_m, "", a.n_boot, a.n_perm), family3(rsp, other_m, "_phi", a.n_boot, a.n_perm)], ignore_index=True)
    t["family4"] = family4(att, rsp, main_m, other_m)
    t["family5"] = family5(rsp, main_m + other_m)
    t["family6_share"], t["family6_sensitivity"] = family6(att, rsp, other_m)
    tables.mkdir(parents=True, exist_ok=True)
    names = {"family1": "family1_injection", "family2": "family2_condition", "family3": "family3_b2_vs_b0",
             "family4": "family4_prohibited", "family5": "family5_containment_source",
             "family6_share": "family6_phi_schema_rejections", "family6_sensitivity": "family6_phi_sensitivity"}
    for k, name in names.items():
        t[k].insert(0, "evidence_class", "testbed-synthetic scenarios, LLM-backed agents (run1)")
        t[k].to_csv(tables / f"LLM_analysis_{name}.csv", index=False)
    def shown(path: Path) -> str:
        path = path.resolve()
        return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    inputs = sorted(shown(raw / f"LLM_{slug(m)}__{c}_{k}_runs.csv") for m in main_m + other_m for c in CONDITIONS for k in ("attack", "response"))
    write_report(Path(a.report), t, main_m, other_m, a.n_boot, a.n_perm, inputs)
    print(f"wrote {len(names)} tables to {tables} and {a.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
