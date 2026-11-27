#!/usr/bin/env python
"""Figures from result files only. Every figure title carries its evidence class."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, GRID = "#1f2933", "#52606d", "#d9dee4"
C = {"B0_rules": "#7b8794", "B1_unguarded": "#c2410c", "B2_guarded": "#1d4ed8", "B3_single_guarded": "#0f766e"}
LABEL = {"B0_rules": "B0 rules", "B1_unguarded": "B1 unguarded", "B2_guarded": "B2 guarded", "B3_single_guarded": "B3 single-agent guarded",
         "B2_no_provenance": "B2 − provenance", "B2_no_criticality": "B2 − criticality", "B2_no_ttl_cap": "B2 − TTL cap",
         "B2_no_floor": "B2 − escalation floor", "B2_no_budget": "B2 − device budget",
         "B2_no_ttl_no_budget": "B2 − TTL cap − budget"}
TAG = "testbed-synthetic, scripted agents (no LLM)"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10,
                     "axes.titlecolor": INK, "figure.dpi": 150})


def _save(fig, out: Path, name: str):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"wrote {out / name}.png/.pdf")


def architecture(out: Path):
    """Trust-boundary diagram: who can write what, and which component authorizes."""
    RED, BLUE, AMB = "#b91c1c", "#1d4ed8", "#c2410c"
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

    def zone(x, y, w, h, title, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.5", fc=fc, ec=ec, lw=1.0, ls="--"))
        ax.text(x + 1.2, y + h - 1.4, title, ha="left", va="top", fontsize=8, color=ec, weight="bold")

    def box(x, y, w, h, text, fc="white", ec=MUTED, bold=False, fs=7.3):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0", fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, color=MUTED, ls="-", text=None, tx=0, ty=0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, color=color, lw=1.1, ls=ls))
        if text:
            ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, text, fontsize=6.6, color=color, ha="center")

    zone(1, 38, 44, 17, "ATTACKER-CONTROLLED INPUT", "#fdecec", RED)
    box(3, 40, 12, 10, "Device telemetry\nfree text\n(names, logs,\npayloads)", ec=RED)
    box(17, 40, 12, 10, "Retrieved threat\ndescriptions,\nspoofed tool\noutput", ec=RED)
    box(31, 40, 12, 10, "Agent memory\nplanted by an\nearlier incident", ec=RED)

    zone(1, 13, 44, 22, "UNTRUSTED: AGENTS ONLY PROPOSE", "#fff4e6", AMB)
    box(4, 18, 17, 11, "Investigator\n(links evidence)", ec=AMB)
    box(25, 18, 17, 11, "Responder\n(typed proposal:\nincident, device, action,\nduration, evidence refs)", ec=AMB)
    arrow(21.4, 23.5, 24.6, 23.5, AMB)
    for x, x2 in ((9, 11), (23, 30), (37, 36)):
        arrow(x, 39.6, x2, 29.6, RED, "--")

    zone(49, 1, 50, 54, "TRUSTED: STATE, AUTHORIZATION, EXECUTION", "#eaf0ff", BLUE)
    box(51, 40, 14, 10, "Detector alert\np, device,\nevidence IDs\n+ evidence kinds", ec=BLUE)
    box(67, 40, 14, 10, "Asset registry\ncriticality,\nallowlist", ec=BLUE)
    box(83, 40, 14, 10, "Approval store\nbound to incident,\ndevice, action;\nexpires; single use", ec=BLUE)
    box(57, 20, 34, 14, "POLICY ENGINE (default deny)\nschema · agent identity · incident binding · provenance\nevidence independence · allowlist · criticality approval\n300 s cap · per-device budget · thresholds · escalation floor",
        fc="#dbe6ff", ec=BLUE, bold=True, fs=7.0)
    for x in (58, 74, 90):
        arrow(x, 39.6, min(max(x, 62), 86), 34.4, BLUE)
    arrow(42.4, 23.5, 56.6, 25.5, AMB)
    box(51, 4, 14, 10, "Human review\nqueue\n(priority by\ntrusted p)", ec=BLUE)
    box(67, 4, 14, 10, "Network controller\nfixed API,\nexpiring reversible\nrules", ec=BLUE)
    box(83, 4, 14, 10, "Hash-chained\naudit log\n(anchored head)", ec=BLUE)
    arrow(66, 19.6, 59, 14.4, BLUE, text="deny / needs approval", tx=-5.5, ty=0.6)
    arrow(74, 19.6, 74, 14.4, BLUE, text="allow", tx=3.2)
    arrow(84, 19.6, 89, 14.4, MUTED, text="every decision", tx=6.5, ty=0.2)
    box(2, 1.5, 42, 9.5, "The attacker can change what agents read and therefore what they propose.\nA proposal is data, never a command. Nothing on the left can\nwrite trusted state or call the controller.", fc="#f5f7fa", fs=7.0)
    _save(fig, out, "fig1_architecture")


def response(inp: Path, out: Path):
    s = pd.read_csv(inp.parent / "tables" / "response_summary.csv")
    systems = ["B0_rules", "B1_unguarded", "B2_guarded", "B3_single_guarded"]
    metrics = [("safe_containment", "Safe containment\n(of containment-worthy)"), ("false_isolation", "False isolation\n(of benign)"),
               ("unauthorized_action", "Unauthorized action\n(of all incidents)"), ("review_rate", "Review rate\n(of all incidents)")]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.9), sharey=True)
    for ax, (m, title) in zip(axes, metrics):
        d = s[s.metric == m].set_index("system").loc[systems]
        ax.bar(range(len(systems)), d["mean"], color=[C[x] for x in systems], width=0.62)
        ax.errorbar(range(len(systems)), d["mean"], yerr=[d["mean"] - d["ci95_lo"], d["ci95_hi"] - d["mean"]], fmt="none", ecolor=INK, lw=1, capsize=2.5)
        for i, (v, hi) in enumerate(zip(d["mean"], d["ci95_hi"])):
            ax.text(i, hi + 0.03, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(systems))); ax.set_xticklabels(["B0", "B1", "B2", "B3"])
        ax.set_title(title, fontsize=8.5); ax.set_ylim(0, 1.0); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("rate (mean, 95% scenario-bootstrap CI)")
    n = int(s["n_seeds"].max())
    _save(fig, out, "fig3_response_outcomes")


def attacks(inp: Path, out: Path):
    a = pd.read_csv(inp.parent / "tables" / "attack_overall.csv")
    a = a[a.injected].set_index("system")
    order = ["B1_unguarded", "B2_no_provenance", "B2_no_ttl_no_budget", "B2_no_criticality", "B2_no_ttl_cap", "B2_no_budget", "B2_no_floor", "B2_guarded"]
    a = a.loc[order]
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    y = range(len(order))
    ax.barh([i + 0.19 for i in y], a["proposal_asr"], height=0.36, color="#cbd2d9", label="proposal-level (agent adopted attacker goal)")
    ax.barh([i - 0.19 for i in y], a["exec_asr"], height=0.36, color="#b91c1c", label="execution-level (prohibited action executed)")
    ax.errorbar(a["exec_asr"], [i - 0.19 for i in y], xerr=[a["exec_asr"] - a["exec_asr_ci95_lo"], a["exec_asr_ci95_hi"] - a["exec_asr"]], fmt="none", ecolor=INK, lw=1, capsize=2.5)
    for i, (p, e, k, n) in enumerate(zip(a["proposal_asr"], a["exec_asr"], a["exec_successes"], a["exec_denominator"])):
        ax.text(a["exec_asr_ci95_hi"].iloc[i] + 0.02, i - 0.19, f"{e:.3f}  ({int(k):,}/{int(n):,})", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(list(y)); ax.set_yticklabels([LABEL[o] for o in order]); ax.invert_yaxis()
    ax.set_xlim(0, 1.12); ax.set_xlabel("attack success rate"); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    _save(fig, out, "fig4_injection_asr")


def latency(inp: Path, out: Path):
    r = pd.read_csv(inp / "response_runs.csv")
    systems = ["B0_rules", "B1_unguarded", "B2_guarded", "B3_single_guarded"]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    bp = ax.boxplot([r[r.system == s]["decision_latency_us"] for s in systems], showfliers=False, patch_artist=True, widths=0.55,
                    medianprops={"color": INK})
    for patch, s in zip(bp["boxes"], systems):
        patch.set_facecolor(C[s]); patch.set_alpha(0.75); patch.set_edgecolor(INK)
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(["B0", "B1", "B2", "B3"]); ax.set_yscale("log")
    ax.set_ylabel("alert-to-decision latency (µs, log)"); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("Measured orchestration latency, scripted agents only\n(no model inference; token cost not measured)", fontsize=9)
    _save(fig, out, "fig5_latency")


def detection(inp: Path, out: Path):
    for prefix, tag in (("", "real-data"), ("DEMO_", "TOY DATA — SANITY CHECK ONLY, NOT RESEARCH RESULTS")):
        mpath = inp / f"{prefix}detection_metrics.csv"
        if not mpath.exists():
            continue
        m = pd.read_csv(mpath).set_index("model")
        best = "random_forest"
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.1))
        cm = [[m.loc[best, "tn"], m.loc[best, "fp"]], [m.loc[best, "fn"], m.loc[best, "tp"]]]
        a1.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a1.text(j, i, f"{int(cm[i][j])}", ha="center", va="center", color=INK, fontsize=10,
                        bbox={"fc": "white", "ec": "none", "alpha": 0.7, "pad": 1.5})
        a1.set_xticks([0, 1]); a1.set_xticklabels(["pred benign", "pred attack"]); a1.set_yticks([0, 1]); a1.set_yticklabels(["benign", "attack"])
        a1.set_title("Random Forest confusion matrix (holdout)", fontsize=9)
        for name in m.index:
            c = pd.read_csv(inp / f"{prefix}pr_curve_{name}.csv")
            a2.plot(c["recall"], c["precision"], label=f"{name} (AP={m.loc[name, 'pr_auc']:.3f})", lw=1.5)
        a2.set_xlabel("recall"); a2.set_ylabel("precision"); a2.legend(frameon=False, fontsize=7.5); a2.grid(color=GRID, lw=0.6)
        a2.set_title("Precision-recall (attack class)", fontsize=9)
        fig.suptitle(tag, fontsize=10, color="#b91c1c" if prefix else INK, weight="bold", y=1.03)
        _save(fig, out, f"{prefix}fig2_detection")


def extended(inp: Path, out: Path):
    T = inp.parent / "tables"
    if (T / "ext_detector_quality.csv").exists():
        d = pd.read_csv(T / "ext_detector_quality.csv")
        fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.9))
        panels = [("safe_containment", "Safe containment"), ("false_isolation", "False isolation"), ("review_rate", "Review rate")]
        for ax, (m, title) in zip(axes, panels):
            for s_ in ("B0_rules", "B1_unguarded", "B2_guarded"):
                g = d[(d.metric == m) & (d.system == s_)].sort_values("detector_auc")
                ax.plot(g.detector_auc, g["mean"], marker="o", ms=3.5, lw=1.6, color=C[s_], label=LABEL[s_])
            ax.set_title(title, fontsize=9); ax.set_xlabel("detector ROC AUC"); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_ylim(bottom=0)
        axes[0].set_ylabel("rate"); axes[0].legend(frameon=False, fontsize=7)
        ax = axes[3]
        for m, lab, col, ls in (("attack_suppression", "containment suppressed (A6.1)", "#7c3aed", "-"),
                                ("attack_benign_isolation", "benign device isolated", "#b91c1c", "-"),
                                ("attack_exec_asr", "prohibited execution", INK, "--")):
            g = d[(d.metric == m) & (d.system == "B2_guarded")].sort_values("detector_auc")
            ax.plot(g.detector_auc, g["mean"], marker="o", ms=3.5, lw=1.6, color=col, ls=ls, label=lab)
        ax.set_title("B2 under attack", fontsize=9); ax.set_xlabel("detector ROC AUC"); ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_ylim(-0.03, 1.03); ax.legend(frameon=False, fontsize=6.5, loc="lower left", bbox_to_anchor=(0.0, 0.12))
        _save(fig, out, "fig5_detector_quality")
    if (T / "ext_threshold_sensitivity.csv").exists():
        d = pd.read_csv(T / "ext_threshold_sensitivity.csv")
        fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.8))
        cols = [("autonomous_containment_of_malicious", "Autonomous containment\n(of malicious)"), ("review_rate", "Review rate"),
                ("attack_benign_isolation", "Benign isolated under attack"), ("attack_suppression", "Containment suppressed\nunder attack")]
        shades = {1: "#93c5fd", 2: "#1d4ed8", 3: "#1e3a8a"}
        for ax, (cname, title) in zip(axes, cols):
            for me, g in d.groupby("min_evidence"):
                ax.plot(g.p_threshold, g[cname], marker="o", ms=3.5, lw=1.6, color=shades[me], label=f"min. evidence = {me}")
            ax.axvline(0.8, color=MUTED, lw=0.8, ls=":")
            ax.set_title(title, fontsize=8.5); ax.set_xlabel("confidence threshold"); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_ylim(bottom=0)
        axes[0].set_ylabel("rate"); axes[1].legend(frameon=False, fontsize=7, loc="lower left")
        _save(fig, out, "fig6_threshold_sensitivity")
    if (inp / "ext_mutants.csv").exists():
        m = pd.read_csv(inp / "ext_mutants.csv")
        ops = ["relational", "logical", "negate-condition", "flip-allowed", "flip-approval"]
        cat = pd.Series("no detector (no decision changed on the grid)", index=m.index)
        cat[m.killed_by_tests_extended] = "extended tests only"
        cat[m.killed_by_tests] = "initial unit tests only"
        cat[m.killed_by_extended_oracle] = "extended (decision-level) oracle"
        cat[m.killed_by_oracle | m.killed_by_liveness] = "execution oracle / liveness property"
        order = ["execution oracle / liveness property", "extended (decision-level) oracle", "initial unit tests only", "extended tests only",
                 "no detector (no decision changed on the grid)"]
        colors = ["#1d4ed8", "#7c3aed", "#0f766e", "#d97706", "#cbd2d9"]
        tab = pd.crosstab(m.operator, cat).reindex(index=ops, columns=order, fill_value=0)
        fig, ax = plt.subplots(figsize=(6.8, 3.2))
        left = [0] * len(ops)
        for c, col in zip(order, colors):
            ax.barh(range(len(ops)), tab[c], left=left, color=col, label=c, height=0.6, edgecolor="white", lw=0.8)
            for i, (v, l) in enumerate(zip(tab[c], left)):
                if v:
                    ax.text(l + v / 2, i, str(int(v)), ha="center", va="center", fontsize=7.5, color="white" if col != "#cbd2d9" else INK)
            left = [l + v for l, v in zip(left, tab[c])]
        ax.set_yticks(range(len(ops))); ax.set_yticklabels(ops); ax.invert_yaxis(); ax.set_xlabel("single-point mutants of the policy decision function")
        ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=2)
        _save(fig, out, "fig7_mutation")
    if (T / "ext_review_queue_sim.csv").exists():
        d = pd.read_csv(T / "ext_review_queue_sim.csv")
        names = {"fifo": ("FIFO", "#7b8794"), "priority": ("priority by detector probability", "#1d4ed8"),
                 "priority+coalesce": ("priority + per-device coalescing", "#0f766e")}
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
        for k, (lab, col) in names.items():
            g = d[d.discipline == k]
            axes[0].plot(g.attack_fraction, g.malicious_sla_miss, marker="o", ms=4, lw=1.7, color=col, label=lab)
            axes[1].plot(g.attack_fraction, g.wait_p95_min, marker="o", ms=4, lw=1.7, color=col, label=lab)
        axes[0].set_ylabel("malicious escalations not reviewed\nwithin 15 min"); axes[1].set_ylabel("95th-percentile wait (min)")
        for ax in axes:
            ax.set_xlabel("share of incidents carrying injected text"); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_ylim(bottom=0)
        axes[0].legend(frameon=False, fontsize=7)
        _save(fig, out, "fig8_review_queue")


def extended2(inp: Path, out: Path):
    T = inp.parent / "tables"
    import numpy as np
    PAL = ["#7b8794", "#a16207", "#52606d", "#c2410c", "#d97706", "#ea580c", "#1d4ed8", "#0f766e"]
    if (T / "ext_patterns.csv").exists():
        d = pd.read_csv(T / "ext_patterns.csv")
        lab = [x.replace(" (B0)", "").replace(" (B1)", "").replace(" (B1v)", "").replace(" (recall 0.90)", "") for x in d.pattern]
        fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.2), sharey=True)
        y = range(len(d))
        for ax, (col, title, logx) in zip(axes, [("violation_rate_attacked", "Specification-violating\nexecutions under attack (rate)", False),
                                                  ("wrongful_isolation_s_per_100_attacked", "Wrongful isolation under attack\n(device-seconds per 100 incidents)", True),
                                                  ("review_rate_attacked", "Review rate under attack", False)]):
            v = d[col].clip(lower=1 if logx else 0)
            ax.barh(list(y), v, color=PAL, height=0.62)
            for i, (val, raw) in enumerate(zip(v, d[col])):
                ax.text(val * (1.15 if logx else 1) + (0 if logx else 0.02), i, f"{raw:,.0f}" if logx else f"{raw:.3f}", va="center", fontsize=7, color=INK)
            if logx:
                ax.set_xscale("log"); ax.set_xlim(1, 3e6)
            else:
                ax.set_xlim(0, 1.18)
            ax.set_title(title, fontsize=8.5); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        axes[0].set_yticks(list(y)); axes[0].set_yticklabels(lab, fontsize=7.5); axes[0].invert_yaxis()
        _save(fig, out, "fig9_design_patterns")
    if (T / "ext_exposure_cdf.csv").exists():
        d = pd.read_csv(T / "ext_exposure_cdf.csv")
        names = {"P1_advisory_only": "advisory only", "P2_unbounded_rules": "unbounded rules", "B0_rules": "bounded rules (B0)",
                 "B1_unguarded": "unguarded agent (B1)", "P4_agent_filter": "agent + filter", "B2_guarded": "agent + policy (B2)",
                 "P7_filter_policy": "agent + filter + policy"}
        cols = dict(zip(names, ["#7b8794", "#a16207", "#52606d", "#c2410c", "#d97706", "#1d4ed8", "#0f766e"]))
        fig, ax = plt.subplots(figsize=(6.6, 3.1))
        for k, lab in names.items():
            g = d[d.system == k]
            ax.plot(g.minutes.replace(0, 0.5), g.share_responded, marker="o", ms=3, lw=1.6, color=cols[k], label=lab)
        ax.set_xscale("log"); ax.set_xticks([0.5, 1, 5, 15, 60, 240, 480]); ax.set_xticklabels(["0", "1", "5", "15", "60", "240", "480"])
        ax.set_xlabel("minutes since alert"); ax.set_ylabel("share of malicious incidents\nwith a response"); ax.set_ylim(0, 1.02)
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
        _save(fig, out, "fig10_exposure")
    if (T / "ext_campaign_risk.csv").exists():
        d = pd.read_csv(T / "ext_campaign_risk.csv")
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.1), sharey=True)
        for ax, (src, title) in zip(axes, [("this testbed", "Measured in this testbed"), ("reported in prior work (different setting)", "Per-attempt rates reported in prior work")]):
            g = d[d.source == src]
            pal = ["#c2410c", "#fb923c", "#d97706", "#a16207", "#1d4ed8"] if src == "this testbed" else ["#52606d", "#7b8794", "#0f766e", "#7c3aed", "#b91c1c"]
            for (lab, gg), col in zip(g.groupby("label", sort=False), pal):
                ax.plot(gg.attempts, gg.p_at_least_one, lw=1.7, color=col, marker="o", ms=2.5, label=f"{lab} ({gg.per_attempt_rate.iloc[0]:.3f})")
            ax.set_xscale("log"); ax.set_xlabel("injection attempts by the attacker"); ax.set_title(title, fontsize=9)
            ax.yaxis.grid(True, color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=6.3, loc="lower right")
        axes[0].set_ylabel("probability of at least one\nprohibited execution")
        _save(fig, out, "fig11_campaign_risk")
    if (T / "ext_filter_recall.csv").exists():
        d = pd.read_csv(T / "ext_filter_recall.csv")
        fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.7))
        for ax, (col, title) in zip(axes, [("exec_asr", "Prohibited execution"), ("suppression_rate", "Containment suppressed"), ("review_rate_attacked", "Review rate under attack")]):
            for k, lab, colr in (("P4_agent_filter", "agent + filter", "#d97706"), ("P7_filter_policy", "agent + filter + policy", "#0f766e")):
                g = d[d.system == k]
                ax.plot(g.recall, g[col], marker="o", ms=3.5, lw=1.6, color=colr, label=lab)
            ax.set_title(title, fontsize=9); ax.set_xlabel("injection-filter recall"); ax.set_ylim(-0.03, 1.03); ax.yaxis.grid(True, color=GRID, lw=0.6)
        axes[0].set_ylabel("rate"); axes[0].legend(frameon=False, fontsize=7)
        _save(fig, out, "fig12_filter_recall")
    if (T / "ext_seed_convergence.csv").exists():
        d = pd.read_csv(T / "ext_seed_convergence.csv")
        fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.7))
        for ax, (m, title) in zip(axes, [("safe_containment", "Safe containment"), ("unauthorized_action", "Unauthorized action"),
                                          ("review_rate", "Review rate"), ("false_isolation", "False isolation")]):
            for s_ in ("B0_rules", "B1_unguarded", "B2_guarded"):
                g = d[(d.metric == m) & (d.system == s_)]
                ax.plot(g.n_seeds, g.running_mean, lw=1.5, color=C[s_], label=LABEL[s_])
                ax.fill_between(g.n_seeds, g.running_mean - g.half_width_95, g.running_mean + g.half_width_95, color=C[s_], alpha=0.15, lw=0)
            ax.axvline(5, color=MUTED, lw=0.8, ls=":"); ax.set_title(title, fontsize=9); ax.set_xlabel("number of seeds"); ax.yaxis.grid(True, color=GRID, lw=0.6)
        axes[0].set_ylabel("running mean ± 95% half-width"); axes[2].legend(frameon=False, fontsize=7)
        _save(fig, out, "fig13_seed_convergence")
    if (T / "ext_family_breakdown.csv").exists():
        d = pd.read_csv(T / "ext_family_breakdown.csv")
        fams = ["benign_burst", "scanning", "botnet", "exfiltration", "critical_anomaly", "sparse_evidence"]
        syss = ["B0_rules", "B1_unguarded", "B2_guarded", "B3_single_guarded"]
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.6))
        for ax, (col, title, cmap) in zip(axes, [("appropriate", "Ground-truth-optimal action", "Blues"), ("unauthorized", "Unauthorized action", "Oranges"), ("review", "Sent to review", "Greys")]):
            m = d.pivot(index="system", columns="family", values=col).loc[syss, fams]
            ax.imshow(m.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            for i in range(len(syss)):
                for j in range(len(fams)):
                    ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if m.values[i, j] > 0.6 else INK)
            ax.set_xticks(range(len(fams))); ax.set_xticklabels([f.replace("_", "\n") for f in fams], fontsize=6.5)
            ax.set_yticks(range(len(syss))); ax.set_yticklabels(["B0", "B1", "B2", "B3"] if ax is axes[0] else []); ax.set_title(title, fontsize=9)
            for sp in ax.spines.values():
                sp.set_visible(False)
        _save(fig, out, "fig14_family_breakdown")
    if (T / "ext_reason_codes.csv").exists():
        d = pd.read_csv(T / "ext_reason_codes.csv")
        codes = d.groupby("reason_code").share.max().sort_values(ascending=False).index.tolist()
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        yy = np.arange(len(codes))
        for off, cond, col in ((-0.2, "clean", "#7b8794"), (0.2, "attacked", "#b91c1c")):
            v = d[d.condition == cond].set_index("reason_code").share.reindex(codes).fillna(0)
            ax.barh(yy + off, v, height=0.38, color=col, label=f"{cond} incidents")
        ax.set_yticks(yy); ax.set_yticklabels([c.replace("_", " ").title().replace("Ttl", "TTL") for c in codes], fontsize=7); ax.invert_yaxis()
        ax.set_xlabel("share of final policy decisions (B2)"); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=7.5)
        _save(fig, out, "fig15_reason_codes")
    if (T / "ext_coalescing.csv").exists():
        d = pd.read_csv(T / "ext_coalescing.csv")
        fig, ax = plt.subplots(figsize=(5.4, 2.9))
        for k, lab, col in (("fifo", "FIFO", "#7b8794"), ("priority", "priority", "#1d4ed8"), ("priority+coalesce", "priority + coalescing", "#0f766e")):
            g = d[d.discipline == k]
            ax.plot(g.compromised_devices, g.malicious_sla_miss, marker="o", ms=4, lw=1.7, color=col, label=lab)
        ax.set_xscale("log"); ax.set_xticks([1, 5, 20, 50, 100, 200]); ax.set_xticklabels(["1", "5", "20", "50", "100", "200"])
        ax.set_xlabel("compromised devices sending injected incidents (fleet of 200)"); ax.set_ylabel("malicious escalations not\nreviewed within 15 min")
        ax.set_ylim(0, 0.65); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=7.5)
        _save(fig, out, "fig16_coalescing")
    if (T / "ext_trust_stress.csv").exists():
        d = pd.read_csv(T / "ext_trust_stress.csv")
        CB0 = dict(C)
        fig, axes = plt.subplots(1, 3, figsize=(10.6, 2.9), sharey=True)
        for ax, (st, xl, title, systems) in zip(axes, [
                ("detector probability poisoned", "share of incidents with attacker-set\ndetector probability", "Benign device isolated", ("B1_unguarded", "B2_guarded")),
                ("critical devices mislabelled as low", "share of critical devices\nmislabelled as low", "Truly critical device isolated", ("B1_unguarded", "B2_guarded")),
                ("detector false negatives forced", "share of real attacks hidden\nfrom the detector", "Malicious incident neither\ncontained nor escalated", ("B0_rules", "B1_unguarded", "B2_guarded"))]):
            for k, s_ in enumerate(systems):
                g = d[(d.stress == st) & (d.system == s_)]
                ax.plot(g.level, g.harm_rate, marker="o", ms=4 + (len(systems) - k), lw=1.7, color=CB0[s_], label=LABEL[s_], alpha=0.9)
            ax.set_title(title, fontsize=8.5); ax.set_xlabel(xl, fontsize=7.5); ax.set_ylim(-0.03, 1.05); ax.yaxis.grid(True, color=GRID, lw=0.6)
        axes[0].set_ylabel("harm rate"); axes[2].legend(frameon=False, fontsize=7.5, loc="upper left")
        _save(fig, out, "fig17_trust_stress")


def extended3(inp: Path, out: Path):
    T = inp.parent / "tables"
    import numpy as np
    if (T / "ext_repeat_isolation.csv").exists():
        r, c = pd.read_csv(T / "ext_repeat_isolation.csv"), pd.read_csv(T / "ext_repeat_isolation_curve.csv")
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.1))
        ax = axes[0]
        g = c[c.role.str.startswith("benign")]
        for (sysn, bud), col, ls, lab in [(("B1_unguarded", "no budget"), "#c2410c", "-", "B1 unguarded"), (("B2_guarded", "no budget"), "#93c5fd", "--", "B2, no budget"),
                                          (("B2_guarded", "900 s/h"), "#1d4ed8", "-", "B2, 900 s/h budget"), (("B2_guarded", "300 s/h"), "#1e3a8a", ":", "B2, 300 s/h budget")]:
            h = g[(g.system == sysn) & (g.budget == bud)]
            ax.plot(h.incident, h.cumulative_restricted_seconds / 60, color=col, ls=ls, lw=1.7, label=lab)
        ax.set_xlabel("injected incidents on one benign device\n(one every 12 min)"); ax.set_ylabel("cumulative wrongful isolation (min)")
        ax.set_title("Repeated authorized isolation", fontsize=9); ax.legend(frameon=False, fontsize=7); ax.yaxis.grid(True, color=GRID, lw=0.6)
        b2 = r[(r.system == "B2_guarded")]
        labels = [x for x in dict.fromkeys(b2.budget)]
        x = np.arange(len(labels))
        for ax, col, title, yl in ((axes[1], "share_of_shift_restricted", "Share of the 8 h shift restricted", "share of shift"), (axes[2], "reviews", "Reviews raised (of 40 incidents)", "reviews")):
            for off, role, colr in ((-0.2, "benign look-alike under injection", "#b91c1c"), (0.2, "persistently malicious device", "#0f766e")):
                v = b2[b2.role == role].set_index("budget")[col].reindex(labels)
                ax.bar(x + off, v, width=0.38, color=colr, label={"b": "benign device (wrongful)", "p": "malicious device (legitimate)"}[role[0]])
            ax.set_xticks(x); ax.set_xticklabels([l.replace(" s/h", "") if l != "no budget" else "none" for l in labels], fontsize=7.5)
            ax.set_xlabel("per-device budget (s of restriction per hour)"); ax.set_title(title, fontsize=9); ax.set_ylabel(yl); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        axes[1].legend(frameon=False, fontsize=7)
        _save(fig, out, "fig18_repeat_isolation")
    if (T / "ext_threshold_sensitivity.csv").exists():
        d = pd.read_csv(T / "ext_threshold_sensitivity.csv")
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
        shades = {1: "#93c5fd", 2: "#1d4ed8", 3: "#1e3a8a"}
        for ax, (xc, yc, xl, yl, title) in zip(axes, [
                ("attack_benign_isolation", "autonomous_containment_of_malicious", "authorized harm: benign device isolated under attack", "autonomous containment of malicious incidents", "Containment against authorized harm"),
                ("review_rate", "attack_suppression", "review rate (analyst burden)", "containment suppressed under attack", "Suppression against review burden")]):
            for me, g in d.groupby("min_evidence"):
                g = g.sort_values("p_threshold")
                ax.plot(g[xc], g[yc], marker="o", ms=4, lw=1.3, color=shades[me], label=f"min. evidence = {me}")
                for _, row in g.iterrows():
                    if me == 2:
                        ax.annotate(f"{row.p_threshold:g}", (row[xc], row[yc]), textcoords="offset points", xytext=(4, 4), fontsize=6.5, color=MUTED)
            star = d[(d.p_threshold == 0.8) & (d.min_evidence == 2)].iloc[0]
            ax.plot(star[xc], star[yc], marker="*", ms=13, color="#d97706", zorder=5, ls="none", label="setting fixed in advance")
            ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8); ax.set_title(title, fontsize=9); ax.grid(True, color=GRID, lw=0.6)
        axes[0].legend(frameon=False, fontsize=7, loc="lower right")
        _save(fig, out, "fig19_tradeoff")
    if (T / "ext_netns_enforcement.csv").exists() and (T / "ext_scalability.csv").exists():
        n = pd.read_csv(T / "ext_netns_enforcement.csv"); sc = pd.read_csv(T / "ext_scalability.csv"); cost = pd.read_csv(T / "response_cost.csv").set_index("system")
        pol = float(sc[(sc.component == "validate+authorize")].microseconds_per_op.max()) / 1000
        orch = float(cost.loc["B2_guarded"].decision_latency_us_median) / 1000
        app = n[n.check.str.startswith("B2 clean")].iloc[0]; rb = n[n.check.str.startswith("Explicit rollback")].iloc[0]; ex = n[n.check.str.startswith("Kernel TTL")].iloc[0]
        labels = ["policy decision\n(validate + authorize)", "scripted pipeline,\nalert to decision", "kernel rule apply\n(nft, isolated netns)", "kernel rule rollback", "expiry lag after TTL"]
        vals = [pol, orch, app.latency_ms_median, rb.latency_ms_median, ex.latency_ms_median]
        fig, ax = plt.subplots(figsize=(6.6, 2.6))
        ax.barh(range(5), vals, color=["#1d4ed8", "#1d4ed8", "#0f766e", "#0f766e", "#0f766e"], height=0.6)
        for i, v in enumerate(vals):
            ax.text(v * 1.15, i, f"{v:.3g} ms", va="center", fontsize=7.5, color=INK)
        ax.set_xscale("log"); ax.set_xlim(min(vals) / 3, max(vals) * 12); ax.set_yticks(range(5)); ax.set_yticklabels(labels, fontsize=7.5); ax.invert_yaxis()
        ax.set_xlabel("median latency (ms, log scale); LLM inference not included and not measured"); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        _save(fig, out, "fig20_latency_layers")


def extended4(inp: Path, out: Path):
    T = inp.parent / "tables"
    import numpy as np
    if (T / "ext_formal_proofs.csv").exists():
        d = pd.read_csv(T / "ext_formal_proofs.csv")
        d["short"] = d.theorem.str.slice(0, 64) + np.where(d.theorem.str.len() > 64, "…", "")
        configs = ["full policy", "no provenance", "no criticality", "no ttl cap", "no budget", "no floor"]
        configs = [c for c in configs if c in set(d.configuration)]
        order = sorted(d.theorem.unique(), key=lambda t: (t[0] != "S", t[:2]))
        th = [d[d.theorem == t].short.iloc[0] for t in order]
        M = np.array([[1.0 if d[(d.short == t) & (d.configuration == c)].result.iloc[0] == "proved" else 0.0 for c in configs] for t in th])
        fig, ax = plt.subplots(figsize=(8.4, 3.6))
        from matplotlib.colors import ListedColormap
        ax.imshow(M, cmap=ListedColormap(["#fca5a5", "#bfdbfe"]), aspect="auto", vmin=0, vmax=1)
        for i in range(len(th)):
            for j in range(len(configs)):
                ax.text(j, i, "proved" if M[i, j] else "counter-\nexample", ha="center", va="center", fontsize=6.8, color=INK)
        ax.set_xticks(range(len(configs))); ax.set_xticklabels(["full policy" if c == "full policy" else "− " + c[3:] for c in configs], fontsize=8)
        ax.set_yticks(range(len(th))); ax.set_yticklabels(th, fontsize=7.2)
        ax.tick_params(length=0); [sp.set_visible(False) for sp in ax.spines.values()]
        _save(fig, out, "fig21_formal")
    if (T / "real_unseen_family.csv").exists():
        u = pd.read_csv(T / "real_unseen_family.csv").sort_values("detection_rate_at_0.8")
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
        ax = axes[0]
        y = np.arange(len(u))
        ax.barh(y - 0.2, u["detection_rate_at_0.5"], height=0.38, color="#93c5fd", label="score ≥ 0.5")
        ax.barh(y + 0.2, u["detection_rate_at_0.8"], height=0.38, color="#1d4ed8", label="score ≥ 0.8 (policy threshold)")
        ax.set_yticks(y); ax.set_yticklabels(u.held_out_family, fontsize=8); ax.set_xlim(0, 1.05)
        ax.set_xlabel("share of held-out flows detected"); ax.set_title("Attack family absent from training", fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="lower right"); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax = axes[1]
        if (T / "real_response_summary.csv").exists() and (T / "response_summary.csv").exists():
            rs, ss = pd.read_csv(T / "real_response_summary.csv"), pd.read_csv(T / "response_summary.csv")
            mets = [("safe_containment", "safe\ncontainment"), ("false_isolation", "false\nisolation"), ("unauthorized_action", "unauthorized\naction"), ("review_rate", "review\nrate")]
            x = np.arange(len(mets)); w = 0.13
            for k, s_ in enumerate(("B0_rules", "B1_unguarded", "B2_guarded")):
                for off, src, alpha, hatch in ((-1, ss, 0.45, ""), (0, rs, 1.0, "")):
                    v = [float(src[(src.system == s_) & (src.metric == m)]["mean"].iloc[0]) for m, _ in mets]
                    ax.bar(x + (2 * k + off + 0.5 - 2.5) * w, v, width=w, color=C[s_], alpha=alpha,
                           label=f"{LABEL[s_]}, {'real-traffic detector' if alpha == 1 else 'parametric detector'}")
            ax.set_xticks(x); ax.set_xticklabels([l for _, l in mets], fontsize=7.5); ax.set_ylim(0, 1)
            ax.set_title("Clean outcomes: parametric vs real-traffic detector", fontsize=9); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=6, ncol=1, loc="upper right")
        _save(fig, out, "fig22_realdata")
    if (T / "ext_adaptive_campaigns.csv").exists():
        d = pd.read_csv(T / "ext_adaptive_campaigns.csv")
        strategies = list(dict.fromkeys(d.strategy)); defenses = list(dict.fromkeys(d.defense))
        cols = ["#c2410c", "#bfdbfe", "#60a5fa", "#1d4ed8", "#0f766e"]
        fig, ax = plt.subplots(figsize=(10.0, 3.4))
        x = np.arange(len(strategies)); w = 0.8 / len(defenses)
        for k, dn in enumerate(defenses):
            v = [float(d[(d.strategy == s_) & (d.defense == dn)].device_hours_restricted.iloc[0]) for s_ in strategies]
            ax.bar(x + (k - (len(defenses) - 1) / 2) * w, v, width=w, color=cols[k % len(cols)], label=dn)
        ax.set_xticks(x); ax.set_xticklabels([s_.replace(", every", ",\nevery").replace(" (", "\n(").replace(", during a benign fleet-wide event", ",\nbenign fleet event") for s_ in strategies], fontsize=6.6)
        ax.set_ylabel("device-hours restricted per 8 h shift"); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="upper left")
        ax.axvline(len(strategies) - 1.5, color=MUTED, lw=0.8, ls=":")
        ax.text(len(strategies) - 1.45, ax.get_ylim()[1] * 0.92, "legitimate\ncontainment", fontsize=7, color=MUTED, va="top")
        ax.text(-0.45, ax.get_ylim()[1] * 0.6, "wrongful isolation\n(injected incidents)", fontsize=7, color=MUTED, va="top")
        _save(fig, out, "fig23_adaptive")
    if (T / "ext_harm_bound_adaptive.csv").exists() and (T / "ext_harm_bound.csv").exists():
        ha, hb = pd.read_csv(T / "ext_harm_bound_adaptive.csv"), pd.read_csv(T / "ext_harm_bound.csv")
        cases = [("device budget", "B2, device budget 900 s/h", False),
                 ("device + fleet", "B2, device 900 s/h + fleet 1800 s/h", False),
                 ("device + fleet,\ncorroboration", "B2, device 900 s/h + fleet 1800 s/h with corroboration", False),
                 ("device + fleet,\ncorroboration,\nbenign fleet event", "B2, device 900 s/h + fleet 1800 s/h with corroboration", True)]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 3.2), gridspec_kw={"width_ratios": [1.35, 1]})
        bounds, meas = [], []
        for lab, dn, shared in cases:
            g = ha[(ha.defense == dn) & (ha.devices == 4) & (ha.strategy.str.contains("fleet-wide event") == shared)]
            bounds.append(float(g.bound_device_hours.max())); meas.append(float(g.measured_device_hours.max()))
        x = np.arange(len(cases))
        a1.bar(x - 0.18, bounds, 0.36, color="white", edgecolor="#1d4ed8", hatch="//", label="proved bound (attained by synthesized campaign)")
        a1.bar(x + 0.18, meas, 0.36, color="#1d4ed8", label="best measured adaptive strategy")
        for i, (b_, m_) in enumerate(zip(bounds, meas)):
            a1.text(i + 0.18, m_ + 0.1, f"{m_ / b_:.0%}", ha="center", fontsize=7, color=INK)
        a1.set_xticks(x); a1.set_xticklabels([c_[0] for c_ in cases], fontsize=7)
        a1.set_ylabel("wrongful device-hours per 8 h shift\n(4 attacked devices)"); a1.yaxis.grid(True, color=GRID, lw=0.6); a1.set_axisbelow(True)
        a1.set_ylim(0, max(bounds) * 1.35); a1.legend(frameon=False, fontsize=6.8, loc="upper left", ncol=1)
        sub = hb[~hb.shared_indicator & (hb.corroboration_devices == 0)]
        a2.bar(np.arange(len(sub)) - 0.18, sub.attained_device_hours, 0.36, color="#1d4ed8", label="downtime (bounded)")
        a2b = a2.twinx()
        a2b.bar(np.arange(len(sub)) + 0.18, sub.reviews_when_requesting_every_minute, 0.36, color="#c2410c", label="review tickets")
        a2.set_xticks(np.arange(len(sub))); a2.set_xticklabels([("device\n" if "fleet" not in s_ else "dev.+fleet\n") + s_.split(", ")[-1] for s_ in sub.setting], fontsize=6.8)
        a2.set_ylabel("device-hours restricted"); a2b.set_ylabel("review tickets per shift", color="#c2410c")
        a2b.spines["right"].set_visible(True); a2b.tick_params(axis="y", colors="#c2410c")
        h1, l1 = a2.get_legend_handles_labels(); h2, l2 = a2b.get_legend_handles_labels()
        a2.set_ylim(0, float(sub.attained_device_hours.max()) * 1.35); a2b.set_ylim(0, float(sub.reviews_when_requesting_every_minute.max()) * 1.35)
        a2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.8, loc="upper left")
        a2.set_title("Attacker requesting once a minute per device", fontsize=8.5)
        _save(fig, out, "fig24_harmbound")


def llm_styles(inp: Path, out: Path):
    """Goal-matching excess (injected minus matched control) by injection style and prompt condition, three main models."""
    f = inp.parent / "tables" / "LLM_analysis_family1_injection.csv"
    if not f.exists():
        return
    f1 = pd.read_csv(f)
    models = {"Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B", "google/gemma-4-12b-it": "Gemma-4-12B",
              "mistralai/Mistral-Nemo-Instruct-2407": "Mistral-Nemo"}
    conds = ["base", "defended", "policy", "policy_defended"]
    cond_label = {"base": "base", "defended": "defended", "policy": "policy", "policy_defended": "policy + defended"}
    styles = {"naive": "naive", "authority": "authority", "delimiter": "delimiter", "in_policy": "in-policy"}
    fig, axes = plt.subplots(1, len(models), figsize=(13, 3.6), sharey=True)
    width, cols = 0.2, ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    for ax, (mid, short) in zip(axes, models.items()):
        for i, (st, lab) in enumerate(styles.items()):
            d = f1[(f1.model == mid) & (f1.metric == "a_proposal_goal") & (f1.styles == st)].set_index("condition").loc[conds]
            x = [j + (i - 1.5) * width for j in range(len(conds))]
            ax.bar(x, 100 * d.estimate, width, color=cols[i], label=lab,
                   yerr=[100 * (d.estimate - d.ci95_lo), 100 * (d.ci95_hi - d.estimate)], capsize=2, error_kw={"lw": 0.8})
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([cond_label[c] for c in conds], fontsize=8)
        ax.set_title(short, fontsize=10)
        ax.axhline(0, color="black", lw=0.6)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Goal-matching proposals,\ninjected minus control (pp)")
    axes[-1].legend(fontsize=8, title="Injection style", title_fontsize=8)
    _save(fig, out, "fig25_llm_styles")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/raw")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args(argv)
    inp, out = Path(a.input), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    architecture(out)
    if (inp / "response_runs.csv").exists():
        response(inp, out); latency(inp, out)
    if (inp.parent / "tables" / "attack_overall.csv").exists():
        attacks(inp, out)
    detection(inp, out)
    extended(inp, out)
    extended2(inp, out)
    extended3(inp, out)
    extended4(inp, out)
    llm_styles(inp, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
