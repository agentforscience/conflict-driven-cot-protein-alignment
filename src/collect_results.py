"""Collect every headline number from all experiments into one JSON + a markdown digest.

Run last. Produces results/FINAL_RESULTS.json and results/FINAL_RESULTS.md, which are the
sources for the numbers quoted in REPORT.md.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS

OUT = {}


def load(name, fn=pd.read_csv):
    p = RESULTS / name
    return fn(p) if p.exists() else None


def main():
    lines = ["# Consolidated results\n"]

    # ---------------- E1
    e1 = json.load(open(RESULTS / "e1_summary.json"))
    OUT["E1"] = e1
    au = load("e1_assay_univariate.csv")
    ac = load("e1_assay_categorical.csv")
    mu = load("e1_mutant_univariate.csv")
    aa = load("e1_aa_conflict_map.csv")
    lines += ["## E1 — Conflict cartography\n",
              f"- Panel: 30 models / 10 families / {e1['n_assays']} assays / "
              f"{e1['n_mutants']:,} single mutants (2,465,704 incl. multi-mutants)",
              f"- Family-balanced ensemble mean Spearman: **{e1['mean_rho_ens']:.4f}** "
              f"(best single VenusREM {e1['mean_rho_best_single']:.4f})",
              f"- Mean cross-family agreement (off-diagonal Spearman): {e1['mean_family_corr_offdiag']:.3f}",
              f"- Assay-level EU predicted from dataset covariates: **CV R² = "
              f"{e1['assay_cv_r2_EU']:.3f}** (perm p = {e1['assay_cv_r2_EU_p']:.4f})",
              f"- Mutant-level EU from mutation biology, held-out assays: "
              f"R² = {e1['mutant_groupcv_r2_zEU']:.4f}, Spearman {e1['mutant_groupcv_rho_zEU']:.3f}\n"]
    if au is not None:
        lines.append("Assay covariate -> EU (permutation-tested, FDR corrected):\n")
        lines.append(au[au.target == "EU"][["covariate", "spearman", "p_perm", "p_fdr"]]
                     .round(4).to_markdown(index=False))
        OUT["E1_assay_univariate"] = au[au.target == "EU"].to_dict("records")
    if ac is not None:
        lines.append("\nCategorical covariates -> EU:\n")
        lines.append(ac.round(4).to_markdown(index=False))
    if mu is not None:
        m = mu[mu.target == "z_EU"].sort_values("mean_per_assay_rho")
        lines.append("\nMutation biology -> within-assay standardised EU:\n")
        lines.append(m[["feature", "mean_per_assay_rho", "p_cluster", "p_fdr"]]
                     .round(5).to_markdown(index=False))
        OUT["E1_mutant_univariate"] = m.to_dict("records")
    if aa is not None:
        lines.append("\nMost- and least-contested substitutions (mean within-assay z(EU)):\n")
        lines.append(pd.concat([aa.sort_values("z_EU", ascending=False).head(6),
                                aa.sort_values("z_EU").head(6)])
                     .round(3).to_markdown(index=False))

    # ---------------- E2
    e2 = json.load(open(RESULTS / "e2_summary.json"))
    OUT["E2"] = e2
    sel = load("e2_selective_prediction.csv")
    lines += ["\n\n## E2 — Conflict as a failure detector\n",
              "Area under the selective-Spearman curve:\n"]
    auc = {k.replace("AUC_selective_", ""): round(v, 4)
           for k, v in e2.items() if k.startswith("AUC_selective")}
    lines.append(pd.Series(auc).sort_values(ascending=False).to_frame("AUC").to_markdown())
    if sel is not None:
        s = sel[(sel.signal == "EU") & (sel.coverage.isin([0.8, 0.5, 0.2]))]
        lines.append("\nEU-based abstention vs random (paired over assays):\n")
        lines.append(s[["coverage", "sel_spearman", "delta_vs_random", "cohens_d",
                        "p_wilcoxon"]].round(5).to_markdown(index=False))
    t2 = e2["T2"]
    lines.append("\nConfident-failure 2x2 (ensemble Spearman within stratum):\n")
    lines.append(pd.DataFrame([{"stratum": k, "rho": t2[k]["rho"],
                                "mean |rank err|": t2[k]["err"]}
                               for k in ["loAU_loEU", "loAU_hiEU", "hiAU_loEU", "hiAU_hiEU"]]
                              ).round(4).to_markdown(index=False))
    lines.append(f"\n- Within the low-AU stratum: rho(low EU) − rho(high EU) = "
                 f"**{t2['confident_failure_delta_rho']:.4f}** "
                 f"(Wilcoxon p = {t2['confident_failure_p']:.3g}, "
                 f"Cohen's d = {t2['confident_failure_cohens_d']:.2f})")
    lines.append(f"- Slice dev→holdout conflict transfer: rho = "
                 f"{e2.get('T3_slice_transfer_rho', float('nan')):.3f}")

    # ---------------- E2b
    ab = load("e2b_auroc_summary.csv")
    cb = load("e2b_combination.csv")
    pw = load("e2b_auroc_pairwise.csv")
    lines += ["\n\n## E2b — Range-restriction-free detector comparison\n"]
    if ab is not None:
        lines.append("Per-assay AUROC for detecting the worst-20% ensemble errors:\n")
        lines.append(ab.round(4).to_markdown(index=False))
        OUT["E2b_auroc"] = ab.to_dict("records")
    if pw is not None:
        k = pw[((pw.a == "EU") & (pw.b == "conf_mag")) | ((pw.a == "AU") & (pw.b == "conf_mag"))]
        lines.append("\nKey contrast (conflict beats the confidence-magnitude baseline "
                     "once range restriction is removed):\n")
        lines.append(k.round(5).to_markdown(index=False))
    if cb is not None:
        lines.append("\nHeld-out-assay AUROC of signal combinations:\n")
        lines.append(cb.round(4).to_markdown(index=False))
        OUT["E2b_combination"] = cb.to_dict("records")

    # ---------------- E3
    S = load("e3_summary_table.csv")
    C = load("e3_contrasts.csv")
    H = load("e3_holm.csv")
    L = load("e3_gain_localisation.csv")
    lines += ["\n\n## E3 — Conflict-driven CoT arbitration\n"]
    if S is not None:
        cols = [c for c in ["method", "n_assays", "rho", "rho_pg_official", "rho_ci_lo",
                            "rho_ci_hi", "rho_delta", "rho_d", "rho_p", "ndcg_max@20",
                            "recall_top1pct@20"] if c in S.columns]
        s = S[~S.method.str.startswith("random_weights_")][cols]
        lines.append(s.round(4).to_markdown(index=False))
        OUT["E3_ladder"] = s.to_dict("records")
    if H is not None:
        lines.append("\nHolm-corrected p vs the uniform family-balanced ensemble:\n")
        lines.append(H.round(5).to_markdown(index=False))
    if C is not None:
        lines.append("\nAblation contrasts (paired over assays):\n")
        lines.append(C[["contrast", "n", "delta_rho", "cohens_d", "p_wilcoxon"]]
                     .round(5).to_markdown(index=False))
        OUT["E3_contrasts"] = C.to_dict("records")
    if L is not None:
        lines.append("\nIs the gain concentrated on high-conflict assays?\n")
        lines.append(L[L.conflict_measure == "EU"].round(4).to_markdown(index=False))

    # ---------------- E4
    wq = load("e4_weight_quality.csv")
    gr = load("e4_grounding.csv")
    hc = load("e4_heuristic_check.csv")
    rp = load("e4_rationale_probes.csv")
    lines += ["\n\n## E4 — Are the rationales right?\n"]
    if wq is not None:
        lines.append("[A] Spearman(assigned family weights, true per-assay family accuracy):\n")
        lines.append(wq.round(4).to_markdown(index=False))
        OUT["E4_weight_quality"] = wq.to_dict("records")
    if gr is not None:
        lines.append("\n[B] Grounding in alignment depth:\n")
        lines.append(gr.round(4).to_markdown(index=False))
    if rp is not None:
        piv = rp.pivot_table(index="probe", columns="run", values="auroc")
        lines.append("\n[C] AUROC of 'mentions factor X' for 'factor X truly present':\n")
        lines.append(piv.round(3).to_markdown())
        OUT["E4_probe_auroc"] = piv.round(4).to_dict()
    if hc is not None:
        lines.append("\n[D] Consensus-following vs biological discrimination:\n")
        lines.append(hc.round(4).to_markdown(index=False))
        OUT["E4_heuristic"] = hc.to_dict("records")

    # ---------------- E5
    e5 = load("e5_design_selection_summary.csv")
    lines += ["\n\n## E5 — Budget-constrained variant selection\n"]
    if e5 is not None:
        cols = ["K", "n_assays", "base_max", "rand_max", "pen0.5_max", "pen1.0_max",
                "oracle_loconf_max", "oracle_randhalf_max", "oracle_hiconf_max",
                "oracle_full_max"]
        lines.append(e5[[c for c in cols if c in e5.columns]].round(4).to_markdown(index=False))
        lines.append("\nDeltas vs the plain ensemble top-K, and p-values:\n")
        dc = ["K"] + [f"{n}_delta" for n in ["pen0.5", "pen1.0", "oracle_loconf",
                                             "oracle_randhalf", "oracle_hiconf", "oracle_full"]]
        lines.append(e5[[c for c in dc if c in e5.columns]].round(5).to_markdown(index=False))
        pc = ["K"] + [f"{n}_p" for n in ["pen0.5", "pen1.0", "oracle_hiconf"]]
        lines.append("\n" + e5[[c for c in pc if c in e5.columns]].round(5)
                     .to_markdown(index=False))
        lines.append("\nHit rate on the true top 1%:\n")
        hcx = ["K", "base_hit", "pen0.5_hit", "oracle_randhalf_hit", "oracle_hiconf_hit",
               "oracle_full_hit"]
        lines.append(e5[[c for c in hcx if c in e5.columns]].round(4).to_markdown(index=False))
        OUT["E5"] = e5.to_dict("records")

    (RESULTS / "FINAL_RESULTS.md").write_text("\n".join(str(x) for x in lines))
    json.dump(OUT, open(RESULTS / "FINAL_RESULTS.json", "w"), indent=2, default=float)
    print("[collect] wrote results/FINAL_RESULTS.md and .json")
    print("\n".join(str(x) for x in lines))


if __name__ == "__main__":
    main()
