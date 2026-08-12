"""EXPERIMENT 4 — Do the chain-of-thought rationales actually surface the RIGHT conflicts?

The hypothesis is not just "arbitration improves scores"; it is that disagreements
*surfaced through chain-of-thought reasoning* cluster around real biological ambiguities
and dataset biases.  E3 measures whether arbitration helps.  This experiment measures
whether the reasoning is *correct*, which is a separable and arguably more important claim:
a method can help for the wrong reasons, or reason correctly and still not help.

Four analyses:
  A. WEIGHT-QUALITY.  Per assay, Spearman between the LLM's 10 family weights and those
     families' actual Spearman on that assay (ground truth the LLM never saw).  Positive
     values mean the arbiter really does know which families to trust here.
  B. GROUNDING.  Does the LLM's treatment of alignment-based families track the assay's
     true alignment depth?  E1 found EU vs log(Neff/L) rho = -0.43, so a correctly-reasoning
     arbiter should downweight MSA-dependent families when Neff/L is low.
  C. RATIONALE CONTENT.  Keyword mining of the CoT text: which biological factors does the
     model invoke, and does it invoke them when they are actually present (e.g. does it say
     "shallow alignment" precisely on the shallow-alignment assays)?  Reported as a
     detection AUROC against the true covariate.
  D. HEURISTIC CHECK.  Is the arbiter doing biology, or just "follow the majority"?  We
     correlate assigned weight with the family's agreement-centrality; a purely
     consensus-following arbiter would show a strong positive relationship and no
     independent biological grounding.
"""
import sys, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS, FIGURES, FAM_NAMES, SEED

MSA_FAMS = ["MSA_Potts", "Evo_Conservation", "MSA_Conditioned"]
STRUCT_FAMS = ["Inverse_Folding", "Struct_PLM", "Struct_MSA_Hybrid"]

# Keyword probes: (label, regex).  Deliberately generous patterns -- we are measuring
# whether the topic is raised at all, not the exact wording.
PROBES = {
    "shallow_alignment": r"(shallow|low|limited|sparse|insufficient|poor)[^.]{0,40}(alignment|msa|neff|homolog)",
    "deep_alignment": r"(deep|high|rich|abundant|extensive)[^.]{0,40}(alignment|msa|neff|homolog)",
    "viral": r"\b(viral|virus|zika|influenza|hiv)\b",
    "structure_available": r"(structur\w+|3d|backbone|fold)",
    "stability_assay": r"\b(stabilit\w+|thermodynamic|ddg|folding stability)\b",
    "binding_assay": r"\b(binding|affinity|interaction interface)\b",
    "organismal_fitness": r"(organismal fitness|viral replication|growth rate|survival)",
    "long_sequence": r"(long|large)[^.]{0,25}(sequence|protein|length)",
    "outlier_family": r"(outlier|deviat\w+|disagree\w*|conflict\w*)",
    "epistasis_multi": r"(epistas\w+|multi[- ]?mutant|higher[- ]order)",
}


def load():
    recs = []
    for p in RESULTS.glob("e3_gen_*.jsonl"):
        for line in open(p):
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    G = pd.DataFrame(recs).drop_duplicates(
        subset=["model", "condition", "DMS_id", "seed"], keep="last")
    G["short"] = G.model.str.split("/").str[-1] + "|" + G.condition + "|s" + G.seed.astype(str)
    return G


def main():
    G = load()
    Pr = pd.read_csv(RESULTS / "e3_assay_profiles.csv").set_index("DMS_id")
    Pr["log_neff"] = np.log10(Pr.MSA_Neff_L.clip(lower=1e-3))
    true_rho = Pr[[f"rho_{f}" for f in FAM_NAMES]]
    summary = {}
    print(f"[e4] {len(G)} generations across {G.short.nunique()} run configurations")

    # ------------------------------------------------------------------ A weight quality
    rows = []
    for tag, g in G.groupby("short"):
        per = []
        for _, r in g.iterrows():
            if not isinstance(r.weights, dict) or len(r.weights) < len(FAM_NAMES):
                continue
            if r.DMS_id not in true_rho.index:
                continue
            w = np.array([r.weights[f] for f in FAM_NAMES], float)
            t = true_rho.loc[r.DMS_id].values.astype(float)
            if np.std(w) == 0:
                per.append(0.0)          # a flat weight vector carries no information
                continue
            per.append(stats.spearmanr(w, t).statistic)
        per = np.array([p for p in per if np.isfinite(p)])
        if len(per) < 30:
            continue
        t = stats.ttest_1samp(per, 0)
        rows.append({"run": tag, "n_assays": len(per), "mean_weight_rho": per.mean(),
                     "sd": per.std(ddof=1), "frac_positive": float((per > 0).mean()),
                     "t": t.statistic, "p": stats.wilcoxon(per).pvalue})
    WQ = pd.DataFrame(rows).sort_values("mean_weight_rho", ascending=False)
    WQ.to_csv(RESULTS / "e4_weight_quality.csv", index=False)
    print("\n[A] Do the LLM's family weights track the families' true per-assay accuracy?")
    print("    (Spearman between assigned weights and true family Spearman; 0 = no knowledge)")
    print(WQ.round(4).to_string(index=False))

    # ------------------------------------------------------------------ B grounding
    print("\n[B] Grounding: does the arbiter downweight alignment-based families when the")
    print("    alignment is actually shallow?  (E1: assay conflict vs log Neff/L rho=-0.43)")
    gr = []
    for tag, g in G.groupby("short"):
        d = g[g.weights.apply(lambda w: isinstance(w, dict) and len(w) == len(FAM_NAMES))]
        if len(d) < 30:
            continue
        idx = [i for i in d.DMS_id if i in Pr.index]
        d = d[d.DMS_id.isin(idx)].set_index("DMS_id")
        wmsa = d.weights.apply(lambda w: np.mean([w[f] for f in MSA_FAMS]))
        wstr = d.weights.apply(lambda w: np.mean([w[f] for f in STRUCT_FAMS]))
        neff = Pr.loc[d.index, "log_neff"]
        # The empirically correct direction: with deeper alignments, MSA families do better.
        true_msa = true_rho.loc[d.index, [f"rho_{f}" for f in MSA_FAMS]].mean(1)
        r_true = stats.spearmanr(neff, true_msa)
        r_llm = stats.spearmanr(neff, wmsa)
        r_rel = stats.spearmanr(neff, wmsa - wstr)
        gr.append({"run": tag, "n": len(d),
                   "TRUE_msa_acc_vs_neff": r_true.statistic,
                   "LLM_msa_weight_vs_neff": r_llm.statistic, "p_llm": r_llm.pvalue,
                   "LLM_msa_minus_struct_vs_neff": r_rel.statistic, "p_rel": r_rel.pvalue})
    GR = pd.DataFrame(gr).sort_values("LLM_msa_weight_vs_neff", ascending=False)
    GR.to_csv(RESULTS / "e4_grounding.csv", index=False)
    print(GR.round(4).to_string(index=False))

    # ------------------------------------------------------------------ C rationale content
    print("\n[C] Rationale content: is each biological factor raised when it is actually present?")
    cot = G[G.condition.isin(["conflict_cot", "cot_only"])].copy()
    for k, pat in PROBES.items():
        cot[k] = cot.text.str.lower().str.contains(pat, regex=True, na=False)
    cot_ix = cot[cot.DMS_id.isin(Pr.index)]
    truth = {
        "shallow_alignment": (Pr.log_neff < Pr.log_neff.median()),
        "deep_alignment": (Pr.log_neff >= Pr.log_neff.median()),
        "viral": (Pr.taxon == "Virus"),
        "stability_assay": (Pr.coarse_selection_type == "Stability"),
        "binding_assay": (Pr.coarse_selection_type == "Binding"),
        "organismal_fitness": (Pr.coarse_selection_type == "OrganismalFitness"),
        "long_sequence": (Pr.seq_len > Pr.seq_len.median()),
        "epistasis_multi": (Pr.DMS_number_multiple_mutants > 0),
    }
    rows = []
    for tag, g in cot_ix.groupby("short"):
        for k, tv in truth.items():
            y = tv.loc[g.DMS_id].values.astype(int)
            x = g[k].values.astype(int)
            if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
                continue
            rows.append({"run": tag, "probe": k, "mention_rate": float(x.mean()),
                         "mention_rate_when_true": float(x[y == 1].mean()),
                         "mention_rate_when_false": float(x[y == 0].mean()),
                         "auroc": roc_auc_score(y, x),
                         "p_fisher": stats.fisher_exact(
                             [[int(((x == 1) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())],
                              [int(((x == 0) & (y == 1)).sum()), int(((x == 0) & (y == 0)).sum())]]
                         ).pvalue})
    RC = pd.DataFrame(rows)
    RC.to_csv(RESULTS / "e4_rationale_probes.csv", index=False)
    if len(RC):
        piv = RC.pivot_table(index="probe", columns="run", values="auroc")
        print("    AUROC of 'model mentions factor X' for detecting 'factor X is truly present':")
        print(piv.round(3).to_string())
        summary["mean_probe_auroc"] = float(RC.auroc.mean())
    # overall mention rates
    mr = cot.groupby("short")[list(PROBES)].mean()
    mr.to_csv(RESULTS / "e4_mention_rates.csv")
    print("\n    Mention rates by run:")
    print(mr.round(3).to_string())

    # ------------------------------------------------------------------ D heuristic check
    print("\n[D] Is the arbiter just following the majority?  Weight vs agreement-centrality")
    hr = []
    for tag, g in G.groupby("short"):
        d = g[g.weights.apply(lambda w: isinstance(w, dict) and len(w) == len(FAM_NAMES))]
        d = d[d.DMS_id.isin(Pr.index)]
        if len(d) < 30:
            continue
        wv, cv, tv = [], [], []
        for _, r in d.iterrows():
            wv += [r.weights[f] for f in FAM_NAMES]
            cv += [Pr.loc[r.DMS_id, f"cen_{f}"] for f in FAM_NAMES]
            tv += [Pr.loc[r.DMS_id, f"rho_{f}"] for f in FAM_NAMES]
        hr.append({"run": tag,
                   "weight_vs_centrality": stats.spearmanr(wv, cv).statistic,
                   "weight_vs_true_acc": stats.spearmanr(wv, tv).statistic,
                   "centrality_vs_true_acc": stats.spearmanr(cv, tv).statistic,
                   "weight_sd_within_assay": float(np.mean(
                       [np.std([r.weights[f] for f in FAM_NAMES]) for _, r in d.iterrows()]))})
    HR = pd.DataFrame(hr).sort_values("weight_vs_true_acc", ascending=False)
    HR.to_csv(RESULTS / "e4_heuristic_check.csv", index=False)
    print(HR.round(4).to_string(index=False))

    # ------------------------------------------------- E systematic mis-attribution
    # Qualitative inspection showed the arbiter repeatedly penalising Inverse_Folding for
    # "deviating from the consensus" on assays where it was in fact the most accurate family.
    # A family can be an outlier because it is uniquely right.  Quantify per family:
    #   mean LLM weight rank  vs  mean true accuracy rank  (1 = best of the 10)
    print("\n[E] Systematic mis-attribution: is 'outlier' being treated as 'wrong'?")
    mis = []
    for tag, g in G.groupby("short"):
        d = g[g.weights.apply(lambda w: isinstance(w, dict) and len(w) == len(FAM_NAMES))]
        d = d[d.DMS_id.isin(Pr.index)]
        if len(d) < 30:
            continue
        wr = np.zeros((len(d), len(FAM_NAMES)))
        tr = np.zeros_like(wr)
        cr = np.zeros_like(wr)
        for i, (_, r) in enumerate(d.iterrows()):
            wr[i] = stats.rankdata([-r.weights[f] for f in FAM_NAMES])
            tr[i] = stats.rankdata([-Pr.loc[r.DMS_id, f"rho_{f}"] for f in FAM_NAMES])
            cr[i] = stats.rankdata([-Pr.loc[r.DMS_id, f"cen_{f}"] for f in FAM_NAMES])
        for k, f in enumerate(FAM_NAMES):
            mis.append({"run": tag, "family": f,
                        "mean_llm_weight_rank": wr[:, k].mean(),
                        "mean_true_acc_rank": tr[:, k].mean(),
                        "mean_centrality_rank": cr[:, k].mean(),
                        "misrank": wr[:, k].mean() - tr[:, k].mean(),
                        "p": stats.wilcoxon(wr[:, k] - tr[:, k]).pvalue
                        if np.any(wr[:, k] != tr[:, k]) else np.nan})
    MIS = pd.DataFrame(mis)
    MIS.to_csv(RESULTS / "e4_family_misranking.csv", index=False)
    prim = MIS[MIS.run.str.contains("conflict_cot")]
    print("    (positive misrank = family is trusted LESS than it deserves; "
          "primary conflict_cot runs)")
    print(prim.sort_values("misrank", ascending=False)
          [["run", "family", "mean_llm_weight_rank", "mean_true_acc_rank",
            "mean_centrality_rank", "misrank", "p"]].round(3).to_string(index=False))
    summary["worst_misranked_family"] = (
        prim.groupby("family").misrank.mean().sort_values(ascending=False).index[0])

    # the same, restricted to Stability assays, where structure-based families excel
    stab = Pr.index[Pr.coarse_selection_type == "Stability"]
    srows = []
    for tag, g in G.groupby("short"):
        d = g[g.DMS_id.isin(stab)]
        d = d[d.weights.apply(lambda w: isinstance(w, dict) and len(w) == len(FAM_NAMES))]
        if len(d) < 20:
            continue
        for f in FAM_NAMES:
            w = np.array([r.weights[f] for _, r in d.iterrows()])
            t = Pr.loc[d.DMS_id, f"rho_{f}"].values
            srows.append({"run": tag, "family": f, "n_stability_assays": len(d),
                          "mean_weight": w.mean(),
                          "mean_true_rho": t.mean()})
    ST = pd.DataFrame(srows)
    if len(ST):
        ST.to_csv(RESULTS / "e4_stability_assays.csv", index=False)
        p = ST[ST.run.str.contains("conflict_cot")]
        print("\n    On STABILITY assays specifically (structure-based families are known "
              "to excel there):")
        print(p.sort_values(["run", "mean_true_rho"], ascending=[True, False])
              .round(3).to_string(index=False))

    # ------------------------------------------------------------------ figure
    if len(WQ):
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        o = WQ.sort_values("mean_weight_rho")
        c = ["#b03a2e" if "conflict_cot" in r else "#5d6d7e" for r in o.run]
        ax[0].barh(o.run, o.mean_weight_rho, color=c)
        ax[0].axvline(0, color="k", lw=1)
        ax[0].set_xlabel("Spearman(assigned weights, true family accuracy)")
        ax[0].set_title("[A] Does the arbiter know which families to trust?")
        ax[0].tick_params(axis="y", labelsize=8)
        ax[0].grid(alpha=.25, axis="x")

        o2 = HR.sort_values("weight_vs_true_acc")
        x = np.arange(len(o2))
        ax[1].barh(x - 0.2, o2.weight_vs_centrality, height=0.4,
                   color="#7d3c98", label="weight vs family agreement-centrality")
        ax[1].barh(x + 0.2, o2.weight_vs_true_acc, height=0.4,
                   color="#1e8449", label="weight vs family TRUE accuracy")
        ax[1].set_yticks(x, o2.run, fontsize=8)
        ax[1].axvline(0, color="k", lw=1)
        ax[1].legend(fontsize=8)
        ax[1].set_title("[D] Consensus-following vs genuine biological discrimination")
        ax[1].grid(alpha=.25, axis="x")
        plt.tight_layout()
        plt.savefig(FIGURES / "fig6_rationale_analysis.png", dpi=150)
        plt.close()

    json.dump(summary, open(RESULTS / "e4_summary.json", "w"), indent=2, default=float)
    print("\n[e4] done")


if __name__ == "__main__":
    main()
