"""EXPERIMENT 3c — Evaluate conflict-driven CoT arbitration against the full baseline ladder.

Every method produces a per-assay weight vector over the 10 model families; the weighted
mean of the families' normalised-rank predictions is the ensemble prediction, scored by
Spearman rho against the experimental DMS score, plus top-K selection metrics that proxy
"protein sequence design performance" under a screening budget.

BASELINE LADDER (in increasing order of what they control for)
  best_single         best individual model on the ProteinGym leaderboard (VenusREM)
  uniform             family-balanced uniform ensemble -- THE BAR TO BEAT
  random_weights      random weights, 20 draws -- shows what non-uniformity costs by chance
  learned_meta        non-LLM meta-learner: predicts family weights from dataset covariates,
                      trained on held-out assays (GroupKFold). The real competitor: it uses
                      the same information as the LLM but fits it statistically.
  direct_only         LLM, metadata only, no CoT, no conflict         (C4)
  cot_only            LLM, metadata only, WITH CoT  -- compute-matched control  (C3)
  conflict_direct     LLM, conflict shown, no CoT                     (C2)
  conflict_cot        LLM, conflict shown, WITH CoT -- THE PROPOSED METHOD  (C1)
  oracle              weights fitted on the ground truth of the same assay (cheating upper
                      bound: how much headroom per-assay reweighting has at all)

Statistics: paired Wilcoxon signed-rank over assays (the unit of analysis), Cohen's d on
the paired differences, bootstrap CIs, and Holm correction across the ladder.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS, FIGURES, FAM_NAMES, SEED, BEST_SINGLE

rng = np.random.default_rng(SEED)
TOPK = [10, 20, 30, 40]


def load_generations():
    recs = []
    for p in RESULTS.glob("e3_gen_*.jsonl"):
        for line in open(p):
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    G = pd.DataFrame(recs)
    G = G.drop_duplicates(subset=["model", "condition", "DMS_id", "seed"], keep="last")
    return G


def score_weights(P, W, tag):
    """W: dict DMS_id -> np.array(10). Returns per-assay metrics."""
    fam_cols = [f"fam_{f}" for f in FAM_NAMES]
    rows = []
    for dms, d in P.groupby("DMS_id", sort=False):
        if dms not in W:
            continue
        w = np.asarray(W[dms], float)
        if not np.isfinite(w).all() or w.sum() <= 0:
            continue
        w = w / w.sum()
        pred = d[fam_cols].to_numpy(float) @ w
        y = d.y.to_numpy(float)
        rec = {"method": tag, "DMS_id": dms, "n": len(d),
               "rho": stats.spearmanr(pred, y).statistic}
        # top-K selection: normalised max fitness among the K variants the method picks
        ymin, ymax = y.min(), y.max()
        rng_y = (ymax - ymin) or 1.0
        order = np.argsort(-pred)
        true_top = set(np.argsort(-y)[: max(1, int(0.01 * len(y)))])
        for K in TOPK:
            sel = order[:K]
            rec[f"ndcg_max@{K}"] = float((y[sel].max() - ymin) / rng_y)
            rec[f"recall_top1pct@{K}"] = len(set(sel) & true_top) / max(1, min(K, len(true_top)))
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    P = pd.read_parquet(RESULTS / "panel_mutants.parquet")
    A = pd.read_csv(RESULTS / "panel_assays.csv").set_index("DMS_id")
    Pr = pd.read_csv(RESULTS / "e3_assay_profiles.csv").set_index("DMS_id")
    G = load_generations()
    assays = sorted(P.DMS_id.unique())
    fam_cols = [f"fam_{f}" for f in FAM_NAMES]
    print(f"[e3c] {len(G)} generations: "
          f"{G.groupby(['model','condition','seed']).size().to_dict()}")
    print(f"[e3c] parse status: {G.parse_status.value_counts().to_dict()}")

    all_res, weight_log = [], []

    # ---------------------------------------------------------------- non-LLM baselines
    W_unif = {d: np.ones(10) for d in assays}
    all_res.append(score_weights(P, W_unif, "uniform"))

    # best single model (not a family weighting -- scored directly)
    rows = []
    for dms, d in P.groupby("DMS_id", sort=False):
        y = d.y.to_numpy(float)
        pred = d[f"single_{BEST_SINGLE}"].to_numpy(float)
        ymin, ymax = y.min(), y.max()
        rng_y = (ymax - ymin) or 1.0
        order = np.argsort(-pred)
        true_top = set(np.argsort(-y)[: max(1, int(0.01 * len(y)))])
        rec = {"method": "best_single", "DMS_id": dms, "n": len(d),
               "rho": stats.spearmanr(pred, y).statistic}
        for K in TOPK:
            sel = order[:K]
            rec[f"ndcg_max@{K}"] = float((y[sel].max() - ymin) / rng_y)
            rec[f"recall_top1pct@{K}"] = len(set(sel) & true_top) / max(1, min(K, len(true_top)))
        rows.append(rec)
    all_res.append(pd.DataFrame(rows))

    # random weights: 20 draws, averaged per assay (what does arbitrary non-uniformity cost?)
    rr = []
    for s in range(20):
        r = np.random.default_rng(1000 + s)
        W = {d: r.integers(0, 11, 10).astype(float) + 0.01 for d in assays}
        x = score_weights(P, W, f"random_weights_{s}")
        rr.append(x)
    RW = pd.concat(rr)
    rwm = RW.groupby("DMS_id").mean(numeric_only=True).reset_index()
    rwm["method"] = "random_weights"
    all_res.append(rwm)

    # learned meta-weighting: non-LLM competitor with the same covariate information.
    # For each family, regress that family's per-assay Spearman on dataset covariates,
    # training only on OTHER assays (GroupKFold), then softmax the predictions into weights.
    cov = pd.DataFrame(index=Pr.index)
    cov["log_neff"] = np.log10(Pr.MSA_Neff_L.clip(lower=1e-3))
    cov["log_nseq"] = np.log10(Pr.MSA_num_seqs.clip(lower=1))
    cov["cov"] = Pr.MSA_perc_cov
    cov["log_len"] = np.log10(Pr.seq_len)
    cov["log_n"] = np.log10(Pr.n)
    cov["EU"] = Pr.EU
    cov["AU"] = Pr.AU
    for f in FAM_NAMES:
        cov[f"cen_{f}"] = Pr[f"cen_{f}"]
        cov[f"dev_{f}"] = Pr[f"dev_{f}"]
    for c in ["taxon", "coarse_selection_type"]:
        cov = pd.concat([cov, pd.get_dummies(Pr[c].fillna("NA"), prefix=c).astype(float)], axis=1)
    cov = cov.fillna(cov.median(numeric_only=True))
    ids = cov.index.values
    pred_rho = pd.DataFrame(index=cov.index, columns=FAM_NAMES, dtype=float)
    for f in FAM_NAMES:
        yv = Pr[f"rho_{f}"].values
        for tr, te in GroupKFold(5).split(cov.values, yv, groups=ids):
            sc = StandardScaler().fit(cov.values[tr])
            m = Ridge(alpha=5.0).fit(sc.transform(cov.values[tr]), yv[tr])
            pred_rho.iloc[te, FAM_NAMES.index(f)] = m.predict(sc.transform(cov.values[te]))
    Z = pred_rho.values
    Z = (Z - Z.mean(1, keepdims=True)) / (Z.std(1, keepdims=True) + 1e-9)
    Wlm = np.exp(2.0 * Z)          # temperature chosen a priori, not tuned on the outcome
    W_learned = {d: Wlm[i] for i, d in enumerate(cov.index)}
    all_res.append(score_weights(P, W_learned, "learned_meta"))

    # ABLATION of the learned meta-weighter: is its gain driven by the CONFLICT PROFILE
    # (per-family deviation-from-consensus / agreement-centrality / assay EU,AU) or merely
    # by ordinary dataset covariates?  This is the cleanest non-LLM test of the hypothesis's
    # central claim that *identifying conflict points* is what buys the improvement.
    conflict_feats = [c for c in cov.columns
                      if c.startswith(("cen_", "dev_")) or c in ("EU", "AU")]
    ablations = {"learned_meta_noconflict": [c for c in cov.columns if c not in conflict_feats],
                 "learned_meta_conflictonly": conflict_feats}
    for tag, cols in ablations.items():
        pr2 = pd.DataFrame(index=cov.index, columns=FAM_NAMES, dtype=float)
        Xc = cov[cols].values
        for f in FAM_NAMES:
            yv = Pr[f"rho_{f}"].values
            for tr, te in GroupKFold(5).split(Xc, yv, groups=ids):
                sc = StandardScaler().fit(Xc[tr])
                m = Ridge(alpha=5.0).fit(sc.transform(Xc[tr]), yv[tr])
                pr2.iloc[te, FAM_NAMES.index(f)] = m.predict(sc.transform(Xc[te]))
        Z2 = pr2.values
        Z2 = (Z2 - Z2.mean(1, keepdims=True)) / (Z2.std(1, keepdims=True) + 1e-9)
        all_res.append(score_weights(P, {d: np.exp(2.0 * Z2[i])
                                         for i, d in enumerate(cov.index)}, tag))

    # oracle: softmax of the family's TRUE Spearman on this assay (cheating upper bound)
    T = Pr[[f"rho_{f}" for f in FAM_NAMES]].values
    Tz = (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-9)
    W_or = {d: np.exp(2.0 * Tz[i]) for i, d in enumerate(Pr.index)}
    all_res.append(score_weights(P, W_or, "oracle"))

    # ---------------------------------------------------------------- LLM conditions
    for (mdl, cond, seed), g in G.groupby(["model", "condition", "seed"]):
        W = {}
        for _, r in g.iterrows():
            if not isinstance(r.weights, dict) or len(r.weights) < len(FAM_NAMES):
                continue
            v = np.array([r.weights[f] for f in FAM_NAMES], float)
            v = np.clip(v, 0, 10)
            if v.sum() <= 0:
                continue
            W[r.DMS_id] = v + 1e-3
            weight_log.append({"model": mdl, "condition": cond, "seed": seed,
                               "DMS_id": r.DMS_id,
                               **{f: r.weights[f] for f in FAM_NAMES}})
        tag = f"{cond}|{mdl.split('/')[-1]}|s{seed}"
        if len(W) < 50:
            print(f"  skip {tag}: only {len(W)} parsed")
            continue
        res = score_weights(P, W, tag)
        res["n_parsed"] = len(W)
        all_res.append(res)
        print(f"  scored {tag}: {len(W)}/{len(assays)} assays parsed")

    R = pd.concat(all_res, ignore_index=True)
    R.to_csv(RESULTS / "e3_per_assay_results.csv", index=False)
    pd.DataFrame(weight_log).to_csv(RESULTS / "e3_llm_weights.csv", index=False)

    # ---------------------------------------------------------------- paired comparison
    base = R[R.method == "uniform"].set_index("DMS_id")
    metrics = ["rho"] + [f"ndcg_max@{K}" for K in TOPK] + [f"recall_top1pct@{K}" for K in TOPK]
    rows = []
    for m, g in R.groupby("method"):
        g = g.set_index("DMS_id")
        common = g.index.intersection(base.index)
        rec = {"method": m, "n_assays": len(common)}
        for met in metrics:
            v = g.loc[common, met].values
            b = base.loc[common, met].values
            d = v - b
            rec[met] = float(np.mean(v))
            if m != "uniform" and np.any(d != 0):
                rec[f"{met}_delta"] = float(d.mean())
                rec[f"{met}_d"] = float(d.mean() / d.std(ddof=1))
                rec[f"{met}_p"] = float(stats.wilcoxon(d).pvalue)
            bs = rng.integers(0, len(v), size=(2000, len(v)))
            rec[f"{met}_ci_lo"] = float(np.percentile(v[bs].mean(1), 2.5))
            rec[f"{met}_ci_hi"] = float(np.percentile(v[bs].mean(1), 97.5))
        rows.append(rec)
    # ProteinGym's official bias-corrected aggregation: average per-assay scores within
    # (UniProt_ID, coarse selection type) groups first, then average across groups, so that
    # proteins carrying many assays do not dominate.  A plain mean over assays is NOT
    # comparable to the published leaderboard.
    grp = A[["UniProt_ID", "coarse_selection_type"]]
    Rg = R.join(grp, on="DMS_id")
    off = (Rg.groupby(["method", "UniProt_ID", "coarse_selection_type"])
             .rho.mean().groupby("method").mean().rename("rho_pg_official"))
    S = pd.DataFrame(rows).sort_values("rho", ascending=False)
    S = S.merge(off.reset_index(), on="method", how="left")
    S.to_csv(RESULTS / "e3_summary_table.csv", index=False)

    # Holm correction over the LLM/baseline ladder for the primary metric (rho)
    cand = S[(S.method != "uniform") & (~S.method.str.startswith("random_weights_"))].copy()
    cand = cand.dropna(subset=["rho_p"])
    order = cand.rho_p.rank(method="first").astype(int)
    mm = len(cand)
    cand["rho_p_holm"] = np.minimum(1.0, cand.rho_p * (mm - order + 1))
    cand[["method", "rho", "rho_delta", "rho_d", "rho_p", "rho_p_holm"]] \
        .to_csv(RESULTS / "e3_holm.csv", index=False)

    show = ["method", "n_assays", "rho", "rho_pg_official", "rho_ci_lo", "rho_ci_hi",
            "rho_delta", "rho_d", "rho_p", "ndcg_max@20", "recall_top1pct@20"]
    print("\n================ E3 SUMMARY (primary metric: per-assay Spearman) ============")
    print(S[~S.method.str.startswith("random_weights_")][show].round(4).to_string(index=False))
    print("\n--- Holm-corrected p vs uniform family-balanced ensemble ---")
    print(cand[["method", "rho_delta", "rho_d", "rho_p", "rho_p_holm"]].round(5).to_string(index=False))

    # ---------------------------------------------------------------- direct ablation contrasts
    print("\n--- Head-to-head ablation contrasts (paired over assays) ---")
    contr = []
    def pair(a, b, label):
        ga = R[R.method == a].set_index("DMS_id")
        gb = R[R.method == b].set_index("DMS_id")
        ix = ga.index.intersection(gb.index)
        if len(ix) < 30:
            return
        d = ga.loc[ix, "rho"].values - gb.loc[ix, "rho"].values
        contr.append({"contrast": label, "a": a, "b": b, "n": len(ix),
                      "delta_rho": d.mean(), "cohens_d": d.mean() / d.std(ddof=1),
                      "p_wilcoxon": stats.wilcoxon(d).pvalue})
    for mdl in G.model.unique():
        s = mdl.split("/")[-1]
        pair(f"conflict_cot|{s}|s0", f"cot_only|{s}|s0",
             f"[{s}] effect of SURFACING CONFLICT (compute-matched CoT control)")
        pair(f"conflict_cot|{s}|s0", f"conflict_direct|{s}|s0",
             f"[{s}] effect of CHAIN-OF-THOUGHT (same conflict info)")
        pair(f"cot_only|{s}|s0", f"direct_only|{s}|s0",
             f"[{s}] effect of CoT without conflict")
        pair(f"conflict_cot|{s}|s0", "learned_meta",
             f"[{s}] LLM conflict-CoT vs non-LLM learned meta-weighting")
        # order-randomised replication of the primary contrast
        pair(f"conflict_cot_shuf|{s}|s0", f"cot_only_shuf|{s}|s0",
             f"[{s}] SURFACING CONFLICT, family order randomised per assay")
        pair(f"conflict_cot_shuf|{s}|s0", f"conflict_cot|{s}|s0",
             f"[{s}] order-randomised vs fixed-order conflict CoT")
    pair("learned_meta", "learned_meta_noconflict",
         "[non-LLM] does the conflict profile add to plain dataset covariates?")
    C = pd.DataFrame(contr)
    if len(C):
        C.to_csv(RESULTS / "e3_contrasts.csv", index=False)
        print(C.round(5).to_string(index=False))

    # -------------------------------------------- does the gain land where conflict is high?
    # The hypothesis predicts targeted consensus should pay off specifically on high-conflict
    # assays.  Regress the per-assay improvement over uniform on the assay's conflict level.
    print("\n--- Is the improvement concentrated on high-conflict assays? ---")
    loc_rows = []
    for m in sorted(set(R.method)):
        if m.startswith("random_weights_") or m in ("uniform", "best_single"):
            continue
        g = R[R.method == m].set_index("DMS_id")
        ix = g.index.intersection(base.index).intersection(Pr.index)
        if len(ix) < 50:
            continue
        gain = g.loc[ix, "rho"].values - base.loc[ix, "rho"].values
        hi = (Pr.loc[ix, "EU"] > Pr.loc[ix, "EU"].median()).values
        extra = {"gain_hi_EU": float(gain[hi].mean()),
                 "gain_lo_EU": float(gain[~hi].mean()),
                 "p_hi_vs_lo": float(stats.mannwhitneyu(gain[hi], gain[~hi]).pvalue)}
        for cv_name in ["EU", "AU"]:
            r = stats.spearmanr(Pr.loc[ix, cv_name].values, gain)
            loc_rows.append({"method": m, "conflict_measure": cv_name,
                             "spearman_gain_vs_conflict": r.statistic, "p": r.pvalue,
                             "n": len(ix), **extra})
    LOC = pd.DataFrame(loc_rows)
    LOC.to_csv(RESULTS / "e3_gain_localisation.csv", index=False)
    print(LOC[LOC.conflict_measure == "EU"].round(5).to_string(index=False))

    # ---------------------------------------------------------------- seed stochasticity
    seeds = R[R.method.str.contains(r"\|s[12]$", regex=True)]
    if len(seeds):
        print("\n--- Run-to-run stochasticity (sampled seeds, temperature 0.7) ---")
        tmp = R[R.method.str.startswith("conflict_cot|")]
        print(tmp.groupby("method").rho.mean().round(4).to_string())

    # ---------------------------------------------------------------- figures
    key = [m for m in ["best_single", "random_weights", "learned_meta_noconflict",
                       "learned_meta", "direct_only|Qwen3-14B|s0", "cot_only|Qwen3-14B|s0",
                       "cot_only_shuf|Qwen3-14B|s0", "conflict_direct|Qwen3-14B|s0",
                       "conflict_cot_shuf|Qwen3-14B|s0", "conflict_cot|Qwen3-14B|s0",
                       "conflict_cot|Qwen3-8B|s0", "cot_only|Qwen3-8B|s0", "oracle"]
           if m in set(S.method)]
    # Paired deltas vs the uniform ensemble, with paired-bootstrap CIs.  Absolute per-assay
    # Spearman varies enormously between assays (0.07 to 0.82), so absolute error bars hide
    # the comparison of interest; every method is scored on the same assays, so the paired
    # difference is the right quantity.
    key = [k for k in key if k != "random_weights"] + ["random_weights"]
    deltas, dlo, dhi = [], [], []
    for k in key:
        gk = R[R.method == k].set_index("DMS_id")
        ix = gk.index.intersection(base.index)
        d = gk.loc[ix, "rho"].values - base.loc[ix, "rho"].values
        bs = rng.integers(0, len(d), size=(2000, len(d)))
        deltas.append(d.mean())
        dlo.append(np.percentile(d[bs].mean(1), 2.5))
        dhi.append(np.percentile(d[bs].mean(1), 97.5))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
    colors = ["#5d6d7e" if not k.startswith("conflict_cot") else "#b03a2e" for k in key]
    colors = ["#1e8449" if k == "oracle" else c for k, c in zip(key, colors)]
    colors = ["#2f6f9f" if k.startswith("learned_meta") else c for k, c in zip(key, colors)]
    y = np.arange(len(key))
    ax[0].barh(y, deltas, color=colors,
               xerr=[np.array(deltas) - np.array(dlo), np.array(dhi) - np.array(deltas)],
               capsize=3)
    ax[0].set_yticks(y, [k.replace("|Qwen3", "\nQwen3") for k in key], fontsize=8)
    ax[0].axvline(0, color="k", ls="--", lw=1, label="uniform family-balanced ensemble")
    ax[0].set_xlabel("Δ mean per-assay Spearman $\\rho$ vs uniform ensemble (216 assays)")
    ax[0].set_title("Conflict-driven CoT arbitration vs the baseline ladder\n"
                    "(paired differences, 95% bootstrap CI)")
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].grid(alpha=.25, axis="x")

    Sm = S.set_index("method")
    sub = Sm.loc[key]
    ax[1].barh(y, sub["ndcg_max@20"].values, color=colors)
    ax[1].set_yticks(y, [])
    ax[1].axvline(Sm.loc["uniform", "ndcg_max@20"], color="k", ls="--", lw=1)
    ax[1].set_xlim(min(sub["ndcg_max@20"]) * 0.985, max(sub["ndcg_max@20"]) * 1.005)
    ax[1].set_xlabel("normalised max fitness in top-20 selection")
    ax[1].set_title("Design-relevant selection metric (budget = 20 variants)")
    ax[1].grid(alpha=.25, axis="x")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig5_e3_arbitration.png", dpi=150)
    plt.close()

    json.dump({"ladder": S[~S.method.str.startswith("random_weights_")]
               .to_dict("records"),
               "contrasts": C.to_dict("records") if len(C) else []},
              open(RESULTS / "e3_summary.json", "w"), indent=2, default=float)
    print("\n[e3c] wrote results/e3_summary_table.csv, e3_summary.json, "
          "figures/fig5_e3_arbitration.png")


if __name__ == "__main__":
    main()
