"""EXPERIMENT 2b — Fair detector comparison and complementarity (follow-up to E2/T1).

E2/T1 found that a trivial confidence-magnitude signal (|ensemble rank - 0.5|) gives a
*higher* selective-Spearman curve than cross-family conflict (EU).  That comparison is
partly an artefact: abstaining on mid-range predictions mechanically restricts the range
of the retained predictor and inflates rank correlation, independently of whether the
abstained points were actually wrong.  Two cleaner tests here:

  (a) ERROR-DETECTION AUROC.  Per assay, binarise "is this mutant in the worst 20% of
      |ensemble rank error|?" and compute AUROC of each risk signal.  This asks directly
      "does the signal find the mistakes?" and is not inflated by range restriction.

  (b) COMPLEMENTARITY.  Does EU add anything on top of the magnitude baseline?  We fit a
      per-assay-standardised logistic combination with GROUPED CV over assays, and also
      report the selective curve of a simple rank-averaged EU+magnitude hybrid.
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS, FIGURES, SEED

rng = np.random.default_rng(SEED)
SIGNALS = ["EU", "AU", "TU", "raw_sd", "conf_mag"]


def main():
    P = pd.read_parquet(RESULTS / "panel_mutants.parquet")
    P["conf_mag"] = -np.abs(P["ens_rank"] - 0.5)
    summary = {}

    # ------------------------------------------------------- (a) error-detection AUROC
    rows = []
    for dms, d in P.groupby("DMS_id", sort=False):
        if len(d) < 200:
            continue
        bad = (d.abs_rank_err >= d.abs_rank_err.quantile(0.8)).astype(int).values
        if bad.sum() < 20 or bad.sum() == len(bad):
            continue
        rec = {"DMS_id": dms, "n": len(d)}
        for s in SIGNALS:
            rec[s] = roc_auc_score(bad, d[s].values)
        # hybrid: mean of within-assay ranks of EU and magnitude signal
        hyb = (stats.rankdata(d.EU.values) + stats.rankdata(d.conf_mag.values)) / 2
        rec["EU+conf_mag"] = roc_auc_score(bad, hyb)
        rows.append(rec)
    AU = pd.DataFrame(rows)
    AU.to_csv(RESULTS / "e2b_error_auroc.csv", index=False)
    print(f"[e2b-a] error-detection AUROC over {len(AU)} assays "
          f"(target: worst 20% of |ensemble rank error|)")
    tab = []
    for s in SIGNALS + ["EU+conf_mag"]:
        v = AU[s].values
        t = stats.ttest_1samp(v, 0.5)
        w = stats.wilcoxon(v - 0.5)
        tab.append({"signal": s, "mean_auroc": v.mean(), "sd": v.std(ddof=1),
                    "frac_assays_gt_0.5": float((v > 0.5).mean()),
                    "t_vs_0.5": t.statistic, "p": w.pvalue})
    T = pd.DataFrame(tab).sort_values("mean_auroc", ascending=False)
    print(T.round(4).to_string(index=False))
    T.to_csv(RESULTS / "e2b_auroc_summary.csv", index=False)

    # paired comparisons between signals
    pair = []
    for a in SIGNALS + ["EU+conf_mag"]:
        for b in SIGNALS + ["EU+conf_mag"]:
            if a >= b:
                continue
            d = AU[a] - AU[b]
            pair.append({"a": a, "b": b, "mean_diff": d.mean(),
                         "cohens_d": d.mean() / d.std(ddof=1),
                         "p_wilcoxon": stats.wilcoxon(d).pvalue})
    PR = pd.DataFrame(pair)
    PR.to_csv(RESULTS / "e2b_auroc_pairwise.csv", index=False)
    print("\n[e2b-a] key paired contrasts:")
    print(PR[PR.a.isin(["EU", "AU", "EU+conf_mag"]) | PR.b.isin(["EU+conf_mag"])]
          .round(5).to_string(index=False))

    # ---------------------------------------------------- (b) grouped-CV combination
    P2 = P.groupby("DMS_id").filter(lambda d: len(d) >= 200).copy()
    g = P2.groupby("DMS_id")
    for s in SIGNALS:
        P2[f"z{s}"] = (P2[s] - g[s].transform("mean")) / g[s].transform("std").replace(0, np.nan)
    P2["bad"] = (P2.abs_rank_err >= g.abs_rank_err.transform(lambda x: x.quantile(0.8))).astype(int)
    P2 = P2.dropna(subset=[f"z{s}" for s in SIGNALS])
    samp = P2.sample(n=min(400_000, len(P2)), random_state=SEED)

    combos = {"conf_mag only": ["zconf_mag"], "EU only": ["zEU"], "AU only": ["zAU"],
              "conf_mag+EU": ["zconf_mag", "zEU"],
              "conf_mag+AU+EU": ["zconf_mag", "zAU", "zEU"]}
    print("\n[e2b-b] GroupKFold (held-out assays) AUROC for predicting ensemble failure")
    cb = []
    for name, cols in combos.items():
        X, y, grp = samp[cols].values, samp.bad.values, samp.DMS_id.values
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, grp):
            m = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
            pred[te] = m.predict_proba(X[te])[:, 1]
        # evaluate per-assay then average (pooled AUROC would be assay-confounded)
        per = pd.DataFrame({"g": grp, "y": y, "p": pred}).groupby("g").apply(
            lambda d: roc_auc_score(d.y, d.p) if d.y.nunique() > 1 else np.nan,
            include_groups=False).dropna()
        cb.append({"features": name, "mean_auroc": per.mean(), "n_assays": len(per)})
        print(f"   {name:16s} AUROC = {per.mean():.4f}  (n={len(per)} assays)")
    CB = pd.DataFrame(cb)
    CB.to_csv(RESULTS / "e2b_combination.csv", index=False)
    summary["combination"] = cb
    summary["auroc"] = T.to_dict("records")

    # ---------------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    o = T.sort_values("mean_auroc")
    ax[0].barh(o.signal, o.mean_auroc - 0.5, left=0.5,
               color=["#b03a2e" if s.startswith("EU") else "#5d6d7e" for s in o.signal])
    ax[0].axvline(0.5, color="k", ls="--", lw=1, label="chance")
    ax[0].set_xlabel("mean per-assay AUROC (detecting worst-20% ensemble errors)")
    ax[0].set_title("Which conflict signal finds the ensemble's mistakes?")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.25, axis="x")

    ax[1].bar(CB.features, CB.mean_auroc, color="#2f6f9f")
    ax[1].axhline(0.5, color="k", ls="--", lw=1)
    ax[1].set_ylim(0.48, max(CB.mean_auroc) * 1.03)
    ax[1].set_ylabel("held-out-assay AUROC")
    ax[1].set_title("Does cross-family conflict add to a confidence baseline?")
    ax[1].tick_params(axis="x", rotation=20)
    ax[1].grid(alpha=.25, axis="y")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig4_detector_comparison.png", dpi=150)
    plt.close()

    json.dump(summary, open(RESULTS / "e2b_summary.json", "w"), indent=2, default=float)
    print("\n[e2b] done")


if __name__ == "__main__":
    main()
