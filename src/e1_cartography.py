"""EXPERIMENT 1b — Conflict cartography (direction D1, hypothesis claim C1).

Question: is cross-model conflict *structured*, i.e. does it cluster around identifiable
biological ambiguities and dataset biases, or is it unstructured noise?

Two levels of analysis:
  (A) ASSAY level  -- regress mean epistemic conflict (EU) on ProteinGym's dataset
      covariates (MSA depth, taxon, selection assay, sequence length ...).
      Significance by permutation test (shuffle the covariate, refit, 2000 draws) and
      by grouped cross-validation for the multivariate model.
  (B) MUTANT level -- within-assay standardised EU regressed on mutation-level biology
      (BLOSUM class, hydrophobicity/volume/charge change, relative position, proline
      introduction, wild-type glycine ...).  Within-assay standardisation removes the
      assay as a confound, so any surviving signal is genuinely mutation-level.

Outputs: results/e1_*.csv, results/e1_summary.json, figures/fig1_*.png
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS, FIGURES, SEED, FAM_NAMES

rng = np.random.default_rng(SEED)
N_PERM = 2000


def perm_test_spearman(x, y, n=N_PERM, rng=rng):
    """Permutation p-value for Spearman rho (two-sided), robust to non-normality."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    if len(x) < 8:
        return np.nan, np.nan, len(x)
    obs = stats.spearmanr(x, y).statistic
    null = np.array([stats.spearmanr(x, rng.permutation(y)).statistic for _ in range(n)])
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1)
    return obs, p, len(x)


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    pp = p[ok]
    order = np.argsort(pp)
    m = len(pp)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        val = pp[order[rank]] * m / (rank + 1)
        prev = min(prev, val)
        adj[order[rank]] = prev
    out[ok] = adj
    return out


def main():
    A = pd.read_csv(RESULTS / "panel_assays.csv")
    P = pd.read_parquet(RESULTS / "panel_mutants.parquet")
    summary = {}
    print(f"[e1b] {len(A)} assays, {len(P):,} mutants")

    # ---------------------------------------------------------------- (A) assay level
    A["log_Neff_L"] = np.log10(A["MSA_Neff_L"].clip(lower=1e-3))
    A["log_seq_len"] = np.log10(A["seq_len"])
    A["log_n"] = np.log10(A["n"])
    A["log_MSA_num_seqs"] = np.log10(A["MSA_num_seqs"].clip(lower=1))
    A["frac_multi"] = A["DMS_number_multiple_mutants"] / A["n"]

    num_cov = ["log_Neff_L", "log_MSA_num_seqs", "MSA_perc_cov", "log_seq_len",
               "log_n", "frac_multi"]
    rows = []
    for c in num_cov:
        for tgt in ["EU", "AU", "rho_ens"]:
            r, p, nn = perm_test_spearman(A[c], A[tgt])
            rows.append({"level": "assay", "covariate": c, "target": tgt,
                         "spearman": r, "p_perm": p, "n": nn})
    uni = pd.DataFrame(rows)
    uni["p_fdr"] = bh_fdr(uni.p_perm)
    uni.to_csv(RESULTS / "e1_assay_univariate.csv", index=False)
    print("\n[assay] univariate covariate -> conflict (permutation-tested)")
    print(uni[uni.target == "EU"].to_string(index=False))

    # categorical covariates: Kruskal-Wallis + permutation
    cat_rows = []
    for c in ["taxon", "coarse_selection_type", "MSA_Neff_L_category", "region_mutated"]:
        g = A.dropna(subset=[c, "EU"])
        groups = [v["EU"].values for _, v in g.groupby(c) if len(v) >= 5]
        if len(groups) < 2:
            continue
        H = stats.kruskal(*groups).statistic
        lab = g[c].values
        null = []
        for _ in range(N_PERM):
            sh = rng.permutation(lab)
            gg = [g["EU"].values[sh == u] for u in np.unique(sh)]
            gg = [x for x in gg if len(x) >= 5]
            null.append(stats.kruskal(*gg).statistic if len(gg) >= 2 else 0.0)
        p = (np.sum(np.array(null) >= H) + 1) / (N_PERM + 1)
        # eta-squared style effect size
        eta = (H - len(groups) + 1) / (len(g) - len(groups))
        cat_rows.append({"covariate": c, "n_groups": len(groups), "H": H,
                         "p_perm": p, "eta2_H": eta,
                         "group_means": {str(k): float(v.EU.mean())
                                         for k, v in g.groupby(c) if len(v) >= 5}})
    cat = pd.DataFrame(cat_rows)
    cat["p_fdr"] = bh_fdr(cat.p_perm)
    cat.drop(columns=["group_means"]).to_csv(RESULTS / "e1_assay_categorical.csv", index=False)
    print("\n[assay] categorical covariate -> EU")
    print(cat.drop(columns=["group_means"]).to_string(index=False))
    for r in cat_rows:
        print(f"   {r['covariate']}: " + ", ".join(f"{k}={v:.3f}" for k, v in
              sorted(r["group_means"].items(), key=lambda kv: -kv[1])))

    # multivariate: how much assay-level EU variance do dataset covariates explain?
    X = A[num_cov].copy()
    for c in ["taxon", "coarse_selection_type"]:
        d = pd.get_dummies(A[c].fillna("NA"), prefix=c, drop_first=True).astype(float)
        X = pd.concat([X, d], axis=1)
    X = X.fillna(X.median(numeric_only=True))
    for tgt in ["EU", "AU"]:
        yv = A[tgt].values
        kf = KFold(5, shuffle=True, random_state=SEED)
        pred = np.zeros(len(yv))
        for tr, te in kf.split(X):
            sc = StandardScaler().fit(X.iloc[tr])
            m = Ridge(alpha=1.0).fit(sc.transform(X.iloc[tr]), yv[tr])
            pred[te] = m.predict(sc.transform(X.iloc[te]))
        r2 = 1 - np.sum((yv - pred) ** 2) / np.sum((yv - yv.mean()) ** 2)
        # permutation null on the CV R^2
        null = []
        for _ in range(200):
            ys = rng.permutation(yv)
            pr = np.zeros(len(ys))
            for tr, te in kf.split(X):
                sc = StandardScaler().fit(X.iloc[tr])
                m = Ridge(alpha=1.0).fit(sc.transform(X.iloc[tr]), ys[tr])
                pr[te] = m.predict(sc.transform(X.iloc[te]))
            null.append(1 - np.sum((ys - pr) ** 2) / np.sum((ys - ys.mean()) ** 2))
        p = (np.sum(np.array(null) >= r2) + 1) / 201
        summary[f"assay_cv_r2_{tgt}"] = float(r2)
        summary[f"assay_cv_r2_{tgt}_p"] = float(p)
        print(f"\n[assay] 5-fold CV R^2 predicting {tgt} from dataset covariates: "
              f"{r2:.3f} (perm p={p:.4f}, null mean {np.mean(null):.3f})")

    # ---------------------------------------------------------------- (B) mutant level
    # Within-assay z-scoring removes assay identity as a confound.
    P = P[P.n_muts == 1].copy()          # mutation-level features only defined for singles
    print(f"\n[mutant] {len(P):,} single mutants across {P.DMS_id.nunique()} assays")
    g = P.groupby("DMS_id")
    for c in ["EU", "AU", "abs_rank_err"]:
        P[f"z_{c}"] = (P[c] - g[c].transform("mean")) / g[c].transform("std").replace(0, np.nan)

    mfeat = ["blosum", "d_hydro", "d_vol", "d_charge", "rel_pos", "is_proline", "is_glycine_wt"]
    P["abs_d_hydro"] = P.d_hydro.abs()
    P["abs_d_vol"] = P.d_vol.abs()
    P["abs_d_charge"] = P.d_charge.abs()
    mfeat += ["abs_d_hydro", "abs_d_vol", "abs_d_charge"]

    rows = []
    sub = P.dropna(subset=["z_EU"]).sample(n=min(300_000, len(P)), random_state=SEED)
    for c in mfeat:
        for tgt in ["z_EU", "z_AU"]:
            ok = np.isfinite(sub[c]) & np.isfinite(sub[tgt])
            r = stats.spearmanr(sub.loc[ok, c], sub.loc[ok, tgt]).statistic
            # cluster-robust p: recompute rho per assay, one-sample t-test over assays
            per = sub[ok].groupby("DMS_id").apply(
                lambda d: stats.spearmanr(d[c], d[tgt]).statistic if d[c].nunique() >= 2 else np.nan,
                include_groups=False)
            per = per.dropna()
            t = stats.ttest_1samp(per, 0)
            rows.append({"level": "mutant", "feature": c, "target": tgt,
                         "spearman_pooled": r, "mean_per_assay_rho": per.mean(),
                         "n_assays": len(per), "t": t.statistic, "p_cluster": t.pvalue})
    mut_uni = pd.DataFrame(rows)
    mut_uni["p_fdr"] = bh_fdr(mut_uni.p_cluster)
    mut_uni.to_csv(RESULTS / "e1_mutant_univariate.csv", index=False)
    print("\n[mutant] mutation biology -> within-assay standardised conflict")
    print(mut_uni[mut_uni.target == "z_EU"].sort_values("mean_per_assay_rho")
          .to_string(index=False))

    # multivariate mutant model with GROUPED CV (held-out assays) -- the honest test:
    # can mutation-level biology predict conflict on proteins never seen in training?
    S = sub.dropna(subset=mfeat + ["z_EU"])
    Xm, ym, grp = S[mfeat].values, S["z_EU"].values, S["DMS_id"].values
    gkf = GroupKFold(5)
    pred = np.zeros(len(ym))
    for tr, te in gkf.split(Xm, ym, grp):
        m = GradientBoostingRegressor(random_state=SEED, n_estimators=200, max_depth=3,
                                      subsample=0.5)
        m.fit(Xm[tr], ym[tr])
        pred[te] = m.predict(Xm[te])
    r2 = 1 - np.sum((ym - pred) ** 2) / np.sum((ym - ym.mean()) ** 2)
    rho_pred = stats.spearmanr(pred, ym).statistic
    summary["mutant_groupcv_r2_zEU"] = float(r2)
    summary["mutant_groupcv_rho_zEU"] = float(rho_pred)
    print(f"\n[mutant] GroupKFold (held-out assays) R^2 for z_EU from mutation biology: "
          f"{r2:.4f}  (Spearman {rho_pred:.3f}, n={len(ym):,})")

    mfull = GradientBoostingRegressor(random_state=SEED, n_estimators=200, max_depth=3,
                                      subsample=0.5).fit(Xm, ym)
    imp = pd.DataFrame({"feature": mfeat, "importance": mfull.feature_importances_}) \
        .sort_values("importance", ascending=False)
    imp.to_csv(RESULTS / "e1_mutant_feature_importance.csv", index=False)
    print(imp.to_string(index=False))

    # amino-acid substitution conflict map: which wt->mut classes are most contested?
    aa = P.groupby(["wt_aa", "mut_aa"]).agg(z_EU=("z_EU", "mean"), n=("z_EU", "size"))
    aa = aa[aa.n >= 500].reset_index()
    aa.sort_values("z_EU", ascending=False).to_csv(RESULTS / "e1_aa_conflict_map.csv", index=False)
    print(f"\n[mutant] most-contested substitutions (>=500 obs):")
    print(aa.sort_values("z_EU", ascending=False).head(8).to_string(index=False))
    print(f"[mutant] least-contested:")
    print(aa.sort_values("z_EU").head(5).to_string(index=False))

    # ---------------------------------------------------------------- family structure
    # Which families disagree with which?  Correlation of family rank predictions.
    fam_cols = [f"fam_{f}" for f in FAM_NAMES]
    C = P[fam_cols].sample(n=min(200_000, len(P)), random_state=SEED).corr(method="spearman")
    C.to_csv(RESULTS / "e1_family_corr.csv")

    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].scatter(A.log_Neff_L, A.EU, s=18, alpha=.7, c="#2f6f9f")
    r, p, _ = perm_test_spearman(A.log_Neff_L, A.EU)
    ax[0].set_xlabel("log$_{10}$ MSA depth (N$_{eff}$/L)")
    ax[0].set_ylabel("Assay mean epistemic conflict (EU)")
    ax[0].set_title(f"Conflict vs evolutionary data depth\n$\\rho$={r:.3f}, perm p={p:.4f}")
    ax[0].grid(alpha=.25)

    order = A.groupby("taxon").EU.mean().sort_values(ascending=False)
    dat = [A.loc[A.taxon == t, "EU"].values for t in order.index]
    bp = ax[1].boxplot(dat, tick_labels=list(order.index), patch_artist=True)
    for b in bp["boxes"]:
        b.set_facecolor("#7fb3d5")
    ax[1].set_ylabel("Assay mean EU")
    ax[1].set_title("Conflict by taxon")
    ax[1].tick_params(axis="x", rotation=25)
    ax[1].grid(alpha=.25, axis="y")

    ax[2].scatter(A.EU, A.rho_ens, s=18, alpha=.7, c="#b03a2e")
    r2_, p2_, _ = perm_test_spearman(A.EU, A.rho_ens)
    ax[2].set_xlabel("Assay mean EU")
    ax[2].set_ylabel("Ensemble Spearman $\\rho$")
    ax[2].set_title(f"Conflict predicts where the ensemble fails\n$\\rho$={r2_:.3f}, perm p={p2_:.4f}")
    ax[2].grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig1_conflict_cartography.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(C.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(FAM_NAMES)), FAM_NAMES, rotation=90, fontsize=8)
    ax.set_yticks(range(len(FAM_NAMES)), FAM_NAMES, fontsize=8)
    ax.set_title("Cross-family Spearman agreement (per-mutant rank predictions)")
    plt.colorbar(im, label="Spearman $\\rho$")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig2_family_agreement.png", dpi=150)
    plt.close()

    summary["n_assays"] = int(len(A))
    summary["n_mutants"] = int(len(P))
    summary["mean_rho_ens"] = float(A.rho_ens.mean())
    summary["mean_rho_best_single"] = float(A.rho_best_single.mean())
    summary["mean_family_corr_offdiag"] = float(
        C.values[np.triu_indices(len(FAM_NAMES), 1)].mean())
    json.dump(summary, open(RESULTS / "e1_summary.json", "w"), indent=2)
    print("\n[e1b] summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
