"""EXPERIMENT 2 — Conflict as a failure-mode detector (direction D2, claim C3).

Three tests, all designed around the Phase-1 finding that per-mutant conflict correlates
only weakly (r~0.05) with per-mutant |rank error|.  A weak pointwise correlation does NOT
imply a useless detector: selective prediction only needs the *ordering* induced by the
signal to be informative in aggregate.  So we evaluate with stratified / selective metrics.

  T1 SELECTIVE PREDICTION.  Rank mutants within each assay by a risk signal, abstain on
     the top-q fraction, and measure the ensemble's Spearman on the retained mutants.
     Signals compared: EU (cross-family conflict), AU (within-family noise), TU,
     naive unbalanced SD, |ensemble score - 0.5| (confidence-magnitude baseline),
     and random.  Random abstention is the correct null: removing any mutants at all
     changes the score distribution, so "conflict helps" must be measured against it.

  T2 CONFIDENT-FAILURE TEST.  The sharp, falsifiable prediction transplanted from
     Hamidieh et al. 2026: within-family agreement (low AU) plus cross-family
     disagreement (high EU) marks *confident failures* -- cases a self-consistency style
     uncertainty estimate would call certain but that are actually wrong.  We test the
     2x2 AU x EU median split, per assay, paired across assays.

  T3 SLICE DISCOVERY.  Cluster high-conflict mutants in a biological feature space, then
     check the resulting slices transfer: are they still high-conflict AND high-error on
     assays never used to define them?  Held-out validation distinguishes a real failure
     mode from a post-hoc description.

Outputs: results/e2_*.csv, results/e2_summary.json, figures/fig3-4_*.png
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS, FIGURES, SEED

rng = np.random.default_rng(SEED)
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
N_BOOT = 2000


def boot_ci(vals, n=N_BOOT, alpha=0.05):
    """Bootstrap CI over assays (the unit of analysis)."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan
    idx = rng.integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(1)
    return float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2)))


def selective_curve(P, signal_col, higher_is_riskier=True, seed=None):
    """For each coverage level, keep the lowest-risk fraction of mutants per assay and
    return the mean per-assay Spearman of the ensemble against ground truth."""
    out = {}
    for cov in COVERAGES:
        rhos = []
        for dms, d in P.groupby("DMS_id", sort=False):
            k = int(round(cov * len(d)))
            if k < 30:
                continue
            if signal_col == "random":
                r = np.random.default_rng(seed).permutation(len(d))
                keep = d.iloc[np.argsort(r)[:k]]
            else:
                s = d[signal_col].to_numpy(float)
                order = np.argsort(s if higher_is_riskier else -s)
                keep = d.iloc[order[:k]]
            if keep.y.nunique() < 5:
                continue
            rhos.append(stats.spearmanr(keep.ens, keep.y).statistic)
        out[cov] = np.array(rhos, float)
    return out


def main():
    P = pd.read_parquet(RESULTS / "panel_mutants.parquet")
    A = pd.read_csv(RESULTS / "panel_assays.csv")
    summary = {}
    P["conf_mag"] = -np.abs(P["ens_rank"] - 0.5)   # low |score-0.5| == risky (baseline)
    print(f"[e2] {len(P):,} mutants / {P.DMS_id.nunique()} assays")

    # ------------------------------------------------------------------ T1 selective
    signals = {"EU": ("EU", True), "AU": ("AU", True), "TU": ("TU", True),
               "raw_sd": ("raw_sd", True), "conf_mag": ("conf_mag", True)}
    curves = {}
    for name, (col, hi) in signals.items():
        curves[name] = selective_curve(P, col, hi)
        print(f"  computed selective curve: {name}", flush=True)
    # random: 5 seeds averaged, keeps the null tight
    rand_runs = [selective_curve(P, "random", seed=s) for s in range(5)]
    curves["random"] = {c: np.mean([r[c] for r in rand_runs], axis=0) for c in COVERAGES}
    print("  computed selective curve: random", flush=True)

    rows = []
    base = curves["random"]
    for name, cv in curves.items():
        for c in COVERAGES:
            v = cv[c]
            lo, hi_ = boot_ci(v)
            d = v - base[c]                     # paired per-assay difference vs random
            t = stats.ttest_rel(v, base[c]) if c < 1.0 else None
            w = stats.wilcoxon(d) if (c < 1.0 and np.any(d != 0)) else None
            rows.append({"signal": name, "coverage": c, "sel_spearman": float(v.mean()),
                         "ci_lo": lo, "ci_hi": hi_, "n_assays": len(v),
                         "delta_vs_random": float(d.mean()),
                         "cohens_d": float(d.mean() / d.std(ddof=1)) if c < 1.0 and d.std() > 0 else np.nan,
                         "p_paired_t": float(t.pvalue) if t is not None else np.nan,
                         "p_wilcoxon": float(w.pvalue) if w is not None else np.nan})
    sel = pd.DataFrame(rows)
    sel.to_csv(RESULTS / "e2_selective_prediction.csv", index=False)
    piv = sel.pivot(index="coverage", columns="signal", values="sel_spearman")
    print("\n[T1] Selective Spearman by coverage (rows = fraction of mutants retained)")
    print(piv.round(4).to_string())
    print("\n[T1] EU vs random, paired over assays:")
    print(sel[(sel.signal == "EU") & (sel.coverage < 1.0)]
          [["coverage", "sel_spearman", "delta_vs_random", "cohens_d", "p_wilcoxon"]]
          .round(5).to_string(index=False))

    # AURC-style summary: area under the selective-Spearman curve (higher = better)
    for name, cv in curves.items():
        xs = np.array(sorted(COVERAGES))
        ys = np.array([cv[c].mean() for c in xs])
        summary[f"AUC_selective_{name}"] = float(np.trapezoid(ys, xs) / (xs[-1] - xs[0]))
    print("\n[T1] Area under selective-Spearman curve:",
          {k.replace("AUC_selective_", ""): round(v, 4)
           for k, v in summary.items() if k.startswith("AUC_selective")})

    # ------------------------------------------------------------------ T2 confident failure
    cells = []
    for dms, d in P.groupby("DMS_id", sort=False):
        if len(d) < 200:
            continue
        au_hi = d.AU > d.AU.median()
        eu_hi = d.EU > d.EU.median()
        rec = {"DMS_id": dms}
        for an, am in [("loAU", ~au_hi), ("hiAU", au_hi)]:
            for en, em in [("loEU", ~eu_hi), ("hiEU", eu_hi)]:
                m = am & em
                if m.sum() < 30 or d.loc[m, "y"].nunique() < 5:
                    rec[f"{an}_{en}"] = np.nan
                    rec[f"err_{an}_{en}"] = np.nan
                else:
                    rec[f"{an}_{en}"] = stats.spearmanr(d.loc[m, "ens"], d.loc[m, "y"]).statistic
                    rec[f"err_{an}_{en}"] = d.loc[m, "abs_rank_err"].mean()
        cells.append(rec)
    CF = pd.DataFrame(cells)
    CF.to_csv(RESULTS / "e2_confident_failure_cells.csv", index=False)
    print(f"\n[T2] Confident-failure 2x2 (n={len(CF)} assays)")
    t2 = {}
    for k in ["loAU_loEU", "loAU_hiEU", "hiAU_loEU", "hiAU_hiEU"]:
        t2[k] = {"rho": float(CF[k].mean()), "err": float(CF["err_" + k].mean()),
                 "ci": boot_ci(CF[k])}
        print(f"   {k}: ensemble rho = {CF[k].mean():.4f} "
              f"[{t2[k]['ci'][0]:.4f},{t2[k]['ci'][1]:.4f}]   mean |rank err| = {CF['err_'+k].mean():.4f}")
    # the key contrast: within the LOW-AU (apparently confident) stratum, does high EU hurt?
    dd = (CF.loAU_loEU - CF.loAU_hiEU).dropna()
    w = stats.wilcoxon(dd)
    t2["confident_failure_delta_rho"] = float(dd.mean())
    t2["confident_failure_p"] = float(w.pvalue)
    t2["confident_failure_cohens_d"] = float(dd.mean() / dd.std(ddof=1))
    print(f"   >>> low-AU stratum: rho(loEU) - rho(hiEU) = {dd.mean():.4f} "
          f"(Wilcoxon p={w.pvalue:.3e}, d={t2['confident_failure_cohens_d']:.2f}, n={len(dd)})")
    de = (CF.err_loAU_hiEU - CF.err_loAU_loEU).dropna()
    t2["confident_failure_delta_err"] = float(de.mean())
    t2["confident_failure_err_p"] = float(stats.wilcoxon(de).pvalue)
    print(f"   >>> low-AU stratum: |rank err|(hiEU) - |rank err|(loEU) = {de.mean():.4f} "
          f"(p={stats.wilcoxon(de).pvalue:.3e})")
    summary["T2"] = t2

    # ------------------------------------------------------------------ T3 slice discovery
    S = P[(P.n_muts == 1)].dropna(subset=["blosum", "d_hydro", "d_vol", "rel_pos"]).copy()
    g = S.groupby("DMS_id")
    S["z_EU"] = (S.EU - g.EU.transform("mean")) / g.EU.transform("std").replace(0, np.nan)
    S["z_err"] = (S.abs_rank_err - g.abs_rank_err.transform("mean")) / \
                 g.abs_rank_err.transform("std").replace(0, np.nan)
    S = S.dropna(subset=["z_EU", "z_err"])

    assays = sorted(S.DMS_id.unique())
    rs = np.random.default_rng(SEED)
    perm = rs.permutation(len(assays))
    dev = set(np.array(assays)[perm[: int(0.6 * len(assays))]])
    hold = set(np.array(assays)[perm[int(0.6 * len(assays)):]])
    feats = ["blosum", "d_hydro", "d_vol", "d_charge", "rel_pos"]

    Sdev = S[S.DMS_id.isin(dev)].sample(n=min(200_000, (S.DMS_id.isin(dev)).sum()), random_state=SEED)
    hi = Sdev[Sdev.z_EU > Sdev.z_EU.quantile(0.8)]          # the high-conflict region
    sc = StandardScaler().fit(hi[feats])
    km = KMeans(n_clusters=6, random_state=SEED, n_init=10).fit(sc.transform(hi[feats]))

    Shold = S[S.DMS_id.isin(hold)]
    lab_hold = km.predict(sc.transform(Shold[feats]))
    rows = []
    for k in range(6):
        m_dev = km.labels_ == k
        m_h = lab_hold == k
        rows.append({
            "slice": k, "n_dev": int(m_dev.sum()), "n_hold": int(m_h.sum()),
            "dev_z_EU": float(hi.z_EU.values[m_dev].mean()),
            "hold_z_EU": float(Shold.z_EU.values[m_h].mean()) if m_h.sum() > 50 else np.nan,
            "hold_z_err": float(Shold.z_err.values[m_h].mean()) if m_h.sum() > 50 else np.nan,
            **{f"c_{f}": float(hi[f].values[m_dev].mean()) for f in feats},
            "top_wt": hi.wt_aa.values[m_dev][:0].tolist() or
                      pd.Series(hi.wt_aa.values[m_dev]).value_counts().head(3).index.tolist(),
            "top_mut": pd.Series(hi.mut_aa.values[m_dev]).value_counts().head(3).index.tolist(),
        })
    SL = pd.DataFrame(rows)
    # held-out significance: is slice mean z_err > 0 (i.e. worse than assay average)?
    for k in range(6):
        m_h = lab_hold == k
        if m_h.sum() > 50:
            v = Shold.z_err.values[m_h]
            SL.loc[SL.slice == k, "hold_err_p"] = stats.ttest_1samp(v, 0).pvalue
    SL.to_csv(RESULTS / "e2_slices.csv", index=False)
    print("\n[T3] Slice discovery (clusters fitted on 60% of assays, evaluated on held-out 40%)")
    print(SL[["slice", "n_dev", "n_hold", "dev_z_EU", "hold_z_EU", "hold_z_err",
              "hold_err_p", "c_blosum", "c_rel_pos", "top_wt", "top_mut"]].round(4).to_string(index=False))
    summary["T3_slice_transfer_rho"] = float(
        stats.spearmanr(SL.dev_z_EU, SL.hold_z_EU, nan_policy="omit").statistic)
    print(f"   slice-level dev->holdout conflict transfer: "
          f"rho={summary['T3_slice_transfer_rho']:.3f}")
    SL.to_json(RESULTS / "e2_slices.json", orient="records", indent=2)

    # ------------------------------------------------------------------ figures
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    cols = {"EU": "#b03a2e", "AU": "#e59866", "TU": "#7d3c98", "raw_sd": "#5d6d7e",
            "conf_mag": "#1e8449", "random": "#000000"}
    for name, cv in curves.items():
        xs = sorted(COVERAGES)
        ys = [cv[c].mean() for c in xs]
        lo = [boot_ci(cv[c])[0] for c in xs]
        hi_ = [boot_ci(cv[c])[1] for c in xs]
        ls = "--" if name == "random" else "-"
        ax[0].plot(xs, ys, ls, marker="o", ms=3.5, color=cols[name], label=name)
        ax[0].fill_between(xs, lo, hi_, color=cols[name], alpha=.12)
    ax[0].set_xlabel("coverage (fraction of mutants retained)")
    ax[0].set_ylabel("selective Spearman $\\rho$ (mean over assays)")
    ax[0].set_title("Risk-coverage: abstain on high-conflict mutants")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.25)
    ax[0].invert_xaxis()

    labels = ["low AU\nlow EU", "low AU\nhigh EU", "high AU\nlow EU", "high AU\nhigh EU"]
    keys = ["loAU_loEU", "loAU_hiEU", "hiAU_loEU", "hiAU_hiEU"]
    means = [CF[k].mean() for k in keys]
    errs = [[CF[k].mean() - boot_ci(CF[k])[0] for k in keys],
            [boot_ci(CF[k])[1] - CF[k].mean() for k in keys]]
    bars = ax[1].bar(labels, means, yerr=errs, capsize=4,
                     color=["#1e8449", "#b03a2e", "#7fb3d5", "#e59866"])
    ax[1].set_ylabel("ensemble Spearman $\\rho$ within stratum")
    ax[1].set_title("Confident-failure test: low AU + high EU is the danger zone")
    ax[1].grid(alpha=.25, axis="y")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig3_failure_detection.png", dpi=150)
    plt.close()

    json.dump(summary, open(RESULTS / "e2_summary.json", "w"), indent=2, default=float)
    print("\n[e2] wrote results/e2_summary.json")


if __name__ == "__main__":
    main()
