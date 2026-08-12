"""EXPERIMENT 5 — Does conflict-awareness improve *variant selection* under a lab budget?

The hypothesis's final clause is about "protein sequence design performance".  The standard
budget-constrained proxy in this literature: from a candidate pool the model ranks, pick K
variants to synthesise, and score how good the best picked variant is.

Three questions:
  Q1 HEADROOM.  If an oracle could perfectly re-rank only the HIGH-CONFLICT variants inside
     the candidate pool, how much better would the selection be?  This upper-bounds what any
     conflict-driven arbiter -- LLM or otherwise -- could possibly deliver, and is therefore
     worth computing before spending inference on one.
  Q2 CONFLICT-PENALISED SELECTION.  A parameter-free "targeted consensus" rule: rank the pool
     by ensemble score minus lambda * conflict, i.e. prefer variants the model families agree
     on.  Directly operationalises "resolve the conflict points by avoiding them".
  Q3 ATTRIBUTION.  Are the selection mistakes actually concentrated on high-conflict variants?

Metrics (per assay, budget K):
  max_fitness@K   normalised max true fitness among the K picked variants (0 = worst variant
                  in the assay, 1 = best variant in the assay)
  hit@K           whether any of the K picked variants is in the true top 1%
  mean_fitness@K  normalised mean true fitness of the K picked
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS, FIGURES, SEED

rng = np.random.default_rng(SEED)
POOL = 100          # candidate pool: the variants the ensemble ranks highest
BUDGET = [10, 20, 30, 40]
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def metrics(y_sel, ymin, rng_y, true_top):
    return {"max": float((y_sel.max() - ymin) / rng_y),
            "mean": float((y_sel.mean() - ymin) / rng_y)}


def main():
    P = pd.read_parquet(RESULTS / "panel_mutants.parquet")
    Pr = pd.read_csv(RESULTS / "e3_assay_profiles.csv").set_index("DMS_id")
    rows = []
    for dms, d in P.groupby("DMS_id", sort=False):
        if len(d) < 300:
            continue
        y = d.y.to_numpy(float)
        ens = d.ens.to_numpy(float)
        eu = d.EU.to_numpy(float)
        ymin, ymax = y.min(), y.max()
        rng_y = (ymax - ymin) or 1.0
        true_top = set(np.argsort(-y)[: max(1, int(0.01 * len(y)))])
        pool = np.argsort(-ens)[:POOL]
        # within-pool standardised signals so lambda has a consistent meaning across assays
        ez = (ens[pool] - ens[pool].mean()) / (ens[pool].std() + 1e-12)
        cz = (eu[pool] - eu[pool].mean()) / (eu[pool].std() + 1e-12)

        for K in BUDGET:
            base = pool[np.argsort(-ez)[:K]]
            rec = {"DMS_id": dms, "K": K, "n": len(d),
                   "base_max": (y[base].max() - ymin) / rng_y,
                   "base_mean": (y[base].mean() - ymin) / rng_y,
                   "base_hit": int(len(set(base) & true_top) > 0)}
            # Q2: conflict-penalised selection
            for lam in LAMBDAS[1:]:
                sel = pool[np.argsort(-(ez - lam * cz))[:K]]
                rec[f"pen{lam}_max"] = (y[sel].max() - ymin) / rng_y
                rec[f"pen{lam}_mean"] = (y[sel].mean() - ymin) / rng_y
                rec[f"pen{lam}_hit"] = int(len(set(sel) & true_top) > 0)
            # random-K-from-pool control (does any deviation from the ensemble order help?)
            rsel = pool[rng.permutation(len(pool))[:K]]
            rec["rand_max"] = (y[rsel].max() - ymin) / rng_y
            rec["rand_mean"] = (y[rsel].mean() - ymin) / rng_y
            rec["rand_hit"] = int(len(set(rsel) & true_top) > 0)

            # Q1: oracle re-ranking restricted to the high-conflict half of the pool.
            # Low-conflict pool members keep their ensemble rank; high-conflict members are
            # re-ordered perfectly.  This is the ceiling for a per-mutant conflict arbiter.
            # Controls: the same oracle applied to a RANDOM half and to the LOW-conflict
            # half of the pool.  Without these, "targeting conflict recovers X% of the
            # oracle gain" is uninterpretable -- fixing any half of the pool helps.
            yr = stats.rankdata(y[pool])

            def partial_oracle(mask):
                score = ez.copy()
                if mask.sum() > 1:
                    score[mask] = np.interp(yr[mask], (yr[mask].min(), yr[mask].max()),
                                            (ez[mask].min(), ez[mask].max()))
                s = pool[np.argsort(-score)[:K]]
                return ((y[s].max() - ymin) / rng_y, (y[s].mean() - ymin) / rng_y,
                        int(len(set(s) & true_top) > 0))

            hi = cz > np.median(cz)
            rand_half = np.zeros(len(pool), bool)
            rand_half[rng.permutation(len(pool))[: hi.sum()]] = True
            for nm, msk in [("oracle_hiconf", hi), ("oracle_loconf", ~hi),
                            ("oracle_randhalf", rand_half)]:
                mx, mn, ht = partial_oracle(msk)
                rec[f"{nm}_max"], rec[f"{nm}_mean"], rec[f"{nm}_hit"] = mx, mn, ht
            # full oracle over the whole pool, for reference
            fsel = pool[np.argsort(-y[pool])[:K]]
            rec["oracle_full_max"] = (y[fsel].max() - ymin) / rng_y
            rec["oracle_full_mean"] = (y[fsel].mean() - ymin) / rng_y
            rec["oracle_full_hit"] = int(len(set(fsel) & true_top) > 0)

            # Q3: attribution -- are the picked-but-bad variants the high-conflict ones?
            picked = np.isin(pool, base)
            bad = y[pool] < np.median(y[pool])
            if picked.sum() and (picked & bad).sum() >= 1:
                rec["conflict_of_picked_bad"] = float(cz[picked & bad].mean())
                rec["conflict_of_picked_good"] = float(cz[picked & ~bad].mean()) \
                    if (picked & ~bad).sum() else np.nan
            rows.append(rec)

    D = pd.DataFrame(rows)
    D.to_csv(RESULTS / "e5_design_selection_per_assay.csv", index=False)
    print(f"[e5] {D.DMS_id.nunique()} assays, pool = top-{POOL} by ensemble\n")

    summ = []
    for K in BUDGET:
        d = D[D.K == K]
        base = d.base_max.values
        rec = {"K": K, "n_assays": len(d), "base_max": base.mean(),
               "base_hit": d.base_hit.mean(), "base_mean": d.base_mean.mean()}
        for name in ["rand", "oracle_hiconf", "oracle_loconf", "oracle_randhalf", "oracle_full"] + [f"pen{l}" for l in LAMBDAS[1:]]:
            v = d[f"{name}_max"].values
            diff = v - base
            rec[f"{name}_max"] = v.mean()
            rec[f"{name}_hit"] = d[f"{name}_hit"].mean()
            rec[f"{name}_delta"] = diff.mean()
            rec[f"{name}_p"] = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else np.nan
            rec[f"{name}_d"] = diff.mean() / diff.std(ddof=1) if diff.std() > 0 else np.nan
        summ.append(rec)
    S = pd.DataFrame(summ)
    S.to_csv(RESULTS / "e5_design_selection_summary.csv", index=False)

    cols = ["K", "n_assays", "base_max", "rand_max", "pen0.5_max", "pen1.0_max",
            "oracle_loconf_max", "oracle_randhalf_max", "oracle_hiconf_max", "oracle_full_max"]
    print("--- Q1/Q2: normalised MAX true fitness among the K selected variants ---")
    print(S[cols].round(4).to_string(index=False))
    print("\n--- vs the plain ensemble top-K (paired Wilcoxon over assays) ---")
    pc = ["K"] + [f"{n}_delta" for n in ["pen0.5", "pen1.0", "oracle_loconf",
                                         "oracle_randhalf", "oracle_hiconf", "oracle_full"]]
    print(S[pc].round(5).to_string(index=False))
    print("\n--- p-values ---")
    pp = ["K"] + [f"{n}_p" for n in ["pen0.5", "pen1.0", "oracle_loconf",
                                     "oracle_randhalf", "oracle_hiconf", "oracle_full"]]
    print(S[pp].round(5).to_string(index=False))
    print("\n--- hit rate on the true top 1% ---")
    hc = ["K", "base_hit"] + [f"{n}_hit" for n in ["pen0.5", "oracle_randhalf",
                                                   "oracle_hiconf", "oracle_full"]]
    print(S[hc].round(4).to_string(index=False))

    d20 = D[D.K == 20]
    dd = (d20.conflict_of_picked_bad - d20.conflict_of_picked_good).dropna()
    print(f"\n--- Q3 attribution (K=20): standardised conflict of picked-but-poor variants "
          f"minus picked-and-good = {dd.mean():.4f} "
          f"(Wilcoxon p={stats.wilcoxon(dd).pvalue:.4g}, n={len(dd)})")

    # what fraction of the achievable oracle gain is reachable by re-ranking only the
    # high-conflict half of the pool?
    for K in BUDGET:
        r = S[S.K == K].iloc[0]
        fh = r["oracle_hiconf_delta"] / r["oracle_full_delta"] if r["oracle_full_delta"] else np.nan
        fr = r["oracle_randhalf_delta"] / r["oracle_full_delta"] if r["oracle_full_delta"] else np.nan
        fl = r["oracle_loconf_delta"] / r["oracle_full_delta"] if r["oracle_full_delta"] else np.nan
        print(f"    K={K}: fraction of the full-oracle gain recovered by fixing half the pool -- "
              f"high-conflict {100*fh:.1f}%  random {100*fr:.1f}%  low-conflict {100*fl:.1f}%")

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(BUDGET))
    series = [("base_max", "ensemble top-K", "#5d6d7e"),
              ("pen1.0_max", "conflict-penalised (λ=1.0)", "#e59866"),
              ("oracle_loconf_max", "oracle on LOW-conflict half", "#aab7b8"),
              ("oracle_randhalf_max", "oracle on RANDOM half", "#2f6f9f"),
              ("oracle_hiconf_max", "oracle on HIGH-conflict half", "#7d3c98"),
              ("oracle_full_max", "full oracle within pool", "#1e8449")]
    w = 0.14
    for i, (c, lab, col) in enumerate(series):
        ax[0].bar(x + (i - 2.5) * w, S[c], width=w, label=lab, color=col)
    ax[0].set_xticks(x, [f"K={k}" for k in BUDGET])
    ax[0].set_ylim(0.75, 1.0)
    ax[0].set_ylabel("normalised max fitness among selected")
    ax[0].set_title(f"Budget-constrained variant selection (pool = ensemble top-{POOL})")
    ax[0].legend(fontsize=7.5)
    ax[0].grid(alpha=.25, axis="y")

    lam_all = [0.0] + LAMBDAS[1:]
    d20s = S[S.K == 20].iloc[0]
    vals = [d20s["base_max"]] + [d20s[f"pen{l}_max"] for l in LAMBDAS[1:]]
    ax[1].plot(lam_all, vals, "o-", color="#b03a2e")
    ax[1].axhline(d20s["base_max"], ls="--", color="k", lw=1, label="no conflict penalty")
    ax[1].set_xlabel("conflict penalty λ")
    ax[1].set_ylabel("normalised max fitness @ K=20")
    ax[1].set_title("Penalising disagreement does not improve selection")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig7_design_selection.png", dpi=150)
    plt.close()

    json.dump(S.to_dict("records"), open(RESULTS / "e5_summary.json", "w"),
              indent=2, default=float)
    print("\n[e5] done")


if __name__ == "__main__":
    main()
