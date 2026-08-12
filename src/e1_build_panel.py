"""EXPERIMENT 1a — Build the per-mutant conflict panel (direction D1).

For every ProteinGym substitution assay we:
  1. load the 30 panel model columns,
  2. convert each model's raw score to a within-assay normalised rank in [0,1]
     (scores from different models live on wildly different scales: log-likelihood
     ratios, Potts energies, GEMME ranks -- ranks are the only common currency),
  3. compute per-mutant uncertainty decomposition following the AU/EU formalism of
     Hamidieh et al. 2026 (arXiv:2604.17112), transplanted from LLM self-consistency
     to protein model families:
         AU  = mean over families of the within-family standard deviation
               (aleatoric-analogue: noise inside a modelling paradigm)
         EU  = standard deviation across the 10 family means
               (epistemic-analogue: genuine cross-paradigm disagreement)
         TU  = AU + EU
  4. compute the family-balanced ensemble prediction (mean of family means),
  5. store per-mutant ground truth rank and mutation-level features.

Output: results/panel_mutants.parquet  (one row per (DMS_id, mutant))
        results/panel_assays.csv       (one row per assay, with metadata covariates)
"""
import sys, time, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import (SCORES, REF, RESULTS, FAMILIES, PANEL, FAM_NAMES, SEED, BEST_SINGLE)

np.random.seed(SEED)

# BLOSUM62 substitution matrix (upper-triangular text form, parsed below).
BLOSUM62_TXT = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


def parse_blosum():
    lines = [l for l in BLOSUM62_TXT.strip().split("\n") if l.strip()]
    aas = lines[0].split()
    m = {}
    for line in lines[1:]:
        p = line.split()
        for a2, v in zip(aas, p[1:]):
            m[(p[0], a2)] = int(v)
    return m


BLOSUM = parse_blosum()
# Kyte-Doolittle hydropathy
KD = dict(zip("ARNDCQEGHILKMFPSTWYV",
              [1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5,
               3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, -0.9, -1.3, 4.2]))
VOL = dict(zip("ARNDCQEGHILKMFPSTWYV",
               [88.6, 173.4, 114.1, 111.1, 108.5, 143.8, 138.4, 60.1, 153.2, 166.7,
                166.7, 168.6, 162.9, 189.9, 112.7, 89.0, 116.1, 227.8, 193.6, 140.0]))
CHARGE = {a: 0 for a in "ARNDCQEGHILKMFPSTWYV"}
CHARGE.update({"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.5})


def norm_rank(x: np.ndarray) -> np.ndarray:
    """Within-assay normalised rank in [0,1]; ties averaged."""
    return (rankdata(x) - 0.5) / len(x)


def mutation_features(mutants: pd.Series, seq_len: int) -> pd.DataFrame:
    """Parse ProteinGym mutant strings (e.g. 'A673C', or ':'-joined multiples)."""
    wt, pos, mt = [], [], []
    for s in mutants:
        first = s.split(":")[0]
        wt.append(first[0])
        mt.append(first[-1])
        try:
            pos.append(int(first[1:-1]))
        except ValueError:
            pos.append(np.nan)
    df = pd.DataFrame({"wt_aa": wt, "mut_aa": mt, "pos": pos})
    df["n_muts"] = [s.count(":") + 1 for s in mutants]
    df["rel_pos"] = df["pos"] / max(seq_len, 1)
    df["blosum"] = [BLOSUM.get((a, b), np.nan) for a, b in zip(df.wt_aa, df.mut_aa)]
    df["d_hydro"] = [KD.get(b, np.nan) - KD.get(a, np.nan) for a, b in zip(df.wt_aa, df.mut_aa)]
    df["d_vol"] = [VOL.get(b, np.nan) - VOL.get(a, np.nan) for a, b in zip(df.wt_aa, df.mut_aa)]
    df["d_charge"] = [CHARGE.get(b, np.nan) - CHARGE.get(a, np.nan) for a, b in zip(df.wt_aa, df.mut_aa)]
    df["is_proline"] = (df.mut_aa == "P").astype(int)
    df["is_glycine_wt"] = (df.wt_aa == "G").astype(int)
    df["to_stop_like"] = 0
    return df


def main():
    ref = pd.read_csv(REF)
    ref = ref.set_index("DMS_id")
    t0 = time.time()

    mut_rows, assay_rows = [], []
    files = sorted(SCORES.glob("*.csv"))
    print(f"[e1] {len(files)} assay score files; panel = {len(PANEL)} models "
          f"in {len(FAMILIES)} families", flush=True)

    skipped = []
    for i, f in enumerate(files):
        dms_id = f.stem
        d = pd.read_csv(f, low_memory=False)
        missing = [m for m in PANEL if m not in d.columns]
        if missing:
            skipped.append((dms_id, f"missing cols {missing[:3]}"))
            continue
        sub = d[PANEL]
        # Require full coverage: a partially-scored model would bias the family mean.
        if sub.isna().any().any() or len(d) < 100:
            skipped.append((dms_id, "NaNs or <100 mutants"))
            continue

        n = len(d)
        R = np.column_stack([norm_rank(sub[m].to_numpy(float)) for m in PANEL])  # n x 30
        y = d["DMS_score"].to_numpy(float)
        y_rank = norm_rank(y)

        # family means and within-family SDs
        fam_mean = np.zeros((n, len(FAM_NAMES)))
        fam_sd = np.zeros((n, len(FAM_NAMES)))
        for k, fam in enumerate(FAM_NAMES):
            idx = [PANEL.index(m) for m in FAMILIES[fam]]
            fam_mean[:, k] = R[:, idx].mean(1)
            fam_sd[:, k] = R[:, idx].std(1, ddof=1)

        AU = fam_sd.mean(1)                 # aleatoric-analogue
        EU = fam_mean.std(1, ddof=1)        # epistemic-analogue
        TU = AU + EU
        ens = fam_mean.mean(1)              # family-balanced ensemble prediction
        raw_sd = R.std(1, ddof=1)           # naive (unbalanced) disagreement

        feats = mutation_features(d["mutant"], ref.loc[dms_id, "seq_len"] if dms_id in ref.index else 1)

        row = pd.DataFrame({
            "DMS_id": dms_id, "mutant": d["mutant"].values,
            "y": y, "y_rank": y_rank, "y_bin": d["DMS_score_bin"].values,
            "ens": ens, "AU": AU, "EU": EU, "TU": TU, "raw_sd": raw_sd,
            "ens_rank": norm_rank(ens),
        })
        for k, fam in enumerate(FAM_NAMES):
            row[f"fam_{fam}"] = fam_mean[:, k]
        row[f"single_{BEST_SINGLE}"] = norm_rank(sub[BEST_SINGLE].to_numpy(float))
        row = pd.concat([row, feats.reset_index(drop=True)], axis=1)
        row["abs_rank_err"] = np.abs(row["ens_rank"] - row["y_rank"])
        mut_rows.append(row)

        rho = spearmanr(ens, y).statistic
        rho_best = spearmanr(sub[BEST_SINGLE], y).statistic
        a = {"DMS_id": dms_id, "n": n, "rho_ens": rho, "rho_best_single": rho_best,
             "AU": AU.mean(), "EU": EU.mean(), "TU": TU.mean(), "raw_sd": raw_sd.mean()}
        for k, fam in enumerate(FAM_NAMES):
            a[f"rho_{fam}"] = spearmanr(fam_mean[:, k], y).statistic
        if dms_id in ref.index:
            r = ref.loc[dms_id]
            for c in ["taxon", "coarse_selection_type", "selection_assay", "seq_len",
                      "MSA_Neff_L", "MSA_num_seqs", "MSA_perc_cov", "MSA_len",
                      "DMS_number_single_mutants", "DMS_number_multiple_mutants",
                      "includes_multiple_mutants", "region_mutated", "UniProt_ID",
                      "DMS_binarization_method", "MSA_Neff_L_category", "source_organism"]:
                a[c] = r[c] if c in r.index else np.nan
        assay_rows.append(a)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(files)}] {time.time()-t0:.0f}s", flush=True)

    panel = pd.concat(mut_rows, ignore_index=True)
    assays = pd.DataFrame(assay_rows)
    panel.to_parquet(RESULTS / "panel_mutants.parquet", index=False)
    assays.to_csv(RESULTS / "panel_assays.csv", index=False)

    print(f"\n[e1] DONE in {time.time()-t0:.0f}s")
    print(f"  assays kept  : {len(assays)}  (skipped {len(skipped)})")
    for s in skipped[:10]:
        print("    skip:", s)
    print(f"  mutants      : {len(panel):,}")
    print(f"  mean ens rho : {assays.rho_ens.mean():.4f}   "
          f"best-single ({BEST_SINGLE}) rho: {assays.rho_best_single.mean():.4f}")
    print(f"  mean AU {panel.AU.mean():.4f}  mean EU {panel.EU.mean():.4f}")
    json.dump({"n_assays": int(len(assays)), "n_mutants": int(len(panel)),
               "skipped": skipped, "panel": PANEL,
               "families": {k: v for k, v in FAMILIES.items()}},
              open(RESULTS / "panel_build_meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
