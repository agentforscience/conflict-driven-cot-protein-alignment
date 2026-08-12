"""EXPERIMENT 3a — Build per-assay conflict profiles and the four prompt conditions.

Design (direction D3, hypothesis claims C2 + C4).
------------------------------------------------------------------------------------
E1/E2 established that cross-model conflict is strongly structured at the *assay/stratum*
level (CV R^2 = 0.41 from dataset covariates) but close to unstructured at the individual
mutant level (error-detection AUROC ~0.51).  We therefore aim the LLM arbiter at the level
where the signal actually lives: it is asked to reweight the 10 model families **for a
given assay**, after being shown how those families disagree on that assay.

The 2x2 ablation isolates the two things the hypothesis conflates:
                         | conflict profile shown | conflict profile hidden
    chain-of-thought     | C1 conflict_cot (OURS) | C3 cot_only  (compute-matched control)
    direct answer        | C2 conflict_direct     | C4 direct_only (metadata-only control)

C3 is the control demanded by arXiv:2502.08788 / 2311.17371: LLM-reasoning gains routinely
evaporate once you match the amount of test-time compute without the proposed mechanism.
C2 isolates whether *chain-of-thought* (as opposed to merely having the conflict numbers)
is what matters -- this is the literal wording of the hypothesis.

CRITICAL -- no ground-truth leakage: every number in every prompt is computed from model
predictions and dataset metadata only.  The DMS score is never shown, and per-assay model
accuracy is never shown.

Outputs: results/e3_assay_profiles.csv, results/e3_prompts.json
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS, FAM_NAMES, FAMILIES, SEED

FAM_DESC = {
    "MSA_Potts": "alignment-based generative/Potts models fitted to the protein family MSA (EVmutation, DeepSequence, EVE)",
    "Evo_Conservation": "evolutionary-tree / conservation-driven predictors (GEMME, ESCOTT, VESPA)",
    "MSA_Conditioned": "transformers conditioned on a retrieved alignment at inference (MSA-Transformer, PoET, TranceptEVE)",
    "ESM_Masked": "masked protein language models trained on UniRef, no alignment (ESM-1v, ESM2-650M, ESM-C)",
    "AR_PLM": "autoregressive protein language models (ProGen2, RITA, ProGen3)",
    "xTrimoPGLM": "xTrimoPGLM general language models with a hybrid masked+causal objective",
    "Inverse_Folding": "inverse-folding models scoring sequences given the 3D backbone (ESM-IF1, ProteinMPNN, MIF-ST)",
    "Struct_PLM": "structure-augmented protein language models using predicted structure tokens (ProSST, ProtSSN, SaProt)",
    "Struct_MSA_Hybrid": "hybrids fusing structure with alignment/retrieval signal (VenusREM, S3F-MSA, RSALOR)",
    "CNN_RNN": "convolutional / recurrent representation learners from the pre-transformer era (CARP, UniRep, WaveNet)",
}


def build_profiles():
    P = pd.read_parquet(RESULTS / "panel_mutants.parquet")
    A = pd.read_csv(RESULTS / "panel_assays.csv").set_index("DMS_id")
    fam_cols = [f"fam_{f}" for f in FAM_NAMES]
    rows = []
    for dms, d in P.groupby("DMS_id", sort=False):
        F = d[fam_cols].to_numpy(float)
        cons = F.mean(1)
        # how much of an outlier is each family relative to the consensus of the others?
        dev = np.abs(F - cons[:, None]).mean(0)
        # mean Spearman of each family with each other family (agreement centrality)
        sub = F if len(F) <= 20000 else F[np.random.default_rng(SEED).choice(len(F), 20000, False)]
        C = np.array(pd.DataFrame(sub, columns=FAM_NAMES).corr(method="spearman").to_numpy(),
                     dtype=float, copy=True)
        np.fill_diagonal(C, np.nan)
        cen = np.nanmean(C, axis=1)
        iu = np.triu_indices(len(FAM_NAMES), 1)
        pairs = sorted(zip(C[iu], [(FAM_NAMES[i], FAM_NAMES[j]) for i, j in zip(*iu)]))
        rec = {"DMS_id": dms, "n": len(d), "EU": d.EU.mean(), "AU": d.AU.mean(),
               "frac_split": float((d.EU > 0.15).mean()),
               "min_pair_rho": float(pairs[0][0]),
               "min_pair": f"{pairs[0][1][0]} vs {pairs[0][1][1]}",
               "min_pair2": f"{pairs[1][1][0]} vs {pairs[1][1][1]}",
               "min_pair2_rho": float(pairs[1][0]),
               "max_pair": f"{pairs[-1][1][0]} vs {pairs[-1][1][1]}",
               "max_pair_rho": float(pairs[-1][0])}
        for k, f in enumerate(FAM_NAMES):
            rec[f"dev_{f}"] = float(dev[k])
            rec[f"cen_{f}"] = float(cen[k])
        rows.append(rec)
    Pr = pd.DataFrame(rows).set_index("DMS_id")
    Pr["EU_pct"] = (Pr.EU.rank(pct=True) * 100).round(0)
    Pr["AU_pct"] = (Pr.AU.rank(pct=True) * 100).round(0)
    Pr = Pr.join(A[["taxon", "coarse_selection_type", "selection_assay", "seq_len",
                    "MSA_Neff_L", "MSA_num_seqs", "MSA_perc_cov", "MSA_Neff_L_category",
                    "region_mutated", "UniProt_ID", "source_organism",
                    "DMS_number_multiple_mutants", "rho_ens", "rho_best_single"] +
                   [f"rho_{f}" for f in FAM_NAMES]])
    return Pr


SYSTEM = ("You are an expert computational protein engineer. You allocate trust across "
          "different families of protein fitness predictors. You answer concisely and you "
          "always end with the requested WEIGHTS line.")

BIO_BLOCK = """PROTEIN AND ASSAY CONTEXT
- UniProt entry: {uni}   (organism: {org}; taxonomic group: {taxon})
- Sequence length: {seqlen} residues; region mutated: {region}
- Deep mutational scan measures: {assay}  (category: {cat})
- Variants scored: {n} ({nmulti} of them multi-mutants)
- Alignment depth available for this protein family: Neff/L = {neff} ({neffcat} depth;
  {nseqs} homologous sequences, {cov}% coverage)"""

CONFLICT_BLOCK = """OBSERVED DISAGREEMENT BETWEEN THE MODEL FAMILIES ON THIS ASSAY
(computed from the families' own predictions; no experimental data was used)
- Cross-family disagreement for this assay is at the {eupct:.0f}th percentile of all 216
  ProteinGym assays (higher = the families disagree more than usual).
- Within-family noise is at the {aupct:.0f}th percentile.
- The two most strongly conflicting family pairs: {p1} (rank correlation {r1:.2f});
  {p2} (rank correlation {r2:.2f}). Most concordant pair: {p3} ({r3:.2f}).
- Per family: mean deviation from the cross-family consensus, and mean rank correlation
  with the other nine families (a low correlation means this family is an outlier here):
{famlines}"""

FAM_LIST = "MODEL FAMILIES\n" + "\n".join(
    f"{i+1}. {f} - {FAM_DESC[f]}" for i, f in enumerate(FAM_NAMES))

TASK_COT = """TASK
Think step by step about this specific protein and assay. In at most 8 short sentences:
(1) identify which biological or dataset properties here should make some model families
more reliable and others less reliable (e.g. alignment depth, taxon, whether the assay
measures stability vs binding vs organismal fitness, sequence length, structural coverage);
{extra}(3) decide how to allocate trust.
Then output a final line, exactly in this format and nothing after it:
WEIGHTS: {schema}
Each weight is an integer from 0 to 10. Use the full range: give 0-2 to families you expect
to be unreliable here and 8-10 to families you expect to be most reliable."""

TASK_DIRECT = """TASK
Do not explain your reasoning. Output only a single line, exactly in this format:
WEIGHTS: {schema}
Each weight is an integer from 0 to 10. Use the full range: give 0-2 to families you expect
to be unreliable for this assay and 8-10 to families you expect to be most reliable."""

SCHEMA = " ".join(f"{f}=<int>" for f in FAM_NAMES)


def make_prompts(Pr):
    out = {}
    for dms, r in Pr.iterrows():
        bio = BIO_BLOCK.format(
            uni=r.UniProt_ID, org=r.source_organism, taxon=r.taxon, seqlen=int(r.seq_len),
            region=r.region_mutated, assay=r.selection_assay, cat=r.coarse_selection_type,
            n=int(r.n), nmulti=int(r.DMS_number_multiple_mutants),
            neff=f"{r.MSA_Neff_L:.2f}", neffcat=str(r.MSA_Neff_L_category).lower(),
            nseqs=int(r.MSA_num_seqs), cov=int(r.MSA_perc_cov))
        famlines = "\n".join(
            f"    {f:18s} deviation from consensus {r['dev_'+f]:.3f}, "
            f"agreement with others {r['cen_'+f]:.2f}" for f in FAM_NAMES)
        conflict = CONFLICT_BLOCK.format(
            eupct=r.EU_pct, aupct=r.AU_pct, p1=r.min_pair, r1=r.min_pair_rho,
            p2=r.min_pair2, r2=r.min_pair2_rho, p3=r.max_pair, r3=r.max_pair_rho,
            famlines=famlines)
        extra_c = ("(2) explain the observed disagreement pattern above -- why are the "
                   "outlier families disagreeing with the consensus on this particular "
                   "protein, and which side of the conflict is more likely to be right; ")
        extra_n = ""
        for cond in ["conflict_cot", "conflict_direct", "cot_only", "direct_only"]:
            show_conf = cond.startswith("conflict")
            use_cot = cond.endswith("cot") or cond == "cot_only"
            body = FAM_LIST + "\n\n" + bio + ("\n\n" + conflict if show_conf else "")
            task = (TASK_COT.format(extra=extra_c if show_conf else extra_n, schema=SCHEMA)
                    if use_cot else TASK_DIRECT.format(schema=SCHEMA))
            out.setdefault(cond, {})[dms] = (
                "You must decide how much to trust each of 10 independently-developed "
                "families of protein fitness predictors when combining them into a single "
                "ranking of variants for one deep mutational scanning experiment.\n\n"
                + body + "\n\n" + task)
    return out


if __name__ == "__main__":
    Pr = build_profiles()
    Pr.to_csv(RESULTS / "e3_assay_profiles.csv")
    pr = make_prompts(Pr)
    json.dump(pr, open(RESULTS / "e3_prompts.json", "w"), indent=1)
    ex = pr["conflict_cot"][list(pr["conflict_cot"])[0]]
    print(f"[e3a] {len(Pr)} assay profiles; conditions: {list(pr)}")
    lens = {c: int(np.mean([len(v) for v in d.values()])) for c, d in pr.items()}
    print("[e3a] mean prompt chars:", lens)
    print("\n===== EXAMPLE PROMPT (conflict_cot) =====\n")
    print(ex)
