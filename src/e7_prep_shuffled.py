"""EXPERIMENT 7a — Order-randomised control prompts.

E6/R1 found a real positional bias: families listed earlier in the prompt receive higher
weights (Spearman weight-vs-position between -0.22 and -0.47 depending on condition).  The
family list order was fixed across all 216 assays, so this bias is perfectly confounded with
family identity, and any per-family conclusion from E3/E4 could in principle be an ordering
artefact.

This script regenerates the `conflict_cot` and `cot_only` prompts with the family list order
independently permuted per assay (seeded).  If the arbiter's weight quality survives
randomisation, the E3/E4 conclusions are not an ordering artefact.

Outputs: results/e7_prompts_shuffled.json, results/e7_order_map.json
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS, FAM_NAMES, SEED
from e3_prep import (FAM_DESC, BIO_BLOCK, CONFLICT_BLOCK, TASK_COT)

rng = np.random.default_rng(SEED + 7)


def main():
    Pr = pd.read_csv(RESULTS / "e3_assay_profiles.csv").set_index("DMS_id")
    prompts, order_map = {"conflict_cot_shuf": {}, "cot_only_shuf": {}}, {}

    for dms, r in Pr.iterrows():
        order = list(rng.permutation(FAM_NAMES))
        order_map[dms] = order
        fam_list = "MODEL FAMILIES\n" + "\n".join(
            f"{i+1}. {f} - {FAM_DESC[f]}" for i, f in enumerate(order))
        schema = " ".join(f"{f}=<int>" for f in order)
        bio = BIO_BLOCK.format(
            uni=r.UniProt_ID, org=r.source_organism, taxon=r.taxon, seqlen=int(r.seq_len),
            region=r.region_mutated, assay=r.selection_assay, cat=r.coarse_selection_type,
            n=int(r.n), nmulti=int(r.DMS_number_multiple_mutants),
            neff=f"{r.MSA_Neff_L:.2f}", neffcat=str(r.MSA_Neff_L_category).lower(),
            nseqs=int(r.MSA_num_seqs), cov=int(r.MSA_perc_cov))
        famlines = "\n".join(
            f"    {f:18s} deviation from consensus {r['dev_'+f]:.3f}, "
            f"agreement with others {r['cen_'+f]:.2f}" for f in order)
        conflict = CONFLICT_BLOCK.format(
            eupct=r.EU_pct, aupct=r.AU_pct, p1=r.min_pair, r1=r.min_pair_rho,
            p2=r.min_pair2, r2=r.min_pair2_rho, p3=r.max_pair, r3=r.max_pair_rho,
            famlines=famlines)
        extra_c = ("(2) explain the observed disagreement pattern above -- why are the "
                   "outlier families disagreeing with the consensus on this particular "
                   "protein, and which side of the conflict is more likely to be right; ")
        head = ("You must decide how much to trust each of 10 independently-developed "
                "families of protein fitness predictors when combining them into a single "
                "ranking of variants for one deep mutational scanning experiment.\n\n")
        prompts["conflict_cot_shuf"][dms] = (
            head + fam_list + "\n\n" + bio + "\n\n" + conflict + "\n\n"
            + TASK_COT.format(extra=extra_c, schema=schema))
        prompts["cot_only_shuf"][dms] = (
            head + fam_list + "\n\n" + bio + "\n\n"
            + TASK_COT.format(extra="", schema=schema))

    # append to the main prompt file so e3_llm.py can serve them unchanged
    main_p = json.load(open(RESULTS / "e3_prompts.json"))
    main_p.update(prompts)
    json.dump(main_p, open(RESULTS / "e3_prompts.json", "w"), indent=1)
    json.dump(order_map, open(RESULTS / "e7_order_map.json", "w"), indent=1)
    print(f"[e7a] added {list(prompts)} for {len(order_map)} assays "
          f"with per-assay randomised family order")
    ex = prompts["conflict_cot_shuf"][list(order_map)[0]]
    print("\nexample family order:", order_map[list(order_map)[0]])
    print(ex[:400])


if __name__ == "__main__":
    main()
