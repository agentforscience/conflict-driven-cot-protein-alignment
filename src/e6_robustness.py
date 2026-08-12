"""EXPERIMENT 6 — Robustness and confound checks on the LLM arbiter.

Four checks that a reviewer would ask for before believing any E3/E4 conclusion:

  R1 POSITIONAL BIAS.  The 10 families are listed in a fixed order in every prompt.  If the
     arbiter's weights track list position, part of the "biological reasoning" is really an
     ordering artefact.
  R2 BACKBONE AGREEMENT.  Do Qwen3-14B and Qwen3-8B assign similar weights to the same assay?
     Low agreement means the arbitration is backbone-specific rather than a property of the
     evidence, which caps how much any single-model result can be trusted.
  R3 SAMPLING STOCHASTICITY.  Across sampled seeds at temperature 0.7, how much does the same
     prompt move?  VenusRAR reports Qwen3-8B is "significantly stochastic" as an auditor.
  R4 PROMPT-LENGTH CONFOUND.  The conflict conditions have longer prompts than the no-conflict
     conditions.  If weight dispersion simply tracks prompt length, the "conflict effect" may
     be an attention/length effect.  Reported descriptively.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS, FAM_NAMES

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
    G = G[G.weights.apply(lambda w: isinstance(w, dict) and len(w) == len(FAM_NAMES))]
    return G


def main():
    G = load()
    prompts = json.load(open(RESULTS / "e3_prompts.json"))
    out = {}
    print(f"[e6] {len(G)} parsed generations, {G.short.nunique()} configurations\n")

    # ---------------------------------------------------------------- R1 positional bias
    print("[R1] Positional bias: weight vs the family's position in the prompt list")
    # For the order-randomised conditions the position of each family varies per assay,
    # so the position must be read from the saved permutation, not from FAM_NAMES.
    omap_p = RESULTS / "e7_order_map.json"
    omap = json.load(open(omap_p)) if omap_p.exists() else {}
    rows = []
    for tag, g in G.groupby("short"):
        shuffled = "_shuf" in tag
        pos, w = [], []
        for _, r in g.iterrows():
            order = omap.get(r.DMS_id, FAM_NAMES) if shuffled else FAM_NAMES
            for i, f in enumerate(order):
                pos.append(i + 1)
                w.append(r.weights[f])
        rho = stats.spearmanr(pos, w)
        # per-position mean weight, to see whether it is a slope or a first/last effect
        mw = pd.Series(w).groupby(pd.Series(pos)).mean()
        rows.append({"run": tag, "spearman_weight_vs_position": rho.statistic,
                     "p": rho.pvalue, "mean_w_pos1": mw.iloc[0], "mean_w_pos10": mw.iloc[-1]})
    R1 = pd.DataFrame(rows)
    R1.to_csv(RESULTS / "e6_positional_bias.csv", index=False)
    print(R1.round(4).to_string(index=False))
    out["R1_max_abs_position_rho"] = float(R1.spearman_weight_vs_position.abs().max())

    # ---------------------------------------------------------------- R2 backbone agreement
    print("\n[R2] Backbone agreement: Qwen3-14B vs Qwen3-8B on the same assay")
    rows = []
    for cond in ["conflict_cot", "cot_only", "direct_only"]:
        a = G[(G.condition == cond) & (G.model.str.contains("14B")) & (G.seed == 0)]
        b = G[(G.condition == cond) & (G.model.str.contains("8B")) & (G.seed == 0)]
        if len(a) < 30 or len(b) < 30:
            continue
        a = a.set_index("DMS_id"); b = b.set_index("DMS_id")
        ix = a.index.intersection(b.index)
        per = []
        for i in ix:
            wa = [a.loc[i, "weights"][f] for f in FAM_NAMES]
            wb = [b.loc[i, "weights"][f] for f in FAM_NAMES]
            if np.std(wa) > 0 and np.std(wb) > 0:
                per.append(stats.spearmanr(wa, wb).statistic)
        rows.append({"condition": cond, "n_assays": len(per),
                     "mean_within_assay_weight_agreement": float(np.mean(per)),
                     "sd": float(np.std(per))})
    R2 = pd.DataFrame(rows)
    R2.to_csv(RESULTS / "e6_backbone_agreement.csv", index=False)
    print(R2.round(4).to_string(index=False) if len(R2) else "  (insufficient data)")
    out["R2"] = R2.to_dict("records")

    # ---------------------------------------------------------------- R3 stochasticity
    print("\n[R3] Sampling stochasticity across seeds (temperature 0.7)")
    rows = []
    for (mdl, cond), g in G[G.seed > 0].groupby(["model", "condition"]):
        seeds = sorted(g.seed.unique())
        if len(seeds) < 2:
            # compare each sampled seed against the greedy run instead
            g0 = G[(G.model == mdl) & (G.condition == cond) & (G.seed == 0)].set_index("DMS_id")
            gs = g.set_index("DMS_id")
            ix = g0.index.intersection(gs.index)
            per = [stats.spearmanr([g0.loc[i, "weights"][f] for f in FAM_NAMES],
                                   [gs.loc[i, "weights"][f] for f in FAM_NAMES]).statistic
                   for i in ix
                   if np.std([g0.loc[i, "weights"][f] for f in FAM_NAMES]) > 0
                   and np.std([gs.loc[i, "weights"][f] for f in FAM_NAMES]) > 0]
            if per:
                rows.append({"model": mdl.split("/")[-1], "condition": cond,
                             "comparison": f"greedy vs seed {seeds[0]}", "n": len(per),
                             "mean_weight_agreement": float(np.mean(per))})
            continue
        A = g[g.seed == seeds[0]].set_index("DMS_id")
        B = g[g.seed == seeds[1]].set_index("DMS_id")
        ix = A.index.intersection(B.index)
        per = [stats.spearmanr([A.loc[i, "weights"][f] for f in FAM_NAMES],
                               [B.loc[i, "weights"][f] for f in FAM_NAMES]).statistic
               for i in ix
               if np.std([A.loc[i, "weights"][f] for f in FAM_NAMES]) > 0
               and np.std([B.loc[i, "weights"][f] for f in FAM_NAMES]) > 0]
        rows.append({"model": mdl.split("/")[-1], "condition": cond,
                     "comparison": f"seed {seeds[0]} vs {seeds[1]}", "n": len(per),
                     "mean_weight_agreement": float(np.mean(per))})
    R3 = pd.DataFrame(rows)
    R3.to_csv(RESULTS / "e6_seed_stochasticity.csv", index=False)
    print(R3.round(4).to_string(index=False) if len(R3) else "  (no sampled seeds available)")
    out["R3"] = R3.to_dict("records")

    # ---------------------------------------------------------------- R4 length confound
    print("\n[R4] Prompt length vs weight dispersion")
    rows = []
    for tag, g in G.groupby("short"):
        cond = tag.split("|")[1]
        plen = np.mean([len(prompts[cond][d]) for d in g.DMS_id if d in prompts[cond]])
        disp = float(np.mean([np.std([r.weights[f] for f in FAM_NAMES])
                              for _, r in g.iterrows()]))
        gen_len = float(g.text.str.len().mean())
        rows.append({"run": tag, "mean_prompt_chars": plen, "mean_output_chars": gen_len,
                     "mean_weight_sd": disp})
    R4 = pd.DataFrame(rows).sort_values("mean_prompt_chars")
    R4.to_csv(RESULTS / "e6_length_confound.csv", index=False)
    print(R4.round(2).to_string(index=False))
    out["R4"] = R4.to_dict("records")

    json.dump(out, open(RESULTS / "e6_summary.json", "w"), indent=2, default=float)
    print("\n[e6] done")


if __name__ == "__main__":
    main()
