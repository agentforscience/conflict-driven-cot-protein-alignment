# Planning — Direction Enumeration, Scoring, and Budget

**Phase:** `resource_finder` (Phase 1)
**Date:** 2026-08-12
**Direction budget:** keep top **3** for implementation; record and justify all prunes.

---

## 1. Decomposing the Hypothesis

> *Disagreements between genomic foundation models, when surfaced through chain-of-thought reasoning,
> will systematically cluster around specific biological ambiguities and dataset biases. By
> identifying and resolving these conflict points, we can detect model failure modes and improve
> protein sequence design performance through targeted consensus mechanisms.*

Four separable, independently-testable claims:

| # | Claim | Testable form |
|---|---|---|
| **C1** | Conflicts cluster around biological ambiguities and dataset biases | Cross-model disagreement is predicted by MSA depth, taxon, selection assay, mutation class — beyond chance |
| **C2** | CoT reasoning is the right instrument for *surfacing* conflict | CoT-surfaced conflict descriptions add information over purely numerical disagreement statistics |
| **C3** | Conflict identification detects failure modes | Disagreement predicts where the ensemble is wrong; supports selective prediction / abstention |
| **C4** | Resolving conflicts improves design via targeted consensus | Conflict-conditioned consensus beats uniform ensembling on ranking and top-K selection |

C1 and C3 are largely **data-analytic** and were confirmed to be computable — and partly already
confirmed — during this phase. C2 and C4 require LLM inference and are where the risk sits.

---

## 2. Enumerated Directions

Nine plausible directions were enumerated and scored on four axes (1–5 each):

- **EVID** — support in the literature that the direction is well-posed and non-redundant
- **REL** — relevance to the stated hypothesis
- **GAIN** — expected information gain (does a negative result also teach us something?)
- **FEAS** — implementation feasibility in this workspace (compute, data, no API keys)

| ID | Direction | EVID | REL | GAIN | FEAS | **Total** | Verdict |
|----|-----------|:----:|:---:|:----:|:----:|:---------:|---------|
| **D1** | **Conflict cartography**: map cross-model disagreement over ProteinGym's 95-model score matrix; decompose into within-family (AU) vs cross-family (EU); test clustering against biological/dataset covariates | 5 | 5 | 5 | 5 | **20** | ✅ **KEEP** |
| **D2** | **Conflict as a failure-mode detector**: disagreement-driven selective prediction, risk–coverage curves, and LLM-assisted slice discovery over conflict clusters | 5 | 5 | 5 | 4 | **19** | ✅ **KEEP** |
| **D3** | **Conflict-driven CoT targeted consensus**: route only high-conflict mutants to a CoT arbiter that is shown the conflict + biological context; conflict-conditioned reweighting; measure ranking + top-K selection | 4 | 5 | 5 | 4 | **18** | ✅ **KEEP** |
| D4 | Conflict-aware LLM-guided sequence design (directed evolution à la 2501.09274 with a conflict-aware oracle) | 3 | 4 | 4 | 2 | 13 | ❌ prune |
| D5 | DNA / genomic FM conflict analysis (DNABERT-2, HyenaDNA, NT, Evo) | 4 | 4 | 3 | 1 | 12 | ❌ prune |
| D6 | Mechanistic interpretability of conflict (SAEs / attribution on conflicting residues) | 3 | 3 | 3 | 2 | 11 | ❌ prune |
| D7 | Multi-agent debate between LLM personas representing each protein model | 2 | 4 | 2 | 3 | 11 | ❌ prune |
| D8 | Fine-tuning / RL on conflict-derived preference pairs | 2 | 3 | 3 | 1 | 9 | ❌ prune |
| D9 | Wet-lab or new-DMS validation | 5 | 4 | 5 | 0 | 14 | ❌ prune (impossible) |

---

## 3. The Three Retained Directions

### D1 — Conflict Cartography (foundation; establishes C1)

**Question.** Is cross-model disagreement structured, and along which axes?

**Method.**
- Build a family-balanced panel from ProteinGym's 95 score columns. Families are independently
  developed lineages: `{ESM2, ProGen2, RITA, MSA/Potts (GEMME·EVE·EVmutation), Tranception,
  Structure (ESM-IF1·ProteinMPNN·MIF-ST), ProtSSN, ProSST, CARP, …}`. Balancing is *required*: the
  raw 95 columns are dominated by families with many members (ProtSSN ×10, ProSST ×6, ESM2 ×6), which
  would silently turn cross-family EU into within-family AU.
- Per assay, convert scores to within-assay normalized ranks.
- **AU** = mean within-family SD. **EU** = SD across family means. **TU = AU + EU**
  (formalism from Hamidieh et al. 2026, arXiv:2604.17112, transplanted to the protein domain — novel).
- Regress assay-level and mutant-level EU on covariates available in
  `reference_files/DMS_substitutions.csv`: `MSA_Neff_L`, `MSA_num_seqs`, `taxon`,
  `coarse_selection_type`, `selection_assay`, `seq_len`, `region_mutated`,
  `DMS_binarization_method`, plus mutation-level features (position, wild-type/mutant amino acid,
  BLOSUM substitution class, relative position in sequence).
- Report variance explained, and permutation tests against a null that shuffles covariates.

**Why it ranks first.** Highest feasibility (no model inference at all — pure analysis of a
downloaded matrix), highest relevance (it *is* claim C1), and the result is informative whichever way
it goes. **Already partially confirmed** in this phase — see §5.

**Status:** de-risked, ready to run.

### D2 — Conflict as a Failure-Mode Detector (establishes C3)

**Question.** Does disagreement tell us *where not to trust* the ensemble, and can the failure regions
be named?

**Method.**
- **Selective prediction.** Rank mutants by conflict; abstain on the top-q fraction; plot
  risk–coverage and *selective Spearman* vs coverage. Compare conflict-based abstention against
  (a) random abstention, (b) ensemble-score-magnitude abstention, (c) AU-only abstention.
- **AU vs EU dissociation.** Test the transplanted core claim of arXiv:2604.17112: EU should flag
  *confident failures* — mutants where within-family agreement is high (low AU) but cross-family
  agreement is low (high EU). This is a sharp, falsifiable prediction.
- **Slice discovery.** Apply the Domino/HiBug2 template to conflict clusters; use the local LLM to
  produce natural-language descriptions of each cluster; validate that descriptions are *predictive
  on held-out assays*, not just post-hoc labels.

**Why it ranks second.** Directly operationalizes "detect model failure modes," has an established
metric vocabulary, and has real practical value given the cost of wet-lab validation.

**Risk noted:** per-mutant correlation between conflict and |rank error| measured at only ≈0.05 in
feasibility testing. Metrics must be stratified/selective, not per-mutant regression. See
`literature_review.md` §5.

### D3 — Conflict-Driven CoT Targeted Consensus (establishes C2 + C4)

**Question.** Does surfacing conflicts through CoT and arbitrating them beat uniform ensembling?

**Method.**
- **Targeted routing.** Only high-conflict mutants (identified by D1/D2) go to the CoT arbiter. This
  is the "targeted" in targeted consensus, and it is what distinguishes the work from VenusRAR, which
  applies LLM weighting unconditionally.
- **The arbiter prompt** presents the *conflict itself*: which families say what, plus the biological
  context the conflict cartography identified as relevant (MSA depth, taxon, assay type, position
  context). The LLM outputs a reweighting or an arbitration, plus a rationale.
- **Evaluation:** ProteinGym Spearman under the repo's bias-corrected aggregation; Top-K hit rate and
  Top-X% precision at N ∈ {10,20,30,40} on a reconstructed **DMS99** subset.
- **Baselines (all mandatory):** best single model; uniform family-balanced ensemble (**the real bar
  — 0.542 in VenusRAR's Table 1**); unconditional LLM weighting; **compute-matched CoT without
  conflict surfacing** (required by arXiv:2502.08788 and arXiv:2311.17371, which show debate/CoT
  gains routinely vanish against matched-compute baselines); random routing of the same number of
  mutants to the arbiter.

**Why it ranks third but is retained.** It is the literal wording of the hypothesis and carries the
"improve protein sequence design performance" clause. It is riskiest: VenusRAR reports that LLM
backbone barely affects ranking (0.543–0.551) and that Qwen3-8B is "significantly stochastic" in the
audit role — and we have no frontier-model API, only local Qwen3-8B/14B. Retained because
(a) conflict-*conditioned* routing is untested and is exactly the residual headroom their
backbone-invariance result implies, and (b) a well-powered null is publishable given D1+D2 carry the
contribution.

---

## 4. Pruned Directions and Reasons

| ID | Reason for pruning |
|----|--------------------|
| **D4** — conflict-aware sequence design loop | Needs a trained ML oracle per landscape (Kirjner et al. 2023 protocol) plus many LLM generations per iteration; at 11 tok/s measured on Qwen3-8B this is the most compute-hungry option, and the oracle itself would become the dominant confound. Defer: it is the natural follow-up **if** D3 shows a positive effect. Note the hypothesis's "protein sequence design" clause is still addressed by D3 via top-K selection under budget, which is the standard proxy in this literature (VenusRAR, 2501.09274). |
| **D5** — DNA/genomic FM conflicts | **Decisive feasibility asymmetry.** No genomic benchmark publishes a per-example cross-model score matrix comparable to ProteinGym's 95 columns, so every model would have to be run from scratch. Worse, GENEB (arXiv:2606.04525) and the shuffling paper (arXiv:2510.12617) document that genomic models are *not currently comparable* across benchmarks — measured "disagreement" would be confounded with evaluation-protocol artifacts. Note: the hypothesis says "genomic foundation models"; we read that as the sequence-foundation-model family generally, and instantiate it on protein FMs where the evidence base permits a clean test. **This reinterpretation is flagged as an explicit assumption.** |
| **D6** — mechanistic interpretability of conflict | Requires white-box access and per-model activation extraction for models we would have to re-run; arXiv:2606.22181 also reports PLM residue attributions failing to recover known biology, lowering expected yield. |
| **D7** — LLM personas debating as protein models | The debate correctives (arXiv:2502.08788, 2311.17371, 2502.19559) indicate gains come from genuine model heterogeneity, which we *already have* in the real ensemble. Simulating it with personas adds cost and a confound rather than signal. |
| **D8** — fine-tuning on conflict-derived preferences | Training budget and no clean held-out protocol; would confound the conflict claim with a training-data claim. |
| **D9** — wet-lab validation | No laboratory. Out of scope by construction. |

**Re-expansion rule:** per the direction budget, the search space stays fixed at D1–D3 unless new
evidence invalidates this ranking. If that happens, the ranking is to be updated in `STATE.md` with
the reason.

---

## 5. Feasibility Evidence Gathered This Phase

Run on the downloaded ProteinGym matrix; 7 families × 3 members = 21 models, 207 assays with full
coverage. Archived: `artifacts/feasibility_AU_EU.csv`.

| Result | Value | Bears on |
|---|---|---|
| Family-balanced ensemble Spearman | **0.512** | Sanity: panel is competitive (published SOTA singles ≈0.518) |
| Ensemble Spearman, **low-conflict** quintile | **0.681** | D2 |
| Ensemble Spearman, **high-conflict** quintile | **0.354** | D2 — large exploitable gap |
| Assay EU vs log(MSA Neff/L) | **ρ = −0.371** | **D1 / C1 — dataset bias axis confirmed** |
| Assay EU vs ensemble Spearman | **ρ = −0.437** | D2 |
| Mean EU: Virus 0.151 vs Human 0.111 | **+35%** | **D1 / C1 — biological axis confirmed** |
| Per-mutant AU vs \|rank error\| | 0.048 | Metric caution |
| Per-mutant EU vs \|rank error\| | 0.055 | Metric caution |

**Read:** C1 is already supported at assay level. The conflict signal is real and large, but lives at
the level of **strata and assays**, not individual mutants — which fixes the metric design for D2/D3.

---

## 6. Environment Constraints Discovered (carry into Phase 2)

- **No `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`.** All CoT must run on local open-weights models.
- **GPUs:** 4× RTX A6000 (48 GB). **GPUs 2 and 3 are fully occupied by another tenant**; only GPUs
  **0 and 1** are available. Plan for ≤2 GPUs.
- **Qwen3-8B bf16 verified working** (16.4 GB, ~11 tok/s unbatched via transformers). Batch or install
  vLLM for throughput.
- **Qwen3-32B-AWQ does NOT work here** — `gptqmodel` requires JIT-building Marlin CUDA kernels, which
  fails in this container. Do not retry this path. **Qwen3-14B bf16** was downloaded instead as the
  stronger reasoner (fits one A6000, no quantization).
- **Disk is the binding constraint:** ~164 GB free on a 100%-full filesystem. Do not download large
  additional assets (the ProteinGym MSA archive alone is 5.2 GB, clinical MSAs 17.8 GB — skip both
  unless required).
- `unzip` is not installed; use Python `zipfile`.
- **VenusRAR code (`github.com/ai4protein/VenusRAR`) returns 404** — not released. Its component
  models (ProSST, ProtSSN, VenusREM) are public, and their *scores* are already in the ProteinGym
  matrix, so re-implementation is unnecessary.

---

## 7. Concrete Next Steps for Phase 2 (`experiment_runner`)

1. Build the family-balanced panel and the AU/EU/TU matrix over all 217 assays; persist as a tidy
   parquet keyed by `(DMS_id, mutant)`. *(D1)*
2. Fit and permutation-test covariate models for assay-level and mutant-level EU. *(D1 → C1)*
3. Reconstruct **ProteinGym-DMS99** (assays with >99% single-mutant coverage) from the reference
   file; verify the count lands near 31. *(needed by D3)*
4. Risk–coverage and selective-Spearman curves for conflict-based abstention vs the three controls;
   test the AU-low/EU-high "confident failure" prediction. *(D2 → C3)*
5. Slice discovery over conflict clusters with LLM-generated descriptions; validate on held-out
   assays. *(D2)*
6. Conflict-routed CoT arbitration with the full baseline ladder, **including the compute-matched
   no-conflict CoT control**. *(D3 → C2, C4)*
7. Use ProteinGym's own `performance_DMS_benchmarks.py` aggregation for any number compared to
   published leaderboard values — a plain mean across assays is not comparable.

---

# Phase 2 (`experiment_runner`) — Motivation & Novelty Assessment

*Added at the start of the experiment_runner phase, 2026-08-12, before any experiment was run.*

## Why This Research Matters

Protein engineering campaigns are budget-limited: a laboratory can synthesise and assay tens
to hundreds of designed variants, not millions. Which variants make that list is decided by
computational fitness predictors, and there are now ~95 of them with no agreed way to choose
between them. When those predictors disagree, the current practice is to average them and hope.
This work asks whether the *disagreement itself* is a usable signal — whether it tells us where
predictions cannot be trusted, and whether an LLM reasoning over the disagreement can arbitrate
it. A reliable answer either saves wet-lab budget (by flagging untrustworthy regions before
synthesis) or closes off a popular but unfounded idea.

## Gap in Existing Work

From `literature_review.md` and the 65 collected papers:
1. **Disagreement is used but never characterised.** VenusRAR (arXiv:2602.00197) has an LLM
   arbitrate between protein predictors, but weights *unconditionally* and never asks where or
   why they conflict. No paper maps cross-model conflict onto ProteinGym's biological covariates.
2. **The AU/EU decomposition has never been transplanted to protein models.** Hamidieh et al.
   (arXiv:2604.17112) separate within-source noise from cross-source disagreement for LLM
   self-consistency and predict that low-AU + high-EU marks *confident failures*. That sharp
   prediction has never been tested on protein foundation models, where "sources" are genuinely
   independent research lineages rather than samples from one model.
3. **The CoT literature's own correctives are rarely applied here.** arXiv:2502.08788 and
   2311.17371 show debate/CoT gains routinely vanish against compute-matched controls. Protein
   + LLM papers seldom run that control.
4. **Rationale correctness is never separated from end-task gain.** Everyone reports scores;
   nobody asks whether the chain-of-thought identified the *right* biological ambiguity.

## Our Novel Contribution

1. The first quantitative **cartography of cross-model conflict** on ProteinGym: 30 models in
   10 independently-developed families, 216 assays, 2.47M variants, with conflict decomposed
   into aleatoric/epistemic components and regressed on biological and dataset covariates.
2. The first test of the **confident-failure prediction** (low AU + high EU) outside LLM
   self-consistency.
3. A **2x2 ablation** that separates "showing the conflict" from "reasoning about it in
   chain-of-thought", including the compute-matched CoT control the literature demands, plus a
   non-LLM learned meta-weighting competitor with access to the same information.
4. A direct measurement of **rationale correctness** — whether the CoT names the biological
   factor that is actually present, and whether the resulting weights track the families' true
   per-assay accuracy — evaluated separately from whether it improves the end metric.

## Experiment Justification

- **E1 Conflict cartography** — needed for hypothesis clause C1. If conflict is unstructured
  noise, nothing downstream can work, and the whole idea is refuted at minimum cost. It also
  supplies the covariates that the E3 prompts and the learned baseline both use.
- **E2 Failure detection** — needed for clause C3 ("detect model failure modes"). Selective
  prediction is the standard operationalisation; the AU/EU dissociation is the falsifiable form.
- **E2b Fair detector comparison** — needed because the selective-Spearman metric in E2 is
  confounded by range restriction: abstaining on mid-range predictions inflates rank correlation
  regardless of whether the abstained points were wrong. Without E2b we could not tell an
  artefact from a detector.
- **E3 Conflict-driven CoT arbitration** — the literal hypothesis (clauses C2 + C4). The 2x2
  design is what makes the result interpretable rather than just a number.
- **E4 Rationale analysis** — needed because "surfaced through chain-of-thought reasoning" is a
  claim about the *reasoning*, not only the score. E3 alone cannot distinguish "helps for the
  right reason" from "helps by accident" or "reasons correctly but does not help".

## Success Criteria (pre-registered)

- C1 supported if dataset/biological covariates predict assay-level conflict with permutation
  p < 0.05 and held-out CV R^2 > 0.15.
- C3 supported if conflict-based abstention beats random abstention (paired Wilcoxon, p < 0.05)
  AND the low-AU/high-EU stratum shows significantly worse ensemble accuracy than low-AU/low-EU.
- C2 supported if the conflict-shown conditions beat the compute-matched CoT control.
- C4 supported if conflict_cot beats the uniform family-balanced ensemble after Holm correction,
  with an effect size that is not negligible.
- A negative result on C2/C4 is reported as such; C1/C3 are able to carry the contribution.
