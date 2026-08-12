# Literature Review

**Topic:** Conflict-Driven Chain-of-Thought Prompting for Protein Engineering Model Alignment

**Hypothesis under test:** Disagreements between genomic/protein foundation models, when surfaced
through chain-of-thought (CoT) reasoning, cluster systematically around specific biological
ambiguities and dataset biases. Identifying and resolving these conflict points enables (a) detection
of model failure modes and (b) improved protein sequence design via targeted consensus.

59 papers downloaded (`papers/`, catalogued in `papers/README.md`, metadata in
`artifacts/paper_meta.json`). Papers marked **[deep read]** were read in full via the PDF chunker;
the rest were screened by abstract via the arXiv API.

---

## 1. Research Area Overview

The hypothesis sits at the intersection of four literatures that have so far developed largely
independently:

1. **Zero-shot protein fitness prediction.** A large family of models (MSA/Potts-based, protein
   language models, structure-conditioned, retrieval-augmented) score variant effects without labels.
   ProteinGym is the field's standard benchmark, and — crucially for this project — it *publishes the
   per-mutant scores of ~95 such models*, making cross-model conflict directly measurable without
   re-running any model.
2. **Ensembling / consensus for protein models.** Naive averaging is the dominant approach; recent
   agentic work (VenusRAR) adds LLM-calibrated weights and a CoT audit stage.
3. **Uncertainty quantification via cross-model disagreement.** In the LLM literature, disagreement
   between independently-trained models has recently been formalized as *epistemic* uncertainty and
   shown to flag "confident failures" that self-consistency misses.
4. **Failure-mode / slice discovery.** A mature methodology exists (Domino, PRIME, HiBug2, LADDER)
   for finding coherent, interpretable subpopulations where a model underperforms — increasingly
   LLM-driven — but it has not been applied to protein/genomic foundation models.

The gap this project targets is the **junction of 2, 3, and 4**: disagreement between protein models
is currently used only *implicitly* (as an ensembling nuisance or a recall-widening trick). Nobody
has asked whether the disagreement itself is *structured* — whether it concentrates on identifiable
biological and dataset-level conditions — and whether that structure can be exploited.

---

## 2. Key Papers

### 2.1 Closest prior work

#### **[deep read]** Rank-and-Reason (VenusRAR): Multi-Agent Collaboration Accelerates Zero-Shot Protein Mutation Prediction
- **Authors / Year:** Tan, Yu, Wu, Zhong, Li, Fan, Zhu, Liang, Dong, Hong — 2026 (ICML submission)
- **Source:** arXiv:2602.00197 · `papers/2602.00197_RankAndReason_multiagent_mutation.pdf`
- **Contribution:** Two-stage agentic framework. *Rank-Stage*: a "Computational Expert" runs a 6-model
  multi-modal ensemble (VenusREM, ProSST-2048, SaProt-AF650M, ProtSSN-ensemble, GEMME, ESM-IF1) and a
  "Virtual Biologist" LLM sets ensemble weights conditioned on context (taxonomy, MSA depth, pLDDT).
  *Reason-Stage*: a three-agent "Expert Panel" (Statistical Auditor, Structural Biologist,
  Experimental Expert) performs CoT auditing of top candidates against biophysical constraints, with
  iterative reject-and-replace.
- **Datasets:** ProteinGym substitutions (217 DMS assays); **ProteinGym-DMS99** — a curated
  31-assay subset with >99% single-mutant coverage, used for selection metrics (needed so that
  Top-K hit rate has no false negatives from missing ground truth).
- **Metrics:** global Spearman; and on DMS99 at budgets N ∈ {10,20,30,40}: normalized max score,
  Top-X% precision (X ∈ {1,5,10}), Top-K hit rate (K ∈ {5,10,30}).
- **Results:**
  - Global Spearman: static arithmetic-mean ensemble **0.542**; LLM-weighted Rank-Stage **0.551**;
    prior SOTA (VenusREM, AIDO-Protein-RAG) 0.518; ESM2-650M 0.414.
  - **Standalone LLM as predictor is near-useless**: DeepSeek-Reasoner alone gets Spearman 0.159,
    and LLM-only mutation selection is at or below random (Table 2). LLMs must be *orchestrators*.
  - **Correlation–precision gap**: the static ensemble is competitive on global Spearman (0.542) but
    "collapses" on top-K selection; the Reason-Stage improves Top-5 hit rate by **367%** over the
    static ensemble and 15–20% over Rank-Stage at N=40.
  - **LLM backbone barely matters for ranking** (0.543–0.551 across all backbones) — "the Rank-Stage
    functions primarily as a statistical aggregation task… it relies on weight calibration rather
    than reasoning." But backbone capacity *does* govern the Reason-Stage audit; Qwen3-8B is
    explicitly reported as exhibiting "significant stochasticity."
  - Wet-lab: 46.7% positive rate (14/30) on Cas12i3, two mutants at 4.23× and 5.05× activity.
- **Code:** claims `github.com/ai4protein/VenusRAR` — **verified 404 as of 2026-08-12** (not yet
  released). Sibling models from the same lab *are* public (ProSST, ProtSSN, VenusREM).
- **Relevance / the gap it leaves:** This is the nearest neighbour to our hypothesis and constrains
  our claims heavily. But note precisely what it does *not* do:
  - Conflict is used **implicitly**, never characterized. The Statistical Auditor is said to provide
    "analytical insights into ensemble ranking inconsistencies," and the candidate pool is built as a
    union of each model's top-200 in order to "recover high-potential variants that defy consensus
    ranking." Disagreement is thus a *recall widener*, not an object of study.
  - No test of whether conflict is **structured** — no clustering of conflicts against MSA depth,
    taxon, selection assay, or mutation class.
  - No use of conflict as an **abstention / routing** signal.
  - The ensemble is a **fixed 6 models**; the disagreement structure across ProteinGym's ~95
    published baselines is untouched.
  - Their own evidence (LLM backbone doesn't matter for ranking) implies that *unconditional* LLM
    weighting is near-saturated — which argues that any further gain must come from *conditioning on
    where models actually conflict*. This is a direct motivation for our framing.

#### **[deep read]** Complementing Self-Consistency with Cross-Model Disagreement for Uncertainty Quantification
- **Authors / Year:** Hamidieh, Thost, Gerych, Yurochkin, Ghassemi — 2026 (ICLR)
- **Source:** arXiv:2604.17112 · `papers/2604.17112_SelfConsistency_CrossModel_Disagreement_UQ.pdf`
- **Contribution:** Formalizes the AU/EU split for black-box models.
  - **AU (aleatoric)** = intra-model response dispersion: `E[1 − s(r₁, r₂)]` over resampled responses
    from the *same* model.
  - **EU (epistemic)** = `self-similarity − cross-model similarity`, i.e. the gap between how similar
    a model is to itself and how similar it is to an ensemble of independently-trained peers
    (Eq. 3); derived by marginalizing a hypothetical ideal model ω* over a surrogate model
    distribution P_Ω.
  - **TU = AU + EU.**
- **Central empirical claim:** *cross-model disagreement is highest on incorrect answers precisely
  when AU is low* — i.e. EU catches **confident failures** that self-consistency structurally cannot.
- **Ensemble design criteria** (their Sec. 3.2, directly transplantable): (i) **support richness** —
  Ω must cover genuinely distinct plausible hypotheses; (ii) **non-collapsing diversity** — members
  must not be noise-perturbations of one model, else EU is spuriously small; (iii) **calibrated
  weighting** — uniform weights only valid when validation risks are comparable. They satisfy these
  by using same-scale, same-architecture-class models *from different vendors* (cross-family).
- **Metrics:** AUROC of uncertainty vs. correctness; selective prediction / abstention curves.
- **Relevance:** This supplies the **formal machinery** for the project. ProteinGym's model list maps
  onto it almost perfectly: model *families* (ESM2 sizes, ProGen2 sizes, ProtSSN variants, RITA
  sizes, Tranception variants) give within-family members → AU, while independently-developed
  lineages (ESM vs GEMME/EVE vs ProSST vs ESM-IF1 vs ProGen2) give cross-family → EU. To our
  knowledge the AU/EU decomposition has never been applied to protein fitness models.

#### **[deep read]** Large Language Model is Secretly a Protein Sequence Optimizer
- **Authors / Year:** Wang, He, Du, Chen, Li, Liu, Xu, Hassoun — 2025
- **Source:** arXiv:2501.09274 · `papers/2501.09274_LLM_secretly_protein_sequence_optimizer.pdf`
- **Contribution:** An LLM (Llama-3.1-8B-Instruct, no fine-tuning) acts as the mutation/crossover
  operator inside a directed-evolution loop: sample a pair from the pool, prompt the LLM to propose a
  new sequence, score with an oracle, select top-k. Extended to budget-constrained and
  multi-objective (Pareto) settings.
- **Datasets / oracles:** GB1 (exact, 149k), TrpB (exact, 159k), Syn-3bfo (SLIP Potts synthetic),
  AAV and GFP (ML oracle trained per Kirjner et al. 2023). Populations 32/48/96 × 4–8 iterations.
- **Results:** Large wins over a matched EA baseline on GFP (0.97 vs 0.50 top-1), AAV (0.76 vs 0.44),
  Syn-3bfo (2.83 vs 1.85), TrpB; **loses to EA on GB1** — an honest negative worth noting.
- **Relevance:** Provides the concrete, cheap **design-loop harness** for the "improve protein
  sequence design" half of our hypothesis, and establishes the correct baseline (matched-budget EA).
  Its weakness — a single fixed oracle — is exactly where a conflict-aware consensus oracle plugs in.

### 2.2 Benchmark and baseline foundations

- **Tranception / ProteinGym** (Notin et al., arXiv:2205.13760): introduces the benchmark and the
  autoregressive+retrieval architecture. Establishes the MSA-depth (Neff/L) and taxon stratification
  that ProteinGym still ships as metadata — i.e. the field already *suspects* these axes matter,
  but uses them for reporting, not for conflict analysis.
- **PoET** (arXiv:2306.06156): family-conditioned "sequences-of-sequences" generative model; a strong
  retrieval-style baseline present in ProteinGym.
- **Inference-only dropout for PLM zero-shot fitness** (arXiv:2506.14793): improves zero-shot
  predictions by injecting dropout at inference — effectively a *within-model* (aleatoric) ensemble.
  Useful as the AU-side control that is *not* cross-family.
- **Multi-Scale Representation Learning for Protein Fitness** (arXiv:2412.01108),
  **Evolutionary Profiles for Protein Fitness** (arXiv:2510.07286),
  **Retrieval-Enhanced Mutation Mastery** (arXiv:2410.21127): recent ProteinGym-topping methods; all
  gain by *adding a modality*, which is indirect evidence that different modalities carry
  complementary (i.e. conflicting) information.
- **Few-shot fitness via in-context learning and test-time training** (arXiv:2512.02315),
  **Fine-tuning PLMs with DMS** (arXiv:2405.06729): the supervised/few-shot regime, relevant if we
  want to test whether conflict-selected mutants are more informative training points.

### 2.3 Dataset bias and benchmark failure modes (the "dataset biases" half of the hypothesis)

- **Better Protein Function Prediction by Modeling Survivorship Bias** (arXiv:2605.06879): explicit
  demonstration that protein databases encode survivorship bias, and that modeling it helps.
- **Same model, better performance: the impact of shuffling on DNA LM benchmarking**
  (arXiv:2510.12617): benchmark-construction artifacts alone move reported genomic-model performance.
- **GENEB: Why Genomic Models Are Hard to Compare** (arXiv:2606.04525): systematic account of why
  cross-model comparison in genomics is confounded — directly supports the premise that apparent
  model disagreement partly reflects evaluation-protocol bias rather than biology.
- **Frozen but Not Always Accessible** (arXiv:2608.05329) and **Reverse-Complement Consistency for
  DNA LMs** (arXiv:2509.18529): representation-level inconsistencies in genomic FMs.
- **Residue-Level Attributions in PLMs Do Not Recover Allergen Epitopes** (arXiv:2606.22181): a clean
  negative result on PLM interpretability — a caution against over-reading model rationales.

### 2.4 Conflict, debate, and CoT reasoning

- **Chain-of-Thought Prompting Elicits Reasoning in LLMs** (arXiv:2201.11903): the origin point.
- **Knowledge Conflicts for LLMs: A Survey** (arXiv:2403.08319) and **Resolving Knowledge Conflicts
  in LLMs** (arXiv:2310.00935): taxonomy of context–memory and inter-context conflict, and methods
  for surfacing/resolving them. Provides vocabulary for a *conflict taxonomy*, which our project
  needs on the biological side.
- **Should we be going MAD?** (arXiv:2311.17371) and **Stop Overvaluing Multi-Agent Debate — Rethink
  Evaluation and Embrace Model Heterogeneity** (arXiv:2502.08788): important correctives. Debate
  gains are frequently overstated and often vanish against properly-matched single-model baselines;
  what actually helps is *heterogeneity* of the debaters. This is a direct warning for our design:
  any conflict-driven CoT mechanism must be compared against a compute-matched non-CoT baseline.
- **Diversity of Thought Elicits Stronger Reasoning** (arXiv:2410.12853), **DynaDebate**
  (arXiv:2601.05746), **Problem Drift in Multi-Agent Debate** (arXiv:2502.19559): debate degrades
  when agents homogenize or drift off-task; both are failure modes we must instrument.
- **Can Reasoning Help LLMs Capture Human Annotator Disagreement?** (arXiv:2506.19467): finds that
  RLVR-style reasoning can actually *hurt* modeling of genuine disagreement — a caution that CoT is
  not automatically the right tool for representing conflict.

### 2.5 Failure-mode / slice discovery methodology

- **Domino** (arXiv:2203.14960): cross-modal embeddings + mixture model to find coherent error slices
  with natural-language descriptions. The methodological template.
- **PRIME** (arXiv:2310.00164 family), **HiBug2** (arXiv:2501.16751), **LADDER** (arXiv:2408.07832),
  **LLM as Dataset Analyst** (arXiv:2405.02363), **Active Slice Discovery in LLMs**
  (arXiv:2511.20713): progressively more LLM-driven slice discovery, prioritizing *interpretable*
  slices over merely low-performing ones.
- **Relevance:** These give us the evaluation standard for "conflicts cluster around specific
  biological ambiguities": a discovered cluster must be (i) coherent, (ii) describable, and
  (iii) predictive of held-out error — not just a low-performing bucket found post hoc.

### 2.6 Genomic foundation models

- **DNABERT-2** (arXiv:2306.15006), **HyenaDNA** (arXiv:2306.15794), **Genomic LMs: Opportunities and
  Challenges** (arXiv:2407.11435), **OmniGenBench** (arXiv:2410.01784, 2505.14402),
  **BioReason** (arXiv:2505.23579, DNA-LLM reasoning).
- **Assessment for this project:** the hypothesis says "genomic foundation models," but the *only*
  benchmark with published per-example scores from ~95 independently-developed models is
  ProteinGym. Genomic (DNA) FM benchmarks lack a comparable published score matrix, and GENEB
  (arXiv:2606.04525) documents that they are hard to compare at all. This is a decisive feasibility
  asymmetry and is why the DNA-side direction was pruned (see `planning.md`).

---

## 3. Common Methodologies

| Method | Used in |
|---|---|
| Zero-shot likelihood / pseudo-likelihood scoring of mutants | ESM series, ProGen2, RITA, Tranception, ProtGPT2 |
| MSA / co-evolution modeling (Potts, VAE) | EVmutation, DeepSequence, EVE, GEMME, MSA-Transformer |
| Structure-conditioned inverse folding | ESM-IF1, ProteinMPNN, MIF-ST, SaProt, ProSST, ProtSSN |
| Retrieval / family-conditioned hybrids | Tranception, TranceptEVE, PoET, VenusREM, AIDO-Protein-RAG |
| Uniform-weight ensembling | ProteinGym baselines, VenusRAR-Ensemble |
| LLM-calibrated weighting | VenusRAR Rank-Stage |
| CoT audit / multi-agent panel | VenusRAR Reason-Stage, ProtAgents, Virtual Lab |
| LLM-as-operator in evolutionary search | Wang et al. 2025 (2501.09274), Swarms (2511.22311) |
| Cross-model disagreement as epistemic uncertainty | Hamidieh et al. 2026 (2604.17112) — *LLM domain only* |
| Slice discovery for failure modes | Domino, PRIME, HiBug2, LADDER — *vision/NLP only* |

The last two rows are empty on the protein side. That is the opening.

---

## 4. Standard Baselines

For zero-shot variant-effect ranking on ProteinGym (avg. Spearman, from VenusRAR Table 1 and the
ProteinGym leaderboard shipped in `code/ProteinGym/benchmarks/`):

| Baseline | Spearman | Role |
|---|---|---|
| Site-Independent | ~0.36 | Trivial floor |
| ESM2-650M | 0.414 | Standard sequence-only PLM |
| GEMME | 0.455 | Strong MSA-based, cheap |
| ProtSSN-ensemble | 0.449 | Structure-aware |
| ProSST-2048 | 0.507 | Strong structure-aware |
| VenusREM / AIDO-Protein-RAG | 0.518 | Published SOTA singles |
| Uniform ensemble of 6 experts | 0.542 | **The baseline to beat** |
| VenusRAR Rank (LLM-weighted) | 0.551 | Current best reported |
| DeepSeek-Reasoner standalone | 0.159 | LLM-alone floor |

For selection: **random selection**, **best-single-model top-N**, and **uniform-ensemble top-N** at
budgets N ∈ {10,20,30,40} on DMS99.

For sequence design (2501.09274): **matched-budget evolutionary algorithm** with random
mutation/crossover, same population size, same iterations, same initial pool.

---

## 5. Evaluation Metrics

- **Ranking:** Spearman ρ per assay, then ProteinGym's bias-corrected aggregation (average within
  UniProt ID, then within function group — the repo's `performance_DMS_benchmarks.py` does this; a
  plain arithmetic mean across assays is *not* comparable to the leaderboard).
- **Classification:** AUC, MCC, NDCG, Top-recall (all four shipped in the ProteinGym benchmarks dir).
- **Selection under budget:** normalized max score, Top-X% precision, Top-K hit rate.
- **Uncertainty / failure detection:** AUROC of the uncertainty score against per-mutant error;
  **risk–coverage curves** and **selective Spearman** (accuracy on the retained fraction after
  abstaining on the most-conflicted mutants); AURC.
- **Slice quality:** held-out predictiveness of a discovered conflict cluster, plus coherence and
  describability (per Domino / HiBug2 conventions).

⚠️ **Metric caution discovered during feasibility testing** (see §7): per-mutant correlation between
disagreement and |rank error| is *weak* (≈0.05), because per-mutant rank error is extremely noisy.
The same signal is very strong when measured in stratified/selective form (ρ = 0.68 on the
low-disagreement quintile vs 0.35 on the high-disagreement quintile). **Use risk–coverage and
stratified metrics, not per-mutant error regression.**

---

## 6. Datasets in the Literature

| Dataset | Used in | Task |
|---|---|---|
| **ProteinGym substitutions** (217 DMS, ~2.5M mutants, 95 published model score columns) | Tranception, VenusRAR, and essentially every fitness paper | Zero-shot variant effect |
| **ProteinGym-DMS99** (31 assays, >99% single-mutant coverage) | VenusRAR | Budget-constrained selection |
| GB1, TrpB | 2501.09274 | Exact combinatorial landscape optimization |
| GFP (Sarkisyan 2016), AAV (Bryant 2021) | 2501.09274, FLIP | ML-oracle sequence design |
| SLIP / Syn-3bfo | 2501.09274 | Synthetic hard landscape |
| FLIP | 2501.18223 | Supervised protein fitness splits |
| Nucleotide/genomic benchmarks (GUE, OmniGenBench, GENEB) | DNABERT-2, HyenaDNA | DNA-side tasks — **no published cross-model score matrix** |

---

## 7. Gaps and Opportunities

1. **Conflict is never characterized in the protein domain.** VenusRAR exploits it implicitly;
   no work asks whether it is structured.
2. **The AU/EU decomposition has not crossed into protein models.** ProteinGym's model families make
   this a nearly free experiment — and it makes a *falsifiable* prediction (cross-family EU should
   flag confident failures that within-family AU misses).
3. **Unconditional LLM ensemble weighting appears saturated.** VenusRAR's own backbone-invariance
   result (0.543–0.551) says the remaining headroom is in *conditioning on conflict*, not in more
   LLM reasoning applied uniformly.
4. **The correlation–precision gap is under-exploited.** Global Spearman hides high-confidence false
   positives; conflict is a natural detector for exactly those.
5. **Slice-discovery methodology has never been applied to protein FMs.**
6. **No abstention/routing mechanism exists for protein fitness prediction**, despite the fact that
   experimental validation budgets make selective prediction economically meaningful.

---

## 8. Recommendations for Our Experiment

### Recommended datasets
- **Primary: ProteinGym v1.3 substitutions** — 217 assays with per-mutant scores from **95 models**
  plus ground truth, already downloaded (`datasets/proteingym/zero_shot_scores/`). This makes the
  entire conflict-mapping study runnable *with zero model inference*, which is the single biggest
  de-risking factor in this project.
- **Selection subset: reconstruct ProteinGym-DMS99** (>99% single-mutant coverage) from
  `reference_files/DMS_substitutions.csv` — needed for unbiased Top-K hit rate.
- **Design loop: GFP / AAV** via the 2501.09274 protocol if the design arm is pursued.

### Recommended baselines
1. Best single model (VenusREM-class, i.e. the strongest available column).
2. Uniform-weight ensemble of the heterogeneous panel — **the real baseline**, since it gets 0.542
   vs. VenusRAR's 0.551.
3. Static (non-conflict-conditioned) LLM-weighted ensemble — isolates the contribution of
   *conflict-conditioning* specifically.
4. Compute-matched CoT baseline (same token budget, no conflict surfacing) — required by the
   multi-agent-debate correctives (2502.08788, 2311.17371).
5. Random / no-abstention for the selective-prediction arm.

### Recommended metrics
Assay-level Spearman with ProteinGym's bias-corrected aggregation; risk–coverage / selective
Spearman; Top-K hit rate and Top-X% precision at N ∈ {10,20,30,40} on DMS99; AUROC of conflict
score vs. error.

### Methodological considerations
- **The LLM cannot be the predictor.** Standalone LLM ≈ 0.159 Spearman, at/below random for
  selection. It must orchestrate, weight, arbitrate, or explain.
- **Model-family structure must be respected.** Averaging 95 ProteinGym columns naively is
  dominated by whichever family has the most members (ESM2 has 6, ProtSSN has 10, ProSST has 6).
  Family-balanced panels are required for the EU estimate to be meaningful — this is criterion (ii),
  non-collapsing diversity, from 2604.17112.
- **Beware the compute-matched-baseline trap.** The debate literature's own correctives say most
  reported multi-agent gains disappear under matched compute.
- **Local LLM capacity is a real risk.** VenusRAR reports Qwen3-8B as "significantly stochastic" in
  the audit role. No Anthropic/OpenAI API key is available in this workspace, so the strongest
  locally-runnable reasoner (Qwen3-32B-AWQ) should be used for any CoT arm, with Qwen3-8B reserved
  as the heterogeneity partner / ablation rather than the primary reasoner.
- **Report the honest negative if the CoT arm fails.** Given points 3 and 5 above, a well-powered
  null result on "CoT resolution improves consensus" would itself be a contribution, provided the
  conflict-structure result (which is already confirmed) carries the paper.

---

## 9. Feasibility Evidence Already Obtained

Run before finalizing this review, on the downloaded ProteinGym score matrix
(script output archived at `artifacts/feasibility_AU_EU.csv`):

**Setup:** 7 model families × 3 members = 21 models, 207 assays with full coverage.
Per-mutant scores converted to within-assay normalized ranks.
AU = mean intra-family SD; EU = SD across family means.

| Finding | Value |
|---|---|
| Ensemble Spearman (family-balanced, 21 models) | **0.512** (competitive with 0.518 published SOTA singles) |
| Ensemble Spearman on **low-disagreement** quintile | **0.681** |
| Ensemble Spearman on **high-disagreement** quintile | **0.354** |
| Assay-level EU vs. log(MSA Neff/L) | **ρ = −0.371** |
| Assay-level EU vs. ensemble Spearman | **ρ = −0.437** |
| Mean EU, Virus taxon | **0.151** |
| Mean EU, Human taxon | **0.111** |
| Per-mutant AU vs. \|rank error\| | 0.048 (weak — see §5 caution) |
| Per-mutant EU vs. \|rank error\| | 0.055 (weak — see §5 caution) |

**Interpretation.** The hypothesis's first clause is already supported: disagreement is *not*
uniform noise. It concentrates where evolutionary data is sparse (low MSA depth), on viral proteins
(~35% higher than human), and it tracks where the ensemble is wrong. The near-doubling of Spearman
between low- and high-conflict strata is a large, exploitable effect and is the empirical core the
project should be built on. The weak per-mutant correlations are a genuine constraint on metric
design, not a refutation — they say the signal lives at the level of *strata and assays*, not
individual mutants.
