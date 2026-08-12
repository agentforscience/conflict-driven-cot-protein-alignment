# Research State

- Current phase: `experiment_runner`
- Pipeline completed: `False`

## Previous phases

resource_finder (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `in_progress`
- Started: `2026-08-12T18:46:34.981007Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Expected: `/workspaces/conflict_driven_chain_of_thoug_20260812_181909_f1a7cb03`
- Actual: `/app`
- Directory usable: `True`
- Current process matches workspace: `False`

## Output validation

No phase output validation recorded yet.

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
**Phase 1 (`resource_finder`) — COMPLETE, 2026-08-12.**

### Completed
- Isolated `uv` venv (`.venv/`, `pyproject.toml` with `[tool.uv] package = false`).
- **65 papers** → `papers/` (grouped in `papers/README.md`, metadata `artifacts/paper_meta.json`).
  Paper-finder service was **down**; used 3 rounds of arXiv API search (21 queries).
  3 papers deep-read in full.
- **ProteinGym v1.3 substitutions** downloaded (5.1 GB) → `datasets/proteingym/`.
- **ProteinGym repo** cloned → `code/ProteinGym/`.
- **Qwen3-14B + Qwen3-8B** (bf16) downloaded and **verified running on GPU**.
- Deliverables: `literature_review.md`, `resources.md`, `planning.md`, `papers/README.md`,
  `datasets/README.md`, `code/README.md`, `datasets/.gitignore`.

### Key finding — the project is largely de-risked
**ProteinGym publishes per-mutant scores from 95 independently-developed models across 217 assays
with ground truth** (`datasets/proteingym/zero_shot_scores/`, 99 cols = 4 meta + 95 models).
Cross-model conflict is measurable over ~2.5M mutants **with zero protein-model inference**.

### Feasibility already run (`artifacts/feasibility_AU_EU.csv`; 21 models, 7 families, 207 assays)
| Measure | Value |
|---|---|
| Family-balanced ensemble Spearman | 0.512 (published SOTA singles ≈ 0.518) |
| Ensemble ρ on **low-conflict** quintile | **0.681** |
| Ensemble ρ on **high-conflict** quintile | **0.354** |
| Assay EU vs log(MSA Neff/L) | **ρ = −0.371** |
| Assay EU vs ensemble ρ | ρ = −0.437 |
| Mean EU: Virus 0.151 vs Human 0.111 | **+35%** |
| Per-mutant AU / EU vs \|rank error\| | 0.048 / 0.055 (**weak**) |

→ Hypothesis clause 1 (**conflicts cluster around biological ambiguity + dataset bias**) is
**already supported at assay level**, pre-LLM. But the signal lives at the level of **strata and
assays, not individual mutants** — so Phase 2 must use **stratified / risk–coverage metrics, not
per-mutant error regression**.

### Direction budget — 3 kept, 6 pruned (full scoring in `planning.md`)
- **D1 Conflict cartography** (20/20) — AU/EU decomposition over family-balanced panel; regress EU on
  ProteinGym's 46 covariates. *Partially confirmed already.*
- **D2 Conflict as failure-mode detector** (19/20) — selective prediction / risk–coverage; test the
  "low-AU + high-EU = confident failure" prediction; LLM slice discovery.
- **D3 Conflict-driven CoT targeted consensus** (18/20) — route only high-conflict mutants to a CoT
  arbiter; measure Spearman + Top-K on DMS99. *Riskiest arm.*
- Pruned: D4 design loop (compute), **D5 DNA/genomic FMs (no cross-model score matrix exists; GENEB
  shows genomic models aren't comparable)**, D6 interpretability, D7 LLM personas, D8 fine-tuning,
  D9 wet-lab.

### Flagged assumption
The hypothesis says "genomic foundation models"; we instantiate on **protein** foundation models,
because no genomic benchmark publishes a comparable cross-model per-example score matrix. Rationale
in `planning.md` §4 (D5). If the reviewer requires a DNA arm, the ranking must be revisited.

### Environment constraints (carry forward)
- **No `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`** — all CoT must be local.
- **GPUs 2 and 3 are occupied by another tenant.** Use **GPU 0 and/or 1** only (48 GB each).
- `Qwen3-32B-AWQ` **fails** (gptqmodel → Marlin CUDA JIT build error). **Do not retry.**
- transformers 5.x: `apply_chat_template(..., return_dict=True)` then `generate(**enc)` — the old
  positional call raises a bare `AttributeError`. Pattern documented in `code/README.md`.
- Disk **~146 GB free on a 100%-full filesystem**. MSA archives (5.2 GB) / clinical MSAs (17.8 GB)
  deliberately **not** downloaded.
- `unzip` not installed — use Python `zipfile`.
- VenusRAR code (`ai4protein/VenusRAR`) is **404**; not a blocker (its models' scores are in the matrix).

### Next phase: `experiment_runner` — concrete steps
1. Build family-balanced panel; compute AU/EU/TU per `(DMS_id, mutant)`; persist as parquet. *(D1)*
2. Permutation-tested covariate models for EU at assay and mutant level. *(D1)*
3. Reconstruct **ProteinGym-DMS99** (>99% single-mutant coverage, expect ≈31 assays) from
   `reference_files/DMS_substitutions.csv`. *(needed by D3)*
4. Risk–coverage / selective-Spearman for conflict abstention vs random, score-magnitude, AU-only;
   test low-AU/high-EU confident-failure prediction. *(D2)*
5. Slice discovery + LLM descriptions; validate on held-out assays. *(D2)*
6. Conflict-routed CoT arbitration with the **full baseline ladder incl. the compute-matched
   no-conflict CoT control** (mandatory — debate gains routinely vanish without it). *(D3)*
7. Use `code/ProteinGym/proteingym/performance_DMS_benchmarks.py` for any number compared to the
   published leaderboard; a plain mean over 217 assays is **not** comparable.

### Unresolved uncertainty
- D3 is the risky arm: VenusRAR reports LLM backbone barely affects ranking (0.543–0.551) and that
  Qwen3-8B is "significantly stochastic" as an auditor. We have only local 8B/14B. Qwen3-14B did
  arbitrate a synthetic conflict correctly, but a null result on D3 remains plausible — D1+D2 should
  be able to carry the contribution on their own.
- The real baseline to beat is the **uniform ensemble at ρ ≈ 0.542**, not the best single model.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
**Phase 2 (`experiment_runner`) — COMPLETE, 2026-08-12.** All six phases run in one session.

### Deliverables on disk
`REPORT.md`, `README.md`, `CODE_WALKTHROUGH.md`, `planning.md` (+ Motivation & Novelty section),
`src/` (10 scripts), `results/` (per-assay CSVs, `FINAL_RESULTS.md`/`.json`, 2,376 raw LLM
generations in `e3_gen_*.jsonl`), `figures/fig1–fig7`, `logs/`.

### What was run
- **Panel**: 30 models / 10 independently-developed families / 216 assays / 2,465,704 variants
  → `results/panel_mutants.parquet`. Uniform family-balanced ensemble ρ = 0.5330 (best single
  VenusREM 0.5345) — competitive with the published leaderboard, so not a strawman.
- **E1 cartography**, **E2 + E2b failure detection**, **E3 LLM arbitration (2×2 + full ladder)**,
  **E4 rationale correctness**, **E5 budget-constrained design selection**, **E6 robustness**,
  **E7 order-randomised control**. Real local Qwen3-14B + Qwen3-8B (no API key existed); nothing
  simulated.

### Headline findings (verdict per pre-registered claim)
| Claim | Verdict | Key number |
|---|---|---|
| C1 conflict is structured | **supported at assay level**, weak at variant level | assay CV R² = 0.414 (perm p = 0.005); mutation-level held-out R² = 0.054 |
| C3 conflict detects failure | **supported at stratum level, refuted per variant** | confident-failure Δρ = 0.108, d = 0.95, p = 4e-26; per-variant error AUROC = 0.511 |
| C2 CoT is the instrument | **split** — surfacing helps, CoT itself does not | conflict vs compute-matched CoT +0.0062/+0.0094 (p ≤ 0.006); CoT vs direct with same info −0.0003 (p = 0.49) |
| C4 targeted consensus improves design | **not supported** | `conflict_cot` 14B +0.0022, Holm p = 0.34 n.s.; conflict-penalised selection significantly worse |

Other results that matter: a non-LLM ridge meta-weighter gains **+0.0336** (61% of the oracle's
+0.0553) vs +0.0022 for the LLM; conflict features add only +0.0013 over plain covariates
(largely redundant); surfacing conflict raises the arbiter's weight–accuracy correlation from
0.10 → **0.44** while moving the end metric by +0.006 (the central dissociation); conflict is a
good **router** (fixing the high-conflict half of a candidate pool recovers 76% of oracle gain vs
59% random) but a bad **filter**.

### Traps found (do not repeat)
1. **Range restriction in selective-Spearman.** A trivial `|score−0.5|` baseline appears to beat
   every conflict signal under selective-Spearman, and is at/below chance under error-detection
   AUROC. Any UQ claim on this benchmark needs the AUROC-style metric (E2b) as well.
2. **`max_new_tokens` dispatch bug.** `is_cot` was matched with `endswith("cot")`, so the
   suffixed `*_shuf` conditions silently got the 96-token direct-answer budget and produced 0
   parseable weights. Fixed to `"cot" in cond`. Watch for this if new condition names are added.
3. **Positional bias.** Family list order was fixed; weight vs list position ρ = −0.22 to −0.47.
   E7 randomises order per assay. Future designs should randomise from the start.
4. Bash tool timeout is 10 min — long GPU sweeps must use `run_in_background` + a polling
   `until` loop. `e3_llm.py` is resumable (skips `(model, condition, DMS_id, seed)` already on
   disk), which made the killed runs recoverable.

### Unresolved / next steps if the work continues
- Variant-level arbitration was **bounded but not attempted**: E5 shows ~10× the headroom of
  per-assay reweighting (+0.046 max-fitness@20), but E2b shows no available signal for *which
  way* to resolve a conflict. Structural features (solvent accessibility, contact number) are
  the obvious missing input and are not in the current feature set.
- The **outlier-is-wrong bias** (arbiter penalises Inverse_Folding, the most distinctive family,
  even when it is the most accurate — ρ = 0.733 vs 0.435 on a stability assay) is the most
  actionable fix: give the arbiter expected per-family deviation profiles.
- The genomic (DNA) arm remains untested; the protein reinterpretation is flagged in
  `planning.md` §4 (D5) and REPORT.md §2.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
