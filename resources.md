# Resources Catalog

**Project:** Conflict-Driven Chain-of-Thought Prompting for Protein Engineering Model Alignment
**Phase:** `resource_finder` (Phase 1) · completed 2026-08-12

---

## Summary

| Resource | Count / Size | Location |
|---|---|---|
| Papers (PDF) | **65** (234 MB) | `papers/` |
| Datasets | **1 primary** (5.1 GB) | `datasets/proteingym/` |
| Code repositories | **1 cloned** | `code/ProteinGym/` |
| Local LLMs | **2 usable** (~45 GB) | `hf_cache/` |
| Derived artifacts | 2 | `artifacts/` |

The single most important fact about this resource set: **ProteinGym publishes per-mutant scores from
95 independently-developed models across 217 assays with ground truth.** The core hypothesis —
that model conflicts are structured — is therefore testable over ~2.5M mutants *without running any
protein model*. Feasibility was verified during this phase and the hypothesis's first clause is
already partially confirmed (§ Feasibility Results).

---

## Papers

65 papers downloaded, all with metadata in `artifacts/paper_meta.json`, grouped by role in
`papers/README.md`. Three were **deep-read in full** via the PDF chunker; the rest were screened by
abstract through the arXiv API.

### Deep-read (full text)

| Title | Year | File | Why |
|---|---|---|---|
| Rank-and-Reason (VenusRAR): Multi-Agent Collaboration Accelerates Zero-Shot Protein Mutation Prediction | 2026 | `papers/2602.00197_*.pdf` | **Nearest prior work.** Defines the gap: uses disagreement implicitly, never characterizes it. Gives the baseline ladder and the DMS99 subset. |
| Complementing Self-Consistency with Cross-Model Disagreement for UQ | 2026 | `papers/2604.17112_*.pdf` | **Supplies the formalism.** AU/EU/TU decomposition and the "confident failure" prediction; ensemble-design criteria. |
| Large Language Model is Secretly a Protein Sequence Optimizer | 2025 | `papers/2501.09274_*.pdf` | The design-loop harness and its correct matched-budget EA baseline. |

### By group (see `papers/README.md` for the full annotated list)

| Group | Count | Representative |
|---|---|---|
| A. Conflict / disagreement / uncertainty | 7 | Cross-model disagreement UQ (2604.17112); Knowledge Conflicts survey (2403.08319) |
| B. Multi-agent debate & CoT | 8 | CoT prompting (2201.11903); *Stop Overvaluing Multi-Agent Debate* (2502.08788) |
| C. LLM + protein engineering | 8 | VenusRAR (2602.00197); AgentPLM (2606.02386); Swarms (2511.22311) |
| D. Fitness prediction & ProteinGym baselines | 12 | Tranception/ProteinGym (2205.13760); PoET (2306.06156) |
| E. Dataset bias / benchmark failure modes | 8 | Survivorship bias (2605.06879); GENEB (2606.04525); DNA-LM shuffling (2510.12617) |
| F. Genomic foundation models | 6 | DNABERT-2 (2306.15006); HyenaDNA (2306.15794) |
| G. Failure-mode / slice discovery | 7 | Domino (2203.14960); HiBug2 (2501.16751); LADDER (2408.07832) |
| H. Protein engineering background | 9 | ML for Protein Engineering (2305.16634); AdaLead (2010.02141) |

---

## Datasets

| Name | Source | Size | Task | Location |
|---|---|---|---|---|
| **ProteinGym v1.3 — DMS substitutions** | marks.hms.harvard.edu (Zenodo mirror) | 91 MB | Ground-truth fitness for 217 assays | `datasets/proteingym/DMS_ProteinGym_substitutions/` |
| **ProteinGym v1.3 — zero-shot model scores** | same | 5.0 GB | **95 model score columns × 217 assays** | `datasets/proteingym/zero_shot_scores/` |
| ProteinGym reference metadata | ProteinGym repo | 1 MB | 46 covariate columns per assay | `code/ProteinGym/reference_files/DMS_substitutions.csv` |
| ProteinGym precomputed leaderboards | ProteinGym repo | 2.4 MB | 99 models × 217 assays × 5 metrics | `code/ProteinGym/benchmarks/` |

Full schema, download commands, loading code, sample records, model-family breakdown, and known
issues: **`datasets/README.md`**. Data files are git-ignored via `datasets/.gitignore`.

---

## Code Repositories

| Name | URL | Purpose | Location |
|---|---|---|---|
| ProteinGym | github.com/OATML-Markslab/ProteinGym | Benchmark metadata, leaderboards, **official bias-corrected aggregation** | `code/ProteinGym/` |

Details, key files, and validation: **`code/README.md`**.

### Local models

| Model | Status |
|---|---|
| `Qwen/Qwen3-14B` (bf16) | ✅ **verified** — 29.5 GB VRAM, 13.6 tok/s. Primary CoT reasoner. On a synthetic conflict it correctly attributed the disagreement to shallow MSA depth and down-weighted the co-evolution model — matching the empirical ρ = −0.371 finding. |
| `Qwen/Qwen3-8B` (bf16) | ✅ **verified** — 16.4 GB VRAM, ~11 tok/s. Heterogeneity partner / ablation. |
| `Qwen/Qwen3-32B-AWQ` | ❌ unusable here (Marlin CUDA kernels fail to JIT-build) — **do not retry** |

Both usable models fit a single A6000, so they can be hosted concurrently on GPUs 0 and 1.

---

## Derived Artifacts

| Path | Contents |
|---|---|
| `artifacts/feasibility_AU_EU.csv` | Per-assay feasibility results (207 assays): ensemble ρ, AU, EU, error-correlations, covariates |
| `artifacts/paper_meta.json` | Metadata + abstracts for all 65 papers |

---

## Resource Gathering Notes

### Search strategy
The **paper-finder service was unavailable** (`localhost:8000` not running), so literature review was
done manually. Three rounds of arXiv API searches (21 structured queries, ~400 candidate abstracts
screened) across: conflict/disagreement/UQ, multi-agent debate and CoT, LLM+protein engineering,
ProteinGym and fitness prediction, dataset bias and benchmark artifacts, genomic FMs, slice
discovery, and directed evolution. Semantic Scholar was rate-limited (HTTP 429) and not used;
OpenAlex was not needed. Dataset and code discovery went through the HuggingFace and GitHub APIs.

### Selection criteria
Papers were kept if they (a) bore directly on one of the four decomposed hypothesis claims, (b) set a
baseline or metric we must match, or (c) documented a failure mode that constrains the design — this
last category deliberately includes results that *undercut* the hypothesis (e.g. *Stop Overvaluing
Multi-Agent Debate*, *Can Reasoning Help LLMs Capture Human Annotator Disagreement?*), because the
design has to survive them.

### Challenges encountered

| Issue | Resolution |
|---|---|
| Paper-finder service down | Manual arXiv API search across 21 queries |
| Semantic Scholar HTTP 429 | Skipped; arXiv + HF + GitHub APIs sufficed |
| `marks.hms.harvard.edu` 403 on site root | File paths work with a normal `User-Agent` header |
| `unzip` not installed | Python `zipfile` |
| VenusRAR code 404 | Not needed — the component models' scores are already in the ProteinGym matrix |
| No LLM API keys | Downloaded and verified local Qwen3 models |
| Qwen3-32B-AWQ won't load (Marlin JIT build failure) | Switched to Qwen3-14B bf16; failure chain documented so it is not repeated |
| transformers 5.x chat-template API change | Documented working call pattern in `code/README.md` |
| Disk at 100% (146 GB free) | Skipped ProteinGym MSA archives (5.2 GB), clinical MSAs (17.8 GB), supervised scores (3.3 GB) |

### Gaps and workarounds
- **No genomic (DNA) cross-model score matrix exists.** The hypothesis names "genomic foundation
  models"; we instantiate it on **protein** foundation models, where ProteinGym makes a clean test
  possible. This is an explicit, flagged reinterpretation — rationale in `planning.md` §4 (D5).
- **ProteinGym-DMS99 is not distributed**; must be reconstructed from `DMS_number_single_mutants` and
  sequence length in the reference file.
- **No wet-lab validation is possible.** All claims are computational.

---

## Recommendations for Experiment Design

### 1. Primary dataset
**ProteinGym v1.3 substitutions zero-shot score matrix.** 217 assays × ~2.5M mutants × 95 models,
with ground truth and 46 covariate columns. Reconstruct **DMS99** (>99% single-mutant coverage,
~31 assays) for budget-constrained selection metrics.

### 2. Baseline methods
1. Best single model (VenusREM-class, ρ ≈ 0.518)
2. **Uniform family-balanced ensemble — the real bar** (ρ ≈ 0.542)
3. Unconditional LLM-weighted ensemble (isolates the value of *conflict-conditioning*)
4. **Compute-matched CoT with no conflict surfacing** — mandatory, per arXiv:2502.08788 / 2311.17371
5. Random routing / random selection / no-abstention controls

### 3. Evaluation metrics
- Ranking: per-assay Spearman under **ProteinGym's bias-corrected aggregation**
  (`proteingym/performance_DMS_benchmarks.py`) — a plain mean across assays is not comparable
- Selection: Top-K hit rate, Top-X% precision, normalized max score at N ∈ {10,20,30,40} on DMS99
- Failure detection: risk–coverage curves, selective Spearman, AURC, AUROC of conflict vs error
- ⚠️ **Use stratified/selective metrics, not per-mutant error regression** — see §Feasibility

### 4. Code to adapt/reuse
- `code/ProteinGym/proteingym/performance_DMS_benchmarks.py` — aggregation (do not reimplement)
- `code/ProteinGym/benchmarks/` — free per-model, per-assay performance for all 99 models
- AU/EU formalism from arXiv:2604.17112 §3.2 — transplant Eq. 3 to model families
- Slice-discovery template from Domino (2203.14960) / HiBug2 (2501.16751)
- Directed-evolution loop from arXiv:2501.09274 — only if D4 is later revived

### 5. Environment constraints
- GPUs **0 and 1 only** (2 and 3 occupied by another tenant); 48 GB each
- Local LLMs only — no API keys
- Disk ~146 GB free on a 100%-full filesystem — avoid large new downloads
- Activate with `source .venv/bin/activate`; add packages with `uv add`

---

## Feasibility Results Obtained This Phase

Run over the downloaded matrix: 7 model families × 3 members = 21 models, **207 assays** with
complete coverage; scores converted to within-assay normalized ranks;
AU = mean intra-family SD, EU = SD across family means.

| Measurement | Value | Interpretation |
|---|---|---|
| Family-balanced ensemble Spearman | **0.512** | Panel is competitive (published SOTA singles ≈ 0.518) |
| Ensemble Spearman, **low-conflict** quintile | **0.681** | |
| Ensemble Spearman, **high-conflict** quintile | **0.354** | **Near-doubling — large exploitable gap** |
| Assay-level EU vs log(MSA Neff/L) | **ρ = −0.371** | Conflict concentrates where evolutionary data is sparse |
| Assay-level EU vs ensemble Spearman | **ρ = −0.437** | Conflict tracks where the ensemble is wrong |
| Mean EU: Virus **0.151** vs Human **0.111** | **+35%** | Conflict concentrates on viral proteins |
| Per-mutant AU vs \|rank error\| | 0.048 | Weak |
| Per-mutant EU vs \|rank error\| | 0.055 | Weak |

**Conclusion.** The hypothesis's first clause — that conflicts cluster around biological ambiguities
and dataset biases — is **already supported at the assay level**, before any LLM is involved. The
signal is real and large, but it lives at the level of **strata and assays rather than individual
mutants**; the two weak per-mutant correlations are a constraint on metric design, not a refutation.
This fixes the evaluation strategy for the downstream directions and is the strongest single piece of
evidence carried into Phase 2.
