"""Shared configuration: paths, model-family panel, seeds.

The family panel is the central design decision of this project. ProteinGym ships 95
zero-shot score columns, but they are *not* 95 independent opinions: ProtSSN contributes
10 columns, ProSST 6, ESM2 6.  Treating the raw matrix as an ensemble would silently
convert cross-lineage disagreement (epistemic) into within-lineage scale noise (aleatoric).
We therefore group columns into independently-developed lineages ("families") and take
exactly 3 members from each, so every family carries equal weight.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "proteingym"
SCORES = DATA / "zero_shot_scores"
DMS = DATA / "DMS_ProteinGym_substitutions"
REF = ROOT / "code" / "ProteinGym" / "reference_files" / "DMS_substitutions.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
for _p in (RESULTS, FIGURES, LOGS):
    _p.mkdir(exist_ok=True)

SEED = 42

# ---------------------------------------------------------------------------
# Family-balanced panel: 10 lineages x 3 members = 30 models.
# Members within a family share a modelling paradigm and usually a codebase/authors;
# members across families were developed independently.
# ---------------------------------------------------------------------------
FAMILIES = {
    # Alignment-based generative / Potts models
    "MSA_Potts": ["EVmutation", "DeepSequence_ensemble", "EVE_ensemble"],
    # Evolutionary-tree / conservation-driven predictors
    "Evo_Conservation": ["GEMME", "ESCOTT", "VESPA"],
    # Transformers conditioned on a retrieved alignment at inference time
    "MSA_Conditioned": ["MSA_Transformer_ensemble", "PoET", "TranceptEVE_L"],
    # Masked (BERT-style) protein language models
    "ESM_Masked": ["ESM1v_ensemble", "ESM2_650M", "ESMC-600M"],
    # Autoregressive protein language models
    "AR_PLM": ["Progen2_large", "RITA_l", "Progen3_1b"],
    # xTrimoPGLM lineage (GLM hybrid objective)
    "xTrimoPGLM": ["xTrimoPGLM-3B-MLM", "xTrimoPGLM-3B-CLM", "xTrimoPGLM-10B-MLM"],
    # Inverse-folding models: structure -> sequence likelihood
    "Inverse_Folding": ["ESM-IF1", "ProteinMPNN", "MIFST"],
    # Structure-augmented protein language models
    "Struct_PLM": ["ProSST-2048", "ProtSSN_ensemble", "SaProt_650M_AF2"],
    # Hybrids that fuse structure with alignment/retrieval signal
    "Struct_MSA_Hybrid": ["VenusREM", "S3F_MSA", "RSALOR"],
    # Convolutional / recurrent representation learners (pre-transformer lineage)
    "CNN_RNN": ["CARP_640M", "Unirep_evotune", "Wavenet"],
}
PANEL = [m for ms in FAMILIES.values() for m in ms]
FAM_OF = {m: f for f, ms in FAMILIES.items() for m in ms}
FAM_NAMES = list(FAMILIES)

META_COLS = ["mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"]

# Single best published model on ProteinGym v1.3 substitutions (leaderboard reference)
BEST_SINGLE = "VenusREM"
