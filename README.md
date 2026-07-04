# Calibrated Concept Placement

Hierarchy-aware, class-conditional **conformal prediction sets** over subsumption (insertion) edges
for ontology enrichment from clinical/biomedical text.

We wrap the Oxford LM-ontology-concept-placement model so that for each out-of-KB concept it
returns a coverage-guaranteed *set* of candidate insertion edges plus an abstain/refer signal,
evaluated against real SNOMED version-diff ontology growth. No released checkpoint or per-edge
scores exist for that backbone, so we retrained its Edge-Bi-encoder and Edge-Cross-encoder
ourselves (from the SapBERT init, faithful to their pipeline) to get scores to calibrate on.

See `PROJECT_BRIEF.md` for the full plan.

## Layout
- `phase1/`   — Phase 0/1 outputs: reproduced backbone score dumps, exact-edge conformal layer
- `phase2/`   — Phase 2: Wu–Palmer region-aware conformal + Mondrian calibration, figures
- `scripts/`  — reproduction probes and cluster launch scripts
- `results/`  — misc probe metrics
- `CCP_findings.pptx` / `make_findings_ppt.py` — findings deck and its generator

Not included here (see `PROJECT_BRIEF.md` for how to fetch them): `repo/` (cloned upstream
backbone, KRR-Oxford/LM-ontology-concept-placement, MIT) and `data/` (Zenodo record 10432003).

## Datasets

Both are SNOMED CT 2014.09 → 2017.03 version-diff benchmarks from the Oxford OET/LM-placement
paper (arXiv:2402.17897), not something we constructed ourselves:

- **MM-S14-Disease** — narrower subtree (Clinical finding → Disease), 276 out-of-KB test
  mentions / 965 mention-edge pairs. This is the **primary** dataset: reproduction, the exact-edge
  conformal layer, the Wu–Palmer region-aware layer, and Mondrian (class-conditional) calibration
  were all built and validated here first.
  Code: `phase1/conformal.py`, `phase1/lenient_ceiling.py`, `phase2/precompute_wp.py`,
  `phase2/granularity_adaptive.py`, `phase2/mondrian.py`, `phase2/conformal_wp.py`.
  Dumps: `phase1/dumps/`. Launch scripts: `scripts/sbatch_disease_*.sh`.

- **MM-S14-CPP** — broader/harder subset spanning three SNOMED top-level hierarchies (Clinical
  finding, Procedure, Pharmaceutical/biologic product), 432 test mentions. Used as a **second,
  held-out benchmark to check the Disease-derived findings generalize** across a more diverse
  concept mix, not to re-derive the method from scratch.
  Code: `phase2/precompute_wp_cpp.py`, `phase2/granularity_adaptive_cpp.py`.
  Dumps: `phase1/dumps_cpp/`. Launch script: `scripts/sbatch_cpp_full.sh`.

**Asymmetry to note:** the Mondrian/class-conditional calibration (`phase2/mondrian.py`) and the
plain exact-edge conformal layer (`phase2/conformal_wp.py`) only exist for Disease so far — CPP
currently only has the reproduction + Wu–Palmer region-aware analysis
(`granularity_adaptive_cpp.py`), not yet a Mondrian counterpart.

## Honest contribution
Combination (first calibrated placement) + a small-but-real hierarchy-aware nonconformity score +
class-conditional/Mondrian per-subgroup coverage. The selling point is **reliability and curator
usability**, not beating the baseline on raw accuracy.
