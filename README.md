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

## Honest contribution
Combination (first calibrated placement) + a small-but-real hierarchy-aware nonconformity score +
class-conditional/Mondrian per-subgroup coverage. The selling point is **reliability and curator
usability**, not beating the baseline on raw accuracy.
