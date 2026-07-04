# Calibrated Concept Placement

Hierarchy-aware, class-conditional **conformal prediction sets** over subsumption (insertion) edges
for ontology enrichment from clinical/biomedical text.

We wrap the Oxford LM-ontology-concept-placement model (we do **not** retrain it) so that for each
out-of-KB concept it returns a coverage-guaranteed *set* of candidate insertion edges plus an
abstain/refer signal, evaluated against real SNOMED version-diff ontology growth.

**Target venues:** JBI, JAMIA Open, or J. Biomedical Semantics (mid-tier, non-MDPI).

See `PROJECT_BRIEF.md` for the full plan and `notes/STATUS.md` for live state and go/no-go gates.

## Layout
- `repo/`     — cloned upstream backbone (KRR-Oxford/LM-ontology-concept-placement, MIT)
- `data/`     — Zenodo processed datasets (record 10432003); git-ignored
- `scripts/`  — our conformal layer, probes, evaluation
- `notes/`    — STATUS, decisions log, prior-art verification
- `results/`  — metrics, figures, calibration tables
- `env/`      — environment specs (conda/pip)

## Honest contribution
Combination (first calibrated placement) + a small-but-real hierarchy-aware nonconformity score +
class-conditional/Mondrian per-subgroup coverage. The selling point is **reliability and curator
usability**, not beating the baseline on raw accuracy.
