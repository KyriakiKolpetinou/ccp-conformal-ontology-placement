# Project Brief: Calibrated Concept Placement

**Working title:** Calibrated Concept Placement: Hierarchy-Aware Conformal Prediction Sets over Subsumption Edges for Ontology Enrichment from Clinical Text


---

## 1. One-line summary
Wrap an existing ontology concept-placement model in a hierarchy-aware, class-conditional conformal-prediction layer so that, for each new (out-of-KB) concept, it outputs a coverage-guaranteed *set* of candidate insertion edges (plus an abstain/refer signal) instead of a single unreliable guess — making placement trustworthy and usable for human curators, evaluated against real ontology growth via SNOMED version diff.

## 2. Benchmark, data, and backbone code
- **Benchmark:** OET concept-placement datasets MM-S14-Disease (276 out-of-KB test mentions / 965 mention-edge pairs) and MM-S14-CPP (432 test mentions). Constructed from SNOMED CT 2014.09 -> 2017.03 version diff; mentions from MedMentions (PubMed abstracts); gold insertion edges from where 2017 curators placed the concept.
- **Processed data (skip licensed pipeline):** Zenodo record 10432003.
- **Backbone to wrap (public code):** https://github.com/KRR-Oxford/LM-ontology-concept-placement (MIT, Python 3.8). Provides Edge-Bi-encoder, edge enrichment, Edge-Cross-encoder, plus inverted-index / fixed-embedding / LLM-prompting baselines. Best baseline ~30% InR_any@10. Original plan: reuse its per-edge scores without retraining. **Correction (see `notes/STATUS.md`, 2026-06-23):** no released checkpoint or scores exist, so we retrained the Edge-Bi-encoder and Edge-Cross-encoder ourselves from their SapBERT init, faithful to the published pipeline.
- **Repo cautions:** old pinned deps (Transformers/PyTorch/NLTK/Flair/tqdm) with Dependabot alerts the authors won't fix; budget time for env setup. Full data creation needs SNOMED CT (NLM license), UMLS/MRCONSO (UMLS license), MedMentions — but the Zenodo processed data lets you skip this unless you do the clinical-notes leg.
- **Optional clinical leg (Phase 3, stretch):** same version-diff trick but find the new concepts mentioned in MIMIC-IV or n2c2 2019 Track 3 (MCN) clinical notes instead of PubMed abstracts. Earns the "from clinical text" claim; requires you to build the eval yourself; needs credentialed data access.

## 3. Metrics to report
- Their metric, InR_any@k / InR_all@k (for comparability to the published baseline).
- Empirical coverage (does the set contain the gold edge at 1-alpha?).
- Average set size; set size vs ontology depth.
- HEADLINE: risk-coverage / AURC curve (accuracy on auto-accepted vs fraction referred to curator).
- Calibration (ECE).
- Subgroup coverage breakdown (leaf vs non-leaf, depth band, Disease vs CPP) — both as honesty and as a finding.
- 
## 4. Reading list
Placement task / baselines:
- Dong, Chen, He, Gao, Horrocks. "A Language Model based Framework for New Concept Placement in Ontologies." ESWC 2024. arXiv:2402.17897.
- Dong, Chen, He, Horrocks. "Ontology Enrichment from Texts (OET)." CIKM 2023. arXiv:2306.14704.
- Chen, He, Geng, Jimenez-Ruiz, Dong, Horrocks. "Contextual semantic embeddings for ontology subsumption prediction (BERTSubs)." World Wide Web 2023.
- Dong, Chen, He, Liu, Horrocks. "Reveal the Unknown: Out-of-KB Mention Discovery (BLINKout)." CIKM 2023. arXiv:2302.07189.

Conformal machinery (borrowed):
- Angelopoulos & Bates. "A Gentle Introduction to Conformal Prediction and Distribution-Free UQ." arXiv:2107.07511.
- Vovk, Gammerman, Shafer. "Algorithmic Learning in a Random World." (foundational text.)

Hierarchy-aware / class-conditional (adapted):
- Mortier et al. "Conformal Prediction in Hierarchical Classification with Constrained Representation Complexity." arXiv:2501.19038.
- "Hierarchical Conformal Classification." arXiv:2508.13288.
- Ding et al. "Class-conditional conformal prediction with many classes." NeurIPS 2023. (Mondrian/per-group guarantees.)

Nearest neighbors:
- Bono, Belotti, Palmonari. "Efficient Uncertainty Estimation for LLM-based Entity Linking in Tabular Data." OAEI/OM 2025, CEUR Vol-4144 pp.52-73.
- PASC: "Pipeline-Aware Conformal Prediction with Joint Coverage Guarantees for Multi-Stage NLP and LLM Pipelines." arXiv:2605.18812.
- CLOZE (the weak unconstrained-LLM baseline): arXiv:2511.16548.
- "Ontology enrichment using a large language model... for concept placement" (SOHOv1/SDoH), J. Biomed. Informatics, June 2025 (ScienceDirect S1532046425000942).

- Contribution is RELIABILITY and CURATOR-USABILITY, not beating BERTSubs on raw accuracy
- Small test sets (276/432) limit statistical strength — use proper calibration/test splits, report confidence intervals, consider cross-conformal.
