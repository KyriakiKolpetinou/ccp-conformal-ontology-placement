# Project Brief: Calibrated Concept Placement

**Working title:** Calibrated Concept Placement: Hierarchy-Aware Conformal Prediction Sets over Subsumption Edges for Ontology Enrichment from Clinical Text


---

## 1. One-line summary
Wrap an existing ontology concept-placement model in a hierarchy-aware, class-conditional conformal-prediction layer so that, for each new (out-of-KB) concept, it outputs a coverage-guaranteed *set* of candidate insertion edges (plus an abstain/refer signal) instead of a single unreliable guess — making placement trustworthy and usable for human curators, evaluated against real ontology growth via SNOMED version diff.

## 2. The gap (why this is publishable)
- Existing placement methods (BERTSubs; the Oxford LM-placement framework; CLOZE) output point predictions with no calibrated confidence. Curators can't tell which suggestions to trust.
- The field's authoritative paper (Oxford LM-placement, ESWC 2024) explicitly leaves as future work: (a) the fixed-k vs manual-effort tradeoff ("a balance needs to be achieved... warrants future studies"), (b) lenient/ancestor-aware scoring ("we leave an efficient, lenient evaluation for future studies"), and (c) "investigate how to use the methods to assist human terminologists." Our proposal concretely answers all three.
- Extensive prior-art checking (deep research pass + ~10 targeted searches + full read of the Oxford survey + clearing LLMs4OL 2025 and OAEI/OM 2025 proceedings) found NO paper combining conformal/calibrated uncertainty with the concept-PLACEMENT (subsumption-edge) task. The pieces exist separately; the combination does not.
- STATUS: strong "open" signal, NOT yet certain. Two verification steps remain before full commitment (see Section 8).

## 3. Contribution (stated honestly: "combination + small methodological adaptation", mid-tier ceiling)
1. **Core (combination):** first calibrated, coverage-guaranteed method for ontology concept placement. Convert the model's per-edge scores into conformal prediction sets over candidate insertion edges with marginal coverage 1-alpha.
2. **Methodological novelty (small but real):** a HIERARCHY-AWARE nonconformity score for edge sets, adapting the hierarchical-conformal idea (Mortier; Hierarchical Conformal Classification) from flat node-labeling-into-a-fixed-taxonomy to the EDGE-PLACEMENT setting, so sets collapse upward to a shared ancestor when leaf-level uncertainty is high. This is also a principled version of the "lenient evaluation" the Oxford authors wanted.
3. **Class-conditional / Mondrian extension:** per-subgroup coverage guarantees (e.g., per ontology-depth band, leaf vs non-leaf, Disease vs CPP) — directly upgrades the guarantee from marginal to conditional on the subgroups curators care about.
4. **Applied evaluation:** a curator risk-coverage / AURC curve quantifying how much placement can be safely automated ("review X% of cases, auto-accept the rest at Y% accuracy").
5. **(Optional bonus):** run a DL reasoner (ELK/HermiT) over accepted placements and report a failure-mode count (cycles, unsatisfiable classes, redundant subsumptions).

## 4. Benchmark, data, and backbone code
- **Benchmark:** OET concept-placement datasets MM-S14-Disease (276 out-of-KB test mentions / 965 mention-edge pairs) and MM-S14-CPP (432 test mentions). Constructed from SNOMED CT 2014.09 -> 2017.03 version diff; mentions from MedMentions (PubMed abstracts); gold insertion edges from where 2017 curators placed the concept.
- **Processed data (skip licensed pipeline):** Zenodo record 10432003.
- **Backbone to wrap (public code):** https://github.com/KRR-Oxford/LM-ontology-concept-placement (MIT, Python 3.8). Provides Edge-Bi-encoder, edge enrichment, Edge-Cross-encoder, plus inverted-index / fixed-embedding / LLM-prompting baselines. Best baseline ~30% InR_any@10. Original plan: reuse its per-edge scores without retraining. **Correction (see `notes/STATUS.md`, 2026-06-23):** no released checkpoint or scores exist, so we retrained the Edge-Bi-encoder and Edge-Cross-encoder ourselves from their SapBERT init, faithful to the published pipeline.
- **Repo cautions:** old pinned deps (Transformers/PyTorch/NLTK/Flair/tqdm) with Dependabot alerts the authors won't fix; budget time for env setup. Full data creation needs SNOMED CT (NLM license), UMLS/MRCONSO (UMLS license), MedMentions — but the Zenodo processed data lets you skip this unless you do the clinical-notes leg.
- **Optional clinical leg (Phase 3, stretch):** same version-diff trick but find the new concepts mentioned in MIMIC-IV or n2c2 2019 Track 3 (MCN) clinical notes instead of PubMed abstracts. Earns the "from clinical text" claim; requires you to build the eval yourself; needs credentialed data access.

## 5. Metrics to report
- Their metric, InR_any@k / InR_all@k (for comparability to the published baseline).
- Empirical coverage (does the set contain the gold edge at 1-alpha?).
- Average set size; set size vs ontology depth.
- HEADLINE: risk-coverage / AURC curve (accuracy on auto-accepted vs fraction referred to curator).
- Calibration (ECE).
- Subgroup coverage breakdown (leaf vs non-leaf, depth band, Disease vs CPP) — both as honesty and as a finding.

## 6. Critical methodological caveat (do not overclaim)
Standard split-conformal gives MARGINAL coverage = the guarantee holds averaged over all test mentions, NOT per-mention and NOT per-subgroup. 90% marginal coverage can still be ~78% on rare/complex concepts and ~95% on easy ones. Because the selling point is curator trust on SPECIFIC cases, you MUST (a) state plainly the base guarantee is marginal, (b) report subgroup coverage, and (c) pursue class-conditional/Mondrian conformal (Ding et al. NeurIPS 2023) for per-group guarantees — this is exactly where the hierarchy-aware angle adds value.

## 7. Build order and timeline (solo PhD, ~3-4 months core; ~5-6 with clinical leg + reasoner)
- **Phase 0 (2-4 wks) — reproduce baseline:** clone repo, pull Zenodo data, run Edge-Bi/Cross-encoder, match published InR@k. GO/NO-GO #1: if you can't reproduce, stop.
- **Phase 1 (3-5 wks) — split-conformal layer:** turn edge scores into calibrated sets; report coverage, set size, risk-coverage, ECE. GO/NO-GO #2: if sets explode to near the full candidate pool (scores too poorly separated to calibrate usefully), stop before Phase 2.
- **Phase 2 (4-8 wks) — hierarchy-aware nonconformity + class-conditional/Mondrian:** the core novelty; adapt Mortier/HCC to edges; per-depth-band guarantees. (Variance lives here.)
- **Phase 3 (4-8 wks, optional) — clinical-notes leg + reasoner failure-mode check.**
- **Writing (3-4 wks).**

## 8. Two verification steps still REQUIRED before full commitment
1. **Forward-citation intersection (you do this on Google Scholar / Semantic Scholar):** pull "Cited by" for the placement papers (OET = arXiv:2306.14704; LM-placement = arXiv:2402.17897; BERTSubs = World Wide Web 2023) AND for the hierarchical-conformal papers (arXiv:2501.19038; arXiv:2508.13288). Search within citing sets for: conformal, uncertainty, calibration, coverage, placement, subsumption. The intersection being empty confirms the gap.
2. **Read Bono et al., "Efficient Uncertainty Estimation for LLM-based Entity Linking in Tabular Data" (OAEI/OM 2025, CEUR Vol-4144, pp. 52-73)** — the nearest neighbor. Confirm it is entity LINKING (mention -> existing concept), NOT PLACEMENT (new concept -> insertion edge), so you can distinguish your work.

## 9. Reading list
Placement task / baselines:
- Dong, Chen, He, Gao, Horrocks. "A Language Model based Framework for New Concept Placement in Ontologies." ESWC 2024. arXiv:2402.17897.
- Dong, Chen, He, Horrocks. "Ontology Enrichment from Texts (OET)." CIKM 2023. arXiv:2306.14704.
- Chen, He, Geng, Jimenez-Ruiz, Dong, Horrocks. "Contextual semantic embeddings for ontology subsumption prediction (BERTSubs)." World Wide Web 2023.
- Dong, Chen, He, Liu, Horrocks. "Reveal the Unknown: Out-of-KB Mention Discovery (BLINKout)." CIKM 2023. arXiv:2302.07189.

Conformal machinery (borrowed):
- Angelopoulos & Bates. "A Gentle Introduction to Conformal Prediction and Distribution-Free UQ." arXiv:2107.07511.
- Vovk, Gammerman, Shafer. "Algorithmic Learning in a Random World." (foundational text.)

Hierarchy-aware / class-conditional (adapted — your novelty source):
- Mortier et al. "Conformal Prediction in Hierarchical Classification with Constrained Representation Complexity." arXiv:2501.19038.
- "Hierarchical Conformal Classification." arXiv:2508.13288.
- Ding et al. "Class-conditional conformal prediction with many classes." NeurIPS 2023. (Mondrian/per-group guarantees.)

Nearest neighbors (cite + distinguish):
- Bono, Belotti, Palmonari. "Efficient Uncertainty Estimation for LLM-based Entity Linking in Tabular Data." OAEI/OM 2025, CEUR Vol-4144 pp.52-73.
- PASC: "Pipeline-Aware Conformal Prediction with Joint Coverage Guarantees for Multi-Stage NLP and LLM Pipelines." arXiv:2605.18812.
- CLOZE (the weak unconstrained-LLM baseline): arXiv:2511.16548.
- "Ontology enrichment using a large language model... for concept placement" (SOHOv1/SDoH), J. Biomed. Informatics, June 2025 (ScienceDirect S1532046425000942).

NOTE: verify arXiv IDs for the hierarchical-conformal and medical-NER conformal papers when pulling them — some IDs came from search snippets and were not individually fetched.

## 10. Framing discipline (carry into the paper)
- Contribution is RELIABILITY and CURATOR-USABILITY, not beating BERTSubs on raw accuracy. You likely won't beat them on accuracy and you don't need to.
- Describe novelty precisely as "combination + small methodological adaptation." Reviewers respect a modest, accurate novelty claim over an inflated one.
- Small test sets (276/432) limit statistical strength — use proper calibration/test splits, report confidence intervals, consider cross-conformal.
- Scoop risk is moderate and slightly rising (the community is drifting toward UQ-on-entity-tasks). Move fast.
