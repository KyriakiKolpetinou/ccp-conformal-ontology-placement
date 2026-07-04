# STATUS

Live state of the project. Update at the end of every working session.

Last updated: 2026-06-23

## Current phase
**Phase 0 — de-risk via recall@pool probe.**

### Key finding 2026-06-23: NO released checkpoint or scores
The repo ships only BLINK placeholder `.bin` paths; the placement bi-encoder is TRAINED from
SapBERT (`train_bi=true`) in their pipeline. Zenodo has data only — no scores, no checkpoint.
=> The brief's "reuse their scores, don't retrain" is NOT directly possible. Exact reproduction
   would require running their full train + edge-enrichment stack (Py3.8 + deeponto + Java/OWLAPI).

### De-risking shortcut chosen: zero-shot SapBERT recall@N (lower bound)
Built clean env `ccp` (py3.11, torch 2.5.1+cu121, transformers 5.12.1). Script
`scripts/recall_probe.py` embeds the edge catalogue + mentions with the SAME base model their
bi-encoder fine-tunes from, retrieves top-N, measures recall@N (gold edge in pool). This is a
conservative LOWER BOUND on their fine-tuned recall but reveals retrieval- vs ranking-bottleneck
and gates the calibration-vs-accuracy fork. Two scorings: endpoint-max (optimistic) + edge-text.
NOTE: GPUs shared with user's medical-seg training (~38-44GB/49GB used); probe pinned to GPU0,
fp16, embeddings moved to CPU, retrieval on CPU to avoid disturbing it.

### DECISION (user, 2026-06-23): run their actual pipeline (Phase 0 reproduction)
User chose authoritative reproduction over the zero-shot shortcut. We need their per-edge scores
for the conformal layer anyway, so this is required work, not a detour.
Findings while standing it up:
- Reproduction is NOT blocked on licensed data: their preprocessing outputs (.owl, edge/entity
  catalogues, mention splits) are exactly what's in the Zenodo zip. Map files into expected paths.
- FORCED deviation from their pins: torch 1.11 CANNOT run on RTX 6000 Ada (sm_89 needs CUDA 11.8+).
  Using torch 2.4.1+cu121 (last py3.8 wheel; verified runs on the Ada GPUs). Everything else faithful.
- flair (0.6.1) SKIPPED: only used in blink/ner.py (NER); our mentions are given, path never imports it.
- pytorch_transformers 1.2.0 IS needed (WarmupLinearSchedule, WEIGHTS_NAME) and imports fine under torch 2.4.
- onto38 env VERIFIED: all critical imports OK. Recipe: py3.8, pip<24/setuptools<60, torch 2.4.1+cu121,
  transformers 4.29.0, pytorch-transformers 1.2.0, faiss-cpu, pytorch_metric_learning 1.4.0, nltk 3.6.3,
  + small utils. deeponto 0.8.8 still to install for the enrichment step.
- Data wired via symlinks: repo/ontologies/<catalogues>, repo/data/MedMentions-preprocessed+/Disease/
  st21pv_syn_attr-all-complexEdge-edges-final/<splits> (read_dataset reads {mode}.jsonl).
- read_dataset(mode,data_path) -> {data_path}/{mode}.jsonl; split names map to Zenodo filenames.
NEXT: debug-run bi-encoder to surface key/format mismatches; then full train+search needs a GPU
decision (their seg run owns ~80-90% of both GPUs).

### 2026-06-23 (cont.): bi-encoder debug-run PASSED — data mapping correct, no fix needed
Ran train_biencoder.py on Disease, debug_max_lines=64, batch 4, 1 epoch, GPU0 (PYTHONPATH=.,
CUDA_VISIBLE_DEVICES=0, no --data_parallel so seg jobs 307/311 untouched). End-to-end clean:
read BOM'd jsonl (reader uses utf-8-sig), tokenized via process_mention_for_insertion_data,
trained 0.07 min, eval acc 0.297 (19/64). NO key/format mismatch.
- IMPORTANT correction: the edge bi-encoder uses `process_mention_for_insertion_data` (NOT the
  generic `process_mention_data`). Expected keys = mention, context_left/right, label_concept_ori,
  parent, child, edge_label_id, entity_label_id, optional world. These EXACTLY match the Zenodo
  jsonl as wired. So the symlink layout is correct as-is; no remapping required.
- Smoke command lives in this session; output_path was models/biencoder/debug_smoke (throwaway).
BLOCKER for step 4 (full train batch16 200k + top-50 search): GPU. GPU0 ~11GB free, GPU1 ~5.6GB
free (seg jobs own the rest, 2-3 days into runs). Full batch-16 SapBERT-base train likely > 11GB.
=> needs user GPU decision: (a) small batch to fit GPU0 headroom now, (b) wait for a seg job to
   finish, or (c) queue on a freed GPU later.

### 2026-06-23 (cont.): USER DECISION = wait for a GPU. deeponto installed during the wait.
- GPU plan (user, 2026-06-23): WAIT for a seg job to free a whole 48GB GPU before the faithful
  batch-16 full train, to protect the seg training (thesis priority). NOTE: seg jobs 307 & 311
  have UNLIMITED time limit (no TIME_LEFT/ETA) -> no predictable free time; user will signal when
  a GPU opens (or kill a job). Full train is NOT queued/auto-launched (user did not pick a watcher).
- deeponto 0.8.8 INSTALLED in onto38 (step 5 env now complete). Gotchas for the record:
  * Plain `pip install deeponto==0.8.8` FAILS: a transitive sdist's *build* deps require
    numpy>=2.0, which has no py3.8 build -> build-isolation error. FIX: install with
    `pip install --only-binary=:all: deeponto==0.8.8` (forces wheels, skips the source build).
  * deeponto 0.8.x prompts INTERACTIVELY (click.prompt) for JVM memory at import of
    deeponto.onto.ontology. With no stdin it raises click.Abort and the import dies. => any script
    that imports deeponto (enrichment: candidate_analysis.py -> preprocessing/onto_snomed_owl_util
    .load_SNOMEDCT_deeponto) MUST be run as `echo "8g" | python ...` (or otherwise feed stdin),
    else it hangs/aborts. Verified: JVM starts successfully on the system Java 8.

## READY-TO-FIRE: full Edge-Bi-encoder train + top-50 search (launch when a GPU frees)
Driver: repo/Edge-Bi-enc+Cross-enc.sh runs train_bi -> rep_ents -> candidate gen (search) ->
enrichment (deeponto) -> cross-enc, all in one. Faithful Disease top-50 invocation (from their
run-example), with-context, batch 16, 200k pairs for cross-enc data:
    cd repo && ./Edge-Bi-enc+Cross-enc.sh Disease true 50 25 16 false true 200000 \
        > results_log_Disease_edge_biencoder_top50_200k_final.txt
(set CUDA_VISIBLE_DEVICES to the freed GPU; script currently hardcodes =0 near its top.)
Smoke already proved the bi-encoder trains on the wired data; deeponto enrichment env is ready.

### 2026-06-23 (cont.): QUEUED on Slurm (job 312). Starts automatically when a GPU frees.
- sbatch wrapper: scripts/sbatch_disease_bienc_top50.sh (partition=gpu, gres=gpu:rtx6000ada:1,
  16 cpus, 64G, 72h). Submitted -> JOB 312, PENDING (reason Priority), behind seg jobs 307/311.
  Under --gres=...:1 the GPU shows as index 0, matching the driver's hardcoded
  CUDA_VISIBLE_DEVICES=0 -> no clash with seg training.
- Slurm logs: /home/kkolpetinou/slurm-ccp_disease_bienc_top50-312.out (+ .err). Detailed driver
  log: repo/results_log_Disease_edge_biencoder_top50_200k_final.txt.
- REQUIRED PATCH applied: deeponto/onto/ontology.py click.prompt(JVM mem) -> os.environ
  ["DEEPONTO_JVM_MEM"] (default 8g). Without it the batch job aborts at eval_biencoder
  (-> nn_prediction -> deeponto import) for lack of a TTY. Verified imports clean with no stdin.
  NOTE: this patch lives in the onto38 site-packages; re-apply if the env is rebuilt.
- This single job covers checklist steps 4 (train+search), 5 (enrichment), 6 (cross-enc+inference).
  On completion: read InR_any@k / InR_all@k from the results log => GO/NO-GO #1 + recall@pool.

### 2026-06-23 (cont.): PRIOR-ART check on two method-improvement ideas (do NOT pivot)
Asked whether 2 accuracy ideas are novel. Both already exist; one is a usable back-pocket lever.
- IDEA 1 = concept-first retrieval (find parent/child concepts, then build the edge): NOT NOVEL.
  This is the whole taxonomy-expansion/completion field: "Find Parent then Label Children",
  TaxBox "Insert or Attach" (ACL 2024), TaxoExpan, HiExpan, HyperExpan, TMN. Also partly
  duplicates Oxford's existing edge-enrichment (1 hop up parents / 1 hop down children). DROP.
- IDEA 2 = hard negatives for the Edge-Bi-encoder: technique is old (BLINK, all dense retrieval),
  BUT verified the Oxford Edge-Bi-encoder trains with ONLY in-batch negatives (max-margin triplet,
  alpha~0.2), NO hard-negative strategy. So structure-aware hard negatives (sibling/ancestor/
  descendant edges in the same subtree = the confusions that matter) are an UNEXPLOITED hole in
  THIS pipeline. Not a headline, but a legit SEPARATION lever: better-separated scores -> smaller
  conformal sets -> directly de-risks Phase-1 GO/NO-GO. KEEP as rescue if calibration sets explode.
- Idea 3 (hyperbolic embeddings) also already exists (HyperExpan). DROP.
- Dataset note: taxonomy-expansion methods run on SMALL CLEAN benchmarks (MAG-CS/PSY, WordNet-Verb/
  Noun, SemEval-2016 Task13 Sci/Env/Food) where the query is a CLEAN TERM. NOT comparable to
  OET/SNOMED: their high Hit@1/MRR are on a much easier problem; most wouldn't run unmodified on
  SNOMED mention-from-text + complex/logical edges + NIL discovery. No shared benchmark => no
  head-to-head "better/worse". DECISION stays: novelty = calibration/reliability, not accuracy.

### 2026-06-24: job 312 finished — CORE PIPELINE OK, crashed only at final inference (flair)
- Job 312 ran ~14.5h (17:32 -> 08:08), exit code 1. Bi-encoder train + edge encode + top-50 search
  + edge enrichment + cross-encoder (4 epochs, best epoch 2, ~66min/epoch) ALL COMPLETED and saved:
    models/biencoder/mm+Disease2017AA-tl-sapbert-NIL-bs16/{pytorch_model.bin,top50_candidates}
    models/SNOMEDCT-US-20140901-Disease_ent_enc_re_tr/*.t7  (edge encodings, 730MB)
    preprocessing/saved_cand_ids_..._re_tr.pt
    models/crossencoder/mm+Disease-2017AA/original-NIL-top50-pubmedbert/pytorch_model.bin
- CRASH cause: final step run_bio_benchmark+.py -> main_dense_plus -> blink/ner.py -> `import flair`
  (flair was deliberately skipped). It's an IMPORT-time failure only; NER is never used (mentions
  are given, NER.get_model() at main_dense_plus.py:661 is on a path we don't hit).
- FIXES applied (both in onto38 / repo, re-apply if env rebuilt):
  * blink/ner.py: wrapped `from flair...` in try/except -> SequenceTagger/Sentence=None.
  * pip install matplotlib (main_dense_plus imports it). Verified full import chain now clean.
- Cross-encoder metrics ALREADY in results_log_Disease_edge_biencoder_top50_200k_final.txt are
  VALIDATION reranking, CONDITIONAL on retrieval (rec_at_50=1.0 => pool always contains gold):
    ins_at_1_any 0.638 | ins_at_5_any 0.774 | ins_at_10_any 0.837 | ins_at_10_all 0.326 |
    ins_all 0.350 | ins_any 0.690 | prec_at_10 0.508.
  These are NOT the end-to-end test-NIL InR@k (those include retrieval misses) => NOT yet
  comparable to published ~0.30 InR_any@10. Need run_bio_benchmark+ output.
- ACTION: job 322 = inference-only re-run (scripts/sbatch_disease_inference.sh ->
  Edge-Bi-enc+Cross-enc_INFERONLY.sh with train_bi/rep_ents/eval_biencoder/train_cross=false;
  loads the saved models; --debug keeps test-NIL FULL, only samples in-KB valid). Output ->
  results_log_Disease_INFERENCE.txt. On completion: read end-to-end test-NIL InR@k => GO/NO-GO #1.

### 2026-06-24: GO/NO-GO #1 RESULT — pipeline reproduced end-to-end (CONDITIONAL GO)
Inference completed on GPU0 (shared w/ avezakis job 319, user-approved). Two more trivial
fixes en route: (a) created valid-sample.jsonl = head -1000 valid.jsonl (debug-path expects it);
(b) mkdir results/ (script writes summary .txt there). Cosmetic crash writing results/ AFTER all
metrics computed + predictions saved.
END-TO-END test-NIL (275 mentions; these INCLUDE retrieval misses, rec_at_50=0.496 != 1.0):
  InR_any@1=0.054 | @5=0.163 | @10=0.203 | @50=0.496(=retrieval ceiling)
  InR_all@1=0.015 | @5=0.062 | @10=0.101 | @50=0.377
valid-NIL: InR_any@10=0.237, InR_any@50=0.590.
EXACT config-matched comparison (got full paper; Table 3 row "+Edge-Cross-enc" k=50, TEST col):
  metric        ours   paper(test)
  InR_any@1     5.4%   7.6%   (below)
  InR_any@5     16.3%  15.6%  (ABOVE)
  InR_any@10    20.3%  26.5%  (~6pt below)
  InR_all@1     1.5%   1.5%   (EXACT)
  InR_all@5     6.2%   4.7%   (ABOVE)
  InR_all@10    10.1%  8.7%   (ABOVE)
  recall@50     49.6%  50.0%  (Table 2 Edge-Bi-enc test; NEAR-EXACT) ; InR_all@50 37.7 vs 38.0
=> FAITHFUL REPRODUCTION. Retrieval reproduced almost exactly (49.6 vs 50.0). Selection matches/
   exceeds on 4/6 metrics; only InR_any@10 (& @1) softer. 276 test mentions => paper itself flags
   "high variance" val-vs-test, so ~6pt swing is within their stated noise. GO/NO-GO #1 = CLEAR GO
   (was conditional). "~30%" in brief = their VALIDATION@10; their TEST@10 is 26.5%.
   Paper Appendix-1 confirms our exact recipe: bi-enc bs16/1ep, cross-enc bs1/4ep, 200k cross rows,
   48GB GPU; bi-enc train ~29h on their RTX8000 (we did full pipeline ~14.5h on RTX6000 Ada).
KEY DIAGNOSIS (matters for the method + the conformal plan):
  - Cross-encoder is GOOD: conditional-on-retrieval reranking (job 312 val, rec@50=1.0) hit
    ins_at_10_any=0.84. End-to-end only 0.20 because retrieval recall@50 is just ~0.50.
  => BOTTLENECK = bi-encoder RETRIEVAL recall, NOT cross-encoder ranking. Half the gold edges
     never reach the top-50 pool. This is exactly where hard-negative training (the back-pocket
     lever, [[project_nlp_toolbox_method_search]] idea-2) would help, AND it caps any conformal
     set: can't cover a gold edge that was never retrieved.
DELIVERABLE SECURED: per-edge cross-encoder SCORES saved (despite .jsonl name, it's ranked text
  "parent -> child (score), ..." per mention) at:
  models/crossencoder/mm+Disease-2017AA/original-NIL-top50-pubmedbert/mm-{test,valid}-NIL_*.jsonl
  (275 test rows; scores descending, e.g. dermatitis -> "disorder of skin -> stasis dermatitis
   (0.987), ..."). This is the raw material for the split-conformal layer (Phase 1).
PATCHES TO RE-APPLY IF ENV REBUILT: ner.py flair-optional; pip matplotlib; deeponto JVM env-var
  (all in STATUS above). Inference-only driver: repo/Edge-Bi-enc+Cross-enc_INFERONLY.sh.

### 2026-06-24: PHASE 1 START (user chose conformal). Structured score dump built.
Need per-mention (gold, per-candidate score) to calibrate. Cleanest source = evaluate_edges
returns res["y"] (gold multi-hot [n_mentions, L]) and res["yhat_raw"] (cross-enc sigmoid scores
[n_mentions, L]); L = top-k candidate slots after enrichment. NO fragile name-matching needed.
- PATCH (main_dense_plus.py, re-apply if repo reset): _run_crossencoder gains dump_tag kwarg;
  after evaluate_edges, if env CCP_DUMP_DIR set -> np.savez(<dataname>.npz, y, yhat_raw). Call
  site passes dump_tag=args.dataname (per-dataset: mm-valid-NIL / mm-test-NIL, confirmed line 354).
- Re-ran inference (GPU0) with CCP_DUMP_DIR=phase1/dumps -> mm-valid-NIL.npz (calib, 329 ment),
  mm-test-NIL.npz (test, 275 ment). [job: background be7ku5481]
- CONFORMAL DESIGN (split-conformal, "any"-coverage over edges):
  cal nonconformity s_i = 1 - max_{gold j} yhat_raw[i][j]; if no gold retrieved (y[i].sum()==0)
  s_i=1 (gold uncoverable). qhat = ceil((n+1)(1-a))/n quantile. test set = {j: score>=1-qhat}.
  EXPECTED CRUX (GO/NO-GO #2): recall@50~0.50 => ~half of mentions have NO gold in pool => for
  1-a > ~0.5 the quantile forces threshold->0 => sets explode to full pool. i.e. exact-edge
  any-coverage at 90% is INFEASIBLE with finite sets -> must reframe as risk-coverage/ABSTAIN
  (refer-to-curator) + hierarchy-aware ANCESTOR coverage. Phase-1 deliverables: coverage vs a,
  set-size vs a, risk-coverage/AURC, ECE, and the recall ceiling as the hard bound.
- Phase-1 work dir: calibrated-concept-placement/phase1/ (dumps/ + conformal.py to come).

### 2026-06-24: PHASE 1 RESULTS — naive conformal NO-GO; points straight at the novelty
Script phase1/conformal.py on the dumps (calib=valid-NIL n=329, test=test-NIL n=276).
(1) EXACT-EDGE split-conformal "any"-coverage = NOT VIABLE (GO/NO-GO #2 as the brief feared):
    - any-coverage HARD-CAPPED at test recall@pool = 49.6% (gold simply absent from pool ~half
      the time); sets EXPLODE to the full 50-edge pool for any target >~0.5 (still only 0.496 cov).
    - even at target 0.50 empirical coverage UNDERSHOOTS to 0.377: calib recall@pool 0.59 vs test
      0.496 => valid-NIL/test-NIL NOT exchangeable (the concept drift the paper itself flags) =>
      the marginal conformal guarantee is itself unreliable here. So small-set exact-edge story dead.
(2) RISK-COVERAGE with the model's own confidence ALSO FAILS (verified, not an artifact):
    - confidence (top-1 score) vs top-1 correctness corr = 0.05 (~zero); AURC=0.957.
    - most-confident decile: mean score 0.999, InR_any@1 = 0.000. Model is MOST WRONG when MOST
      CONFIDENT (confident-FP / overconfidence; saturates to ~0.999, often when gold not even
      retrieved). The 15 correct top-1 cases sit at conf-ranks 70-253, never the top.
    - NB this mirrors the user's seg project confident-FP/mimic finding ([[project_mimic_score_diagnostic]]).
(3) Aggregate ECE_test=0.060 (looks fine) but HIDES the high-confidence-tail miscalibration in (2).
INTERPRETATION (go/no-go, per [[feedback_positive_method_only]]): the two NAIVE fallbacks (small
  sets; abstain-by-confidence) are both RED. This is exactly the wall the brief anticipated, and it
  motivates the actual NOVELTY, both UNTESTED and both rescue-shaped:
    (a) HIERARCHY-AWARE / lenient ANCESTOR coverage — relax "exact gold edge" to "correct ancestor
        region" (Wu&Palmer / subsumption distance). The paper's own case study says predictions are
        "not completely wrong" (parent too general, child near gold) and they LEFT lenient eval as
        future work. This directly lifts the 49.6% ceiling and could make confident-but-near hits count.
    (b) a BETTER confidence signal that detects "gold likely not retrieved" (retrieval-failure abstain),
        since raw cross-enc confidence conflates sure-and-right with sure-but-gold-absent.
  Needs deeponto ontology graph (already installed) for ancestor/lenient distance. results:
  phase1/conformal_results.json.

### 2026-06-24: PHASE 1b START — hierarchy-aware ancestor coverage (user-chosen pivot)
Goal: relax exact-edge coverage to lenient/ancestor coverage to beat the 49.6% ceiling + make
confident-but-near predictions count. KEY DISCOVERY: the repo ALREADY has the lenient machinery,
just off by default:
  - main_dense_plus.py `--measure_wp` path (~line 1051) computes Wu&Palmer pred-vs-gold via deeponto.
    This is literally the Oxford "lenient eval" future-work gap. Our novelty = wrap it in CONFORMAL.
  - helpers in preprocessing/onto_snomed_owl_util.py: extract_SNOMEDCT_deeponto_taxonomy (OntologyTaxonomy
    struct reasoner), calculate_wu_palmer_sim(taxo,iri1,iri2,...) [WP=2*depth(LCA)/(d1+d2)],
    get_shortest_node_depth, get_lowest_common_ancestor (both with dict caching), get_iri_from_SCTID_id.
PATCH 2 (main_dense_plus.py ~line 1040, re-apply if reset): under CCP_DUMP_DIR, also dump
  <dataname>_edges.json = per mention {pred_edges:[[parentSCTID,childSCTID]...] aligned w/ scores,
  scores:[...], gold_edges:[[p,c]...] FULL gold set incl non-retrieved}. Built from the pipeline's
  own pred_idx_tuples/gold_idx_tuples (no name-matching). Re-running inference now (bg b7z22lr9z).
NEXT (hierarchy-aware conformal, once dump lands):
  1. load SNOMED-Disease taxonomy (deeponto) once; 2. per mention compute edge-level WP between each
     top-k pred edge and each gold edge (parent-WP and child-WP); 3. FIRST GO/NO-GO: does lenient
     (WP>=tau) coverage lift meaningfully above exact 0.496? + is the model's confident-but-wrong set
     actually ontologically NEAR gold (rescuing risk-coverage)? 4. then hierarchy-aware nonconformity
     (ancestor-collapse) conformal sets w/ lenient coverage guarantee.

### 2026-06-24: PHASE 1b LENIENT-CEILING = GREEN. Hierarchy-aware pivot validated.
phase1/lenient_ceiling.py (deeponto WP over SNOMED-Disease 2014.09 taxonomy, parent-WP, top-10):
  TEST (n=276):  exact@10=0.203 | lenient parent-WP>=0.70 = 0.895 | >=0.80 = 0.750 | >=0.90 = 0.543 |
    >=0.95 = 0.417. best parent-WP mean 0.887 median 0.941.
    *** AMONG THE 220 EXACT MISSES: best parent-WP mean 0.857, median 0.889, 68.6% have WP>=0.80.***
  VALID (n=329): exact@10=0.237 | WP>=0.70 0.888 | >=0.80 0.842 | >=0.90 0.559 | >=0.95 0.502; misses 79% WP>=0.8.
=> KEY: when the model misses the EXACT gold edge it is still ontologically CLOSE (parent WP~0.86)
   to a gold parent. The 49.6% exact ceiling SHATTERS under lenient matching (~0.90 at WP>=0.7,
   0.54 even at a strict WP>=0.9). Quantifies the Oxford case-study claim ("predictions not
   completely wrong"). This rescues BOTH stories: coverage-guaranteed ANCESTOR sets become feasible,
   AND confident-but-"wrong" preds are confident-and-NEAR (fixes risk-coverage). GO to build the
   hierarchy-aware conformal layer.
   Caveats (honest): parent-only WP so far (child not yet in); WP>=0.7 is generous (region-level),
   clinically-meaningful threshold TBD; complex-concept WP via atomic-id extraction (heuristic, ~2%).
   owl symlinked into repo/ontologies/ (both 2014/2017). results: phase1/lenient_ceiling.json.
   FULL-EDGE update (2026-06-24, parent+child WP, NULL-leaf->parent depth+1, complex via atomic
   extraction): TEST exact@10=0.203 | full-edge-WP>=0.70=0.768 >=0.80=0.620 >=0.90=0.409;
   best full-edge-WP mean 0.850; AMONG 220 misses full-edge-WP mean 0.808, 52%>=0.8, 26%>=0.9.
   VALID similar (>=0.70=0.845, >=0.80=0.638). => child axis lowers it vs parent-only (0.895->0.768
   at 0.70) but STILL 3-4x the exact 0.203 ceiling, and misses are still right-next-door (mean .81).
   PIVOT VALIDATED on the honest stricter metric. (THING parent -> 0 here, minor undercount.)
NEXT DESIGN FORK (shapes the paper): lenient-correctness definition for the conformal layer --
   (a) thresholded WP>=tau binary, (b) continuous WP risk-coverage/AURC, (c) ancestor-hop/subsumption
   set; and parent-only vs full-edge(parent+child). Then: hierarchy-aware nonconformity -> coverage-
   guaranteed sets that COLLAPSE TO A SHARED ANCESTOR under uncertainty (the brief's core novelty).

### 2026-06-24: PHASE 2 BUILD — hierarchy-aware conformal, CONTINUOUS WP (user-chosen metric).
Decision: lenient-correctness = continuous Wu-Palmer risk-coverage (no arbitrary threshold; report
across WP levels). Two-script design:
  - phase2/precompute_wp.py: ONE-TIME ontology pass -> phase2/wp_cache_{valid,test}.npz, arrays
    [n,50] = scores + per-candidate full-edge WP-to-best-gold + exact flag. (expensive; cache it.)
    [running bg bl0dn8xj7]
  - phase2/conformal_wp.py (ready, reads cache, instant): (A) WP risk-coverage AURC (rank by top-1
    score, quality=top-1 full-edge WP) -- tests if abstain is RESCUED once near preds get partial
    credit (Phase-1 binary AURC was 0.957). (B) coverage-guaranteed CORRECT-REGION sets: for quality
    w in {1.0,0.9,0.8,0.7} x target cov {.7,.8,.9}, split-conformal tau on valid, report test emp-cov
    + set size. w=1.0 == exact (Phase-1 blow-up) for contrast. Headline = (quality, coverage, set-size)
    curator-usability surface. NEXT after cache lands: run conformal_wp.py, read the tradeoff surface.

### 2026-06-24: PHASE 2 RESULT — hierarchy-aware continuous-WP conformal WORKS (method viable).
phase2/conformal_wp.py on the cached WP matrix (fix: exact tuple-match -> WP=1.0, since repo's NULL
convention scores exact LEAF edges <1; verified my exact==pipeline y, both recall@pool 0.496).
(A) WP RISK-COVERAGE = RESCUED. confidence (top-1 score) now POSITIVELY tracks WP-quality:
    accept-all mean WP 0.78; accept top-30% most-confident -> mean WP 0.885; top-10% -> 0.906.
    AURC(1-WP)=0.153 (vs Phase-1 binary exact AURC 0.957). KEY: the confident-FP from Phase-1 are
    confident NEAR-misses -- the model knows when it's in the right REGION, just not the exact edge.
(B) COVERAGE-GUARANTEED CORRECT-REGION SETS (calib=valid, test=test, full-edge WP):
    WP>=0.7 (right region): set ~9 edges (median 5) -> cov 0.66 | ~30 edges -> cov 0.89
    WP>=0.8 (close)        : set ~19 (med 16) -> 0.65 | ~30 -> 0.74
    WP>=0.9 / exact        : forces FULL 50-pool -> ~0.62-0.65 (the Phase-1 small-set blow-up persists
                             only at the exact/near-exact level). oracle ceiling by quality:
                             {1.0:0.62, 0.9:0.65, 0.8:0.86, 0.7:0.96}.
=> The exact-edge story is dead but the REGION story is alive: small, coverage-guaranteed sets that
   point a curator at the right ontology neighborhood. CONTRIBUTION VALIDATED.
HONEST CAVEATS: (i) emp coverage UNDERSHOOTS target (valid recall .59 > test .50 drift) => the
   brief's Mondrian/class-conditional conformal is the principled fix (next). (ii) w=1.0-WP (0.62)
   is slightly looser than strict exact-tuple (pool 0.496 / top-10 0.20) due to complex-concept
   atomic-unwrapping -- report strict-exact separately. (iii) leaf/NULL WP convention is the repo's.
   results: phase2/conformal_wp_results.json. cache: phase2/wp_cache_*.npz.
NEXT (brief Phase 2): class-conditional/Mondrian conformal (fix drift, per-subgroup guarantees:
   leaf vs non-leaf, depth band); full risk-coverage/AURC + ECE figures; then write-up.

### 2026-06-24: PHASE 2 FIGURES done (phase2/figures.py -> phase2/figures/*.png + metrics.json).
fig1 risk_coverage: WP region-quality 0.78->0.91 (most-confident) vs exact-edge ~0 flat (the headline).
fig2 calibration: cross-enc score OVERSTATES exact (score~1 -> exact-rate ~0.04; ECE_exact=0.060
  aggregate but high-score bins badly off); WP-quality high-flat ~0.8 across scores -> raw scores
  poorly calibrated => justifies the conformal recalibration.
fig3 setsize_coverage FRONTIER: WP>=0.7 ~10-edge set -> 0.65 cov, ~30 -> 0.88; WP>=0.8 a bit larger;
  exact & WP>=0.9 only exist at set size 27-50 (no small sets) = the curator-effort headline.
fig4 coverage_calib: empirical < target (undershoot) from valid->test drift => motivates Mondrian.
metrics: AURC_wp=0.153, AURC_exact=0.957, meanWP_all=0.783, exact@1=0.054, ECE_exact=0.060.

### 2026-06-24: MONDRIAN / class-conditional conformal (brief core novelty). phase2/mondrian.py
Groups observable at test = top-1 predicted leaf-ness (child=NULL) + predicted-parent DEPTH band
(tertiles). fig5_mondrian_depth.png. Honest, nuanced result:
- MARGINAL conformal HIDES big per-subgroup gaps (brief's prediction confirmed): at W>=0.7 target
  0.80, depth coverage = shallow 1.00 / MID 0.566 / deep 0.720 (mid badly under-served); leaf vs
  non-leaf more even (~0.72/0.73).
- MONDRIAN REBALANCES: lifts the worst group (target 0.70 leaf: 0.50->0.72; depth mid 0.47->0.54)
  and tightens over-served groups with SMALLER sets (shallow set 11->6). More equitable per-group.
- BUT it does NOT fix the AGGREGATE undershoot: pooled cov ~0.73 marginal -> ~0.71 mondrian (target
  0.80). => the valid->test drift is GLOBAL/within-group (valid uniformly easier), not just group
  COMPOSITION, so per-group recalibration redistributes coverage but can't beat the global shift.
  HONEST FINDING for the paper: marginal hides subgroup inequity (Mondrian fixes that), but closing
  the absolute gap needs robustness to concept drift (cross-conformal / report-as-limitation), since
  out-of-KB valid vs test are not exchangeable (the paper itself documents this drift).
- Caveat: small n per group (mid nC=126, deep 80, leaf 113) -> noisy quantiles; mid<deep ordering
  may be partly noise. results: phase2/mondrian_results.json.

### 2026-06-24: CPP (2nd benchmark) SET UP for generalization (addresses single-dataset weakness).
- NIL index gotcha SOLVED: driver hardcodes NIL_ent_ind=64076 (Disease atomic count); for CPP must be
  173177 (= CPP distinct atomic concepts, verified =175895 concepts-2718 complex; Disease pattern
  64900-824=64076 confirmed). Made repo/Edge-Bi-enc+Cross-enc_CPP.sh with NIL_ent_ind=173177.
- Data wired: repo/data/MedMentions-preprocessed+/CPP/st21pv_syn_attr-all-complexEdge-edges-final/*
  symlinks + valid-sample.jsonl; CPP ontology jsonl+owl symlinked into repo/ontologies/.
- DEBUG SMOKE PASSED (GPU1, NIL=173177): read 1,398,111 train pairs (matches paper), eval ~0.30, clean.
- sbatch: scripts/sbatch_cpp_full.sh (gpu, rtx6000ada:1, 80G, 96h; top-50/25 to match Disease).
  CCP_DUMP_DIR=phase1/dumps_cpp (SEPARATE -- dump tag mm-{test,valid}-NIL collides with Disease dir).
  CPP bigger (1.4M pairs, 626k edges) -> expect ~24-36h. Patches (flair/matplotlib/deeponto) already in.
- GPU: user seg job 307 now GONE (finished); avezakis holds BOTH GPUs (321,325). Long CPP train ->
  QUEUE on Slurm (auto-start when a GPU frees); sharing a GPU for 24-36h = poor etiquette/OOM risk.
- After it completes: rerun phase1/phase2 analysis on dumps_cpp (precompute_wp on CPP owl, conformal_wp,
  figures, mondrian) to show the method generalizes across ontology subset.

### 2026-06-25: NOVELTY PROBE — LLM-RAG retrieval anchoring = WEAK/NO-GO (principled). ~30min, no API.
Idea (user-chosen): for the ~50% retrieval misses, have an LLM name the gold PARENT to recover the
edge. Diagnostic (no LLM calls needed) on the 100 wrong-region-miss test mentions (gold parent NOT
in retrieved parents):
- 64% have deepest gold parent = "disease (disorder)" = the GENERIC ROOT of the disease branch
  (depth~3). Gold places these directly under root (likely a dataset artifact: specific 2014 parent
  didn't exist). Anchoring to a specific parent CANNOT help / won't match. (NB part of why recall caps ~50%.)
- 36% have a specific gold parent, but DOMINATED by "renal impairment" (CKD/AKI, ~21) which is
  LEXICALLY OBVIOUS -> SapBERT already encodes it; LLM adds no knowledge the embedding lacks. The
  miss there is edge-formation/enrichment mechanics, not unknown parent.
- Genuine "LLM knows a non-lexical parent SapBERT can't" residual (e.g. dental fluorosis->mottling of
  enamel) is SMALL (~10-15 mentions), concentrated. Not worth an LLM-RAG retriever build.
RAW test-NIL gold-parent dist (965 pairs, independent of dump): renal impairment 50%, disorder of
skin 25%, disease(root) 18% -- the easy specific parents are mostly RETRIEVED; the MISSES skew generic.
DECISION: do NOT build LLM-RAG anchoring. Reusable as ANALYSIS (recall ceiling = ~half generic-gold
artifact + half enrichment mechanics). Back-pocket novelty = hierarchy-aware BACK-OFF output (option
2, low-risk, ours) or LLM-as-verifier (option 3). Probe data: phase2/llm_anchor_probe_sample.json.
NB: a mention-text<->dump join is off-by-one (275 vs 276 unique); gold-edge analyses are reliable
(self-contained per row, reconciled with npz y), only mention-TEXT labels need the fixed join.

### 2026-06-25: NOVELTY PROBE 2 — hierarchy-aware BACK-OFF output = YELLOW (modest, real, low-effort).
phase2/backoff_probe.py: back-off ancestor = LCA of top-m predicted parents; coverage = ancestor
subsumes a gold parent; specificity = ancestor depth (root~3). TEST:
  m=1 cov0.41 d7 | m=2 0.60 d6 | m=3 0.65 d5 | m=5 0.72 d4 | m=8 0.81 d3(root) | m=10 0.83 d3.
  (valid similar; gold-parent depth median 3 -- half are the generic-root bucket.)
=> Real coverage/specificity TRADEOFF but can't get BOTH: 80%+ coverage only at depth3=root(useless).
   Useful regime = ONE ancestor at depth5-6 covering ~60-65%. High-cov end partly inflated by
   generic-gold cases. VALUE: at ~65-72% cov, a single ancestor vs the flat method's ~10-15 edges =
   more COMPACT; biggest win on UNCERTAIN mentions where the flat set explodes to 50. Together =
   "adaptive granularity" output (precise set when confident, broad ancestor when not).
   VERDICT: genuine + novel + LOW-EFFORT (machinery exists) but SECONDARY/usability contribution,
   not a headline. Worth including to complete the story; frame honestly as modest.
   results: phase2/backoff_probe.json.

### 2026-06-25: MAJOR finding — 64% of MM-S14-Disease test gold is DEGENERATE ("disease->NULL").
Triggered by user pushback on "we don't know the answer" (CORRECT pushback: we DO know gold from 2017).
FACTS (verified): 176/276 (64%) test-NIL mentions have gold = ("disease (disorder)" SCTID 64572001
-> NULL) ONLY = place as a leaf under the GENERIC ROOT. Same in test-NIL.jsonl AND test-NIL-all.jsonl
(not a filtering artifact). This is the file the paper + we evaluate on (run_bio_benchmark reads it).
Examples: acute kidney injury, AKI, cervical spondylosis, CAD, AVF, SPA... CKD/dermatitis are WELL-FORMED
("renal impairment -> {CKD stages}", "disorder of skin -> harara").
=> These are the dataset's LEAF cases; the paper's OWN Table 2 shows leaf InRany 38% vs NON-leaf 95%
   (test) -- consistent. The degenerate gold = the hard/uninformative leaf bucket.
OUR metrics SPLIT (test, best full-edge WP@10):
   DEGENERATE (n=176): exact-in-pool 0.48 | WP>=0.7 cov 0.65 | mean WP 0.77
   WELL-FORMED (n=100): exact-in-pool 0.52 | WP>=0.7 cov 0.97 | mean WP ~1.0
   ALL (276):                              WP>=0.7 cov 0.77 (the diluted headline)
=> REFRAME: on the REAL placement task (well-formed/non-leaf), the hierarchy-aware method gets ~0.97
   region coverage -- the degenerate 64% drags the headline to 0.77. "accuracy structurally capped"
   is REFINED: real-case accuracy is HIGH; the cap is degenerate gold no method can/should solve.
LIKELY CAUSE (inference, flag as such): version-diff construction couldn't find a specific 2014-existing
   parent/child (concept lineage new/restructured by 2017, e.g. AKI terminology), defaulted to root-leaf.
IMPLICATIONS: (1) report ALL metrics split well-formed vs degenerate (or leaf/non-leaf) -- more honest
   + makes method look strong on real task; (2) this characterization is itself a publishable analysis
   finding; (3) my earlier "we don't know the answer/junk" wording was WRONG -- gold is KNOWN but
   degenerate/uninformative for 64%. Caveat: well-formed n=100 is ~30-40 distinct concepts (CKD x20).

### 2026-06-25: NOVELTY PROBE 3 — LLM-as-VERIFIER (paper's RAG/prompting future direction) = NO-GO (principled). No API.
User wants methodological novelty via the paper's stated future directions (LLM explanations / advanced
RAG / prompting / assist terminologists). Mapped those onto our diagnosis BEFORE building:
  - RAG/prompting FOR ACCURACY fights a measured ceiling: end-to-end accuracy is capped by retrieval
    recall@pool 0.496 (gold absent from top-50 pool half the time); an LLM reranker can't rank an edge
    that was never retrieved. LLM-RAG *anchoring* already NO-GO (probe 1).
  - One LLM angle NOT yet ruled out = LLM-as-VERIFIER (rerank/abstain over the EXISTING pool, then wrap
    its confidence in conformal). User chose to probe this first.
ORACLE CEILING (no-API gate, phase1/dumps): verifier can only help when gold IN-pool (137/276 test).
  Of those: 85 degenerate (gold parent = disease-root 64572001) -> verifier INAPPLICABLE (LLM never
  picks "disease->root-leaf" for AKI/TNBC); 52 WELL-FORMED in-pool, of which 41 "recoverable" (gold
  in-pool but ranked >1). Well-formed in-pool top1=0.21 / top10=0.71 -> looked like real headroom.
VERIFIER ADJUDICATION (I acted as the LLM on 15 well-formed recoverable cards w/ concept NAMES; cards
  at scratchpad/verifier_probe_cards.json; builder reusable): <=2/15 recoverable. NO-GO, because:
  (1) Gold is SYSTEMATICALLY OVER-SPECIFIC vs the mention. mention "chronic kidney disease" (no HTN in
      context) -> every gold edge is a hypertension-specific child (renal impairment -> "CKD due to
      hypertension"); top-1 is a sibling ("hypertensive CKD"). Same parent, indistinguishable siblings,
      neither matches generic CKD. PsA mention -> gold child "PsA with DISTAL INTERPHALANGEAL joint
      involvement" (no signal in "PsA"). The exact answer is NOT determined by the input.
  (2) Where the distinction is real it's a near-synonym coin flip: TNBC top-1 "carcinoma of breast"
      vs gold "malignant tumor of breast" (TNBC is both) -> LLM can't reliably call it.
  (3) REGION already correct at top-1 (parent "renal impairment" rank1) -> no region headroom either
      (= the existing WP finding). (4) 5/15 "recoverable" gold are rank 11-45 -> a top-10 LLM reranker
      never even sees them.
INTERPRETATION: second PRINCIPLED negative, same disease as anchoring ("no signal the mention lacks").
  EXTENDS the headline: exact-edge recovery is INFORMATION-THEORETICALLY capped by noisy version-diff
  gold -- now shown INSIDE the well-formed 36%, not just the degenerate 64%. Strengthens the thesis
  "no LLM/RAG/rerank helps; contribution = calibrated region placement + curator triage" and pre-empts
  the reviewer's "did you try the paper's LLM directions?" => YES, ruled out for a measurable reason.
DECISION: LLM-verifier-for-accuracy + advanced-RAG-for-accuracy = NO-GO. SURVIVING LLM angle =
  LLM-generated EXPLANATIONS of the guaranteed region set (usability, doesn't need to beat noisy gold).
  Core methodological novelty = GRANULARITY-ADAPTIVE CONFORMAL (build next). Probe data: scratchpad/
  verifier_probe_cards.json; ceiling recomputable from phase1/dumps + ontology edges-all catalogue.

### 2026-06-26: CPP (2nd benchmark) job 326 FINISHED clean (exit 0) — reproduction + cross-dataset diff.
Dumps -> phase1/dumps_cpp/ (test n=432, valid n=568). Driver log results_log_CPP_..._final.txt.
REPRODUCTION (same regime as Disease => backbone faithful on CPP too):
  test : recall@pool 0.403 | InR_any@1 0.025 @5 0.132 @10 0.211
  valid: recall@pool 0.551 | InR_any@1 0.037 @5 0.116 @10 0.199
  (Disease was recall@pool 0.496 / InR@10 0.203 -> CPP slightly harder retrieval, same ballpark,
   same bi-encoder-recall bottleneck.)
CROSS-DATASET DIFFERENCE (real, for the writeup): CPP is NOT dominated by one degenerate root.
  Disease = 64% "disease(disorder) 64572001 -> NULL". CPP top gold-edge parents are SPREAD
  (236423003, 95320005=skin, 64572001=disease, 373873005, 71388002); "236423003 -> NULL" only
  2.8% of mentions, ANY-gold-parent-236423003 = 12.7%. => the single-root degenerate artifact is
  DATASET-SPECIFIC in magnitude; CPP's leaf cases are distributed across branches. For CPP we split
  the frontier by the dataset-agnostic LEAF criterion (all gold edges NULL-child) instead of one root.
IN PROGRESS: phase2/precompute_wp_cpp.py (CPP variant, paths swapped) building wp_cache_cpp_{valid,
  test}.npz over the 626k-edge CPP taxonomy (slow cold cache; PID 1531774, log precompute_wp_cpp.log).
  phase2/granularity_adaptive_cpp.py is STAGED to run the triage frontier (all / non-leaf / leaf) the
  moment the cache lands -> granularity_adaptive_cpp_results.json. Then compare to Disease frontier
  => cross-dataset generality claim (the single biggest publishability de-risk).

### 2026-06-26: CPP "is it improvable?" ERROR DECOMPOSITION (user challenged "don't just say hard").
Checked facts, not opinion. CPP exact-placement is HARD TO IMPROVE for 3 concrete reasons + a correction:
  (1) RETRIEVAL CAP WORSE: recall@pool 0.403 (Disease 0.496). 60% of gold never in top-50 -> exact
      hard-capped ~40%; only lever is a better RETRIEVER (crowded accuracy game), not rerank/calibration.
  (2) RERANK "HEADROOM" IS 4 CONCEPTS: in-pool non-leaf top10=0.84 but top1=0.14 looked improvable, BUT
      the 40 rerank-reachable mentions span only 4 distinct concepts (36/40 = literally "CKD"); the WHOLE
      non-leaf test set = 8 concepts / 64 mentions. Any "method" win here = overfitting ~4-8 concepts.
  (3) MISRANKED GOLD UNRECOVERABLE (same wall as Disease verifier NO-GO): mention "CKD" -> gold "renal
      impairment -> CKD DUE TO HYPERTENSION" (no HTN in text); gold differs from top-1 only by near-
      synonym PARENT (renal impairment vs kidney disease vs chronic renal impairment). Info not in input.
CORRECTION to 2026-06-26 entry above: I called CPP "less degenerate" — true ONLY for single-root
  concentration (236423003->NULL = 2.8%). By the LEAF test CPP is 85% leaf > Disease 64% => CPP's
  informative NON-LEAF task is SMALLER (15%, 8 concepts) than Disease well-formed (36%). CPP is a
  THINNER target for accuracy, not a cleaner one. Do not pitch CPP as an accuracy-method opening.
IMPLICATION: refutes "build a better algorithm on CPP" on data. SUPPORTS "ceiling+diagnosis generalize
  to a 2nd DAG". WARNING for the frontier: CPP fine-resolution auto-place numbers will be on 8 non-leaf
  concepts => NOT trustworthy in isolation; the trustworthy CPP result = leaf/region/refer triage, not
  a fine-resolution accuracy claim. (Cards builder reused from Disease, repointed to dumps_cpp + CPP cat.)

### 2026-06-26: CPP RETRIEVAL-STAGE decomposition (user: "why not improve the edge-SEARCH stage?").
The earlier-stage (bi-encoder edge search) IS the bottleneck (recall@pool 0.403). Decomposed the 258
retrieval misses; CORRECTION to a number I first reported wrong:
  - region-miss (gold PARENT region not even retrieved) = 236/258 (91%); region-hit-edge-miss = 22 (9%).
  - [CORRECTED] gold = generic-root-only among region-misses = 45% (NOT 4%; I first used wrong root id
    236423003 instead of true disease root 64572001). 37% are disease-root(64572001)-only specifically.
  - 58% of region-misses are CRYPTIC ACRONYMS/short tokens (AKI, KTRs, CBCT, EGSFI, AP); 45% are BOTH
    cryptic AND generic-root gold = unfixable from both ends.
  - per-mention: AKI=88 (gold="disease" root => SAME degenerate case as the opening-question example,
    now the dominant RETRIEVAL miss), miltefosine=40, KTRs=21, CBCT=19; 29 distinct mentions total.
  - Genuinely fixable residual (well-formed gold + resolvable mention an undertrained retriever missed)
    is SMALL (~handful: miltefosine, KTRs, rare syndromes). chemotherapy-type misses = convention
    disagreement (retriever's "administration of antineoplastic agent" not even wrong vs gold "regimen").
VERDICT: improving edge-search hits the SAME ceiling as reranking in different clothes (degenerate gold
  + cryptic/under-determined mentions + annotation noise). Hard-negative bi-enc training (known lever,
  unexploited in this pipeline) could recover the small residual + a few recall points => RAISES the
  calibrated tool's REGION-COVERAGE ceiling (capped by recall@pool) => helps OUR METHOD, not an accuracy
  paper. Not worth a retrain detour now; note as future work / ceiling-lifting option.

### 2026-06-26: 2025+ RELATED-WORK SCAN — is anything newer/better on OUR task? (user-requested).
Q: any post-ESWC-2024 paper that does NIL new-concept placement into SNOMED (OET/MedMentions, InR@k)
better than the Oxford backbone? ANSWER: NO head-to-head competitor; Oxford framework still the
reference method on that benchmark => stays the backbone. (Caveat: keyword + forward-cite search, not
exhaustive citation-graph crawl.) Adjacent 2025 work found, none beats Oxford on OET, NONE does calib:
  - Hierarchical Retrieval / OnT+HiT (Manchester, Hui Yang, Nov 2025, arXiv 2511.16698): hyperbolic
    ontology embeddings to RETRIEVE existing ancestors for OOV queries. Retrieval-only, no insertion,
    no NIL, 50-query eval, no UQ. = our edge-SEARCH stage done better; cite as SOTA hierarchy retriever
    + the "stronger retriever?" rebuttal. Bounded by our degenerate/cryptic miss structure.
  - Ontology enrichment using an LLM (JBI 2025, S1532046425000942): GPT-4 triples + lexical/semantic/
    knowledge-network similarity for concept placement, but DIFFERENT DOMAIN (Social Determinants of
    Health, not SNOMED) + GROWTH eval (SOHOv1 173 -> SOHOv2 572 concepts), NOT held-out InR@k, no
    head-to-head, no calibration. Matters for VENUE positioning: JBI publishes LLM concept-placement =>
    target-tier-friendly, but a JBI reviewer knows this space => must distinguish "calibrated reliability
    + accuracy-ceiling diagnosis" from "another LLM enrichment pipeline".
  - LLMs4OL 2025 Challenge (ISWC 2025): "Taxonomy Discovery" = build hierarchy among given terms, NOT
    NIL insertion into existing SNOMED. Adjacent, winners = hybrid LLM+domain-embedding+RAG. No UQ.
CONCLUSION: (1) not scooped; nothing newer is a better backbone. (2) 2025 trend = LLM+similarity/RAG +
  hyperbolic retrieval, all on ACCURACY/construction; UQ/conformal/calibration STILL EMPTY across every
  recent paper => our reliability niche confirmed open (strongest signal yet). (3) cite all 3 as 2025
  related work; community is active+overlapping (Oxford KRR + Manchester/Hui Yang + LLMs4OL) -> position
  carefully. Same-community note: OnT/HiT + ICON (taxonomy completion) + this scan all Manchester-adjacent.

### 2026-06-26: CPP TRIAGE FRONTIER RESULT — MIXED / partial generalization (not the clean n=1->n=2 win).
WP cache built (wp_cache_cpp_*.npz, 8927s). phase2/granularity_adaptive_cpp.py -> _cpp_results.json.
At precision>=0.80, max auto-place fraction:
  Disease all : exact0 fine0   region0.20 coarse0.53 | well-formed: fine1.00 | degenerate: coarse0.26
  CPP     all : exact0 fine0   region0    coarse0    | NON-LEAF(64): fine1.00 | LEAF(368): coarse0
REPLICATES (the DIAGNOSIS): (a) exact-edge auto-place = 0% on BOTH datasets every resolution =>
  "exact placement info-capped" holds on a 2nd DAG. (b) informative (non-leaf) cases auto-place at
  fine WP>=0.9 ~100% on both => the mechanism STRUCTURE (fine region for informative, refer the rest)
  generalizes.
DOES NOT replicate (the USABILITY YIELD): Disease all-comers had a usable COARSE tier (0.53 @p80);
  CPP all = 0 => CPP is an almost-entirely-REFER regime (85% leaf, recall@pool 0.40, cryptic/degenerate
  leaf mentions AKI/KTRs/CBCT too far from gold to certify even coarse). 
CRITICAL CAVEAT (flagged in advance): CPP "fine=1.00" is on 64 mentions / 8 distinct concepts (CKD-
  dominated) => consistent with Disease but NOT trustworthy as a standalone number; DO NOT headline it.
IMPLICATION for write-up (tempers the publishability read): strongest 2-dataset-backed claim = the
  DIAGNOSIS + the calibrated triage STRUCTURE, NOT a "we auto-place X% accurately" headline. The
  granularity-adaptive tool's YIELD is dataset-dependent (Disease: usable coarse tier; CPP: mostly
  principled referral). Present CPP as CONFIRMING THE CEILING, not boosting numbers. Still a legit
  applied-reliability contribution, but honest/tempered. Disease well-formed (100 mentions ~30 concepts)
  is also a small slice -> both fine-resolution wins need larger-scale / concept-level validation.

## Reordering vs the brief (decision 2026-06-23)
The brief's Phase 1 GO/NO-GO ("sets explode to near full pool") is the project's true crux,
not a tail risk: base signal is weak (~30% InR_any@10) and calibration budget is tiny
(276/432 mentions, split further for Mondrian). To guarantee 90% coverage on a top-10 hit rate
of ~30%, sets may be uselessly large — which would be a negative result.

Therefore we front-load a cheap **feasibility probe** so we learn this in week 0, not week 6:
1. Pull Zenodo processed data + any released score files.
2. If per-edge scores for the test mentions are available without rerunning the model, compute
   directly: what set size does 90% / 80% marginal coverage actually require? (script:
   `scripts/feasibility_probe.py`)
3. Decision rule:
   - set size <= ~10 edges at 90%  -> small-set usability story alive; proceed as planned.
   - set size huge (>~40)          -> pivot framing to risk-coverage/abstain (large sets OK
                                      because contribution is *referral*, not small sets)
                                      BEFORE investing in full env reproduction.

## GO/NO-GO gates (from brief Section 7)
- [x] GO/NO-GO #0 (added): feasibility probe — analytic result in `feasibility_probe.md`.
      Outcome: small-set 90% EXACT-edge coverage = NOT viable (best InR_any@10 ~26%).
      Project pivots to risk-coverage/abstain + hierarchy-aware ancestor coverage (was already
      the brief's strength). Hard ceiling now to measure = candidate-retrieval recall@pool.
- [ ] GO/NO-GO #1: reproduce published InR@k with the backbone.
- [ ] GO/NO-GO #2: split-conformal sets are usefully calibrated (not near full pool).

## Verification steps still owed (brief Section 8) — do early, they gate commitment
- [x] Forward-citation intersection (2026-06-25): searched concept-placement/taxonomy-expansion x
      conformal/calibration/UQ across several phrasings => EMPTY. No paper combines concept placement
      (or taxonomy expansion) with conformal/coverage guarantees. (Caveat: search-based, not exhaustive
      citation-graph traversal.) Confirms the core "first conformal concept-placement" claim.
- [x] medical-NER conformal IDs (2026-06-25): conformal-for-medical-NER EXISTS and is REAL but is a
      DIFFERENT task (span/sequence labeling + entity extraction, NOT placement): arXiv 2601.16999
      (full/subsequence conformal NER, Jan 2026), 2603.00924 (risk-controlled medical entity extraction,
      Mar 2026). Adjacent (medical + conformal) but non-overlapping => cite as related, not competing.
- [ ] Read Bono et al. OAEI/OM 2025 — confirm it's entity LINKING not PLACEMENT.
- [x] Verify arXiv IDs for hierarchical-conformal papers (2501.19038, 2508.13288) — BOTH REAL,
      see prior-art finding 2026-06-25 below. medical-NER conformal / PASC IDs — still unchecked.

### 2026-06-25: PRIOR-ART CHECK — hierarchical-conformal exists; conformal taxonomy-EXPANSION does NOT.
Lit check (user-requested, gates the "clean-benchmark method novelty" question).
- 2508.13288 "Hierarchical Conformal Classification" (Aug 2025): produces prediction sets with
  "nodes at DIFFERENT LEVELS of the hierarchy", backs off to coarser/ancestor labels per-instance
  by confidence, keeps finite-sample coverage. => This IS granularity-adaptive conformal. The
  GENERIC mechanism I was about to claim as novel is ALREADY PUBLISHED. BUT: it is FLAT CLASSIFICATION
  over a FIXED, COMPLETE label taxonomy. Does NOT touch taxonomy expansion / placement / link pred.
- 2501.19038 "CP in Hierarchical Classification w/ Constrained Representation Complexity" (Mortier,
  Huellermeier et al.): also flat hierarchical classification. Same setting.
- Searched conformal x {taxonomy completion / node insertion / concept placement / hypernym}: NOTHING.
  Closest = ICON (taxonomy completion, NOT conformal); conformal taxonomic VALIDATION = species
  classification; conformal graph node classification = fixed transductive nodes. None = our task.
=> DELTA (what is actually open / defensible): conformal/coverage-guaranteed prediction has NEVER been
   applied to concept PLACEMENT / EDGE INSERTION / open-world(NIL) taxonomy expansion. The methodo-
   logical difference from 2508.13288 that we MUST articulate to survive review: our candidate space
   is (a) EDGES not nodes, (b) RETRIEVAL-LIMITED (recall@pool 0.50 -> finest level sometimes
   UNREACHABLE because gold was never retrieved, not just uncertain), (c) OPEN-WORLD (new concept,
   NIL) -- flat-classification conformal assumes the gold label is always in a fixed complete set.
   Our adaptive backoff is FORCED by retrieval failure + degenerate gold, a regime their guarantee
   does not model.
REVIEWER RISK (record): a referee WILL cite 2508.13288 and say "your adaptive granularity = their HCC."
   Preempt by leading with the task/regime difference above; do NOT pitch the abstract mechanism as new.
IMPLICATION for venue: weak as an ML METHOD paper (mechanism taken). Fine/stronger as APPLIED
   biomed-informatics: contribution = first coverage-guaranteed concept-placement/curation tool +
   the capped-exact-accuracy characterization (3 negatives) + cross-dataset (CPP). Cite 2508.13288
   as the classification analog we extend, not duplicate. Clean-benchmark question: conformal-
   classification is taken there too, but conformal taxonomy-EXPANSION is open across BOTH domains.

## Next actions
1. [DONE] Clone backbone -> repo/. [DONE] Download+extract Zenodo data -> data/ (3.5G+5.7G).
2. [DONE] Confirmed: no released scores/candidate dumps in data => scores need the model.
3. [AWAITING USER] Build conda `onto38` (Python 3.8, torch 1.11, transformers 4.29, deeponto
   0.8.8) to run Edge-Bi-encoder on valid-NIL+test-NIL and produce per-edge scores. This yields
   recall@pool and the real coverage-vs-set-size curve. Expect env friction (rotting pinned deps).
4. Then: split-conformal layer over scores -> coverage, set size, risk-coverage/AURC, ECE.
5. Owed verification (brief S8): forward-citation intersection; read Bono et al.; verify arXiv IDs.

## Log
- 2026-06-23: Project approved. Scaffold created. Env verified (conda, 2x RTX 6000 Ada 48GB,
  1.5TB free, GitHub+Zenodo reachable). Reordering decision recorded.
- 2026-06-23: Backbone cloned (Python 3.8, deps rotting). Zenodo OET-data-ver4 (568MB) downloaded
  + extracted. Analytic feasibility probe done: best InR_any@10 ~26% => small-set 90% EXACT-edge
  coverage NOT viable; pivot headline to risk-coverage/abstain + hierarchy-aware ancestor coverage.
  Search space measured: 238k (Disease)/626k (CPP) edges; calib set valid-NIL 328/567 is separate
  from test 275/431 (eases small-N). Next cost = Phase-0 env build. Paused for user go-ahead.
