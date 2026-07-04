# probes/

De-risking checks run before/alongside committing to the real pipeline — not conformal-layer
results (those live under `phase1/` and `phase2/`).

## `recall_probe_Disease.json`

Output of `scripts/recall_probe.py --dataset Disease`, run **before** we committed to
retraining the Oxford Edge-Bi-encoder/Cross-encoder
Purpose: a cheap de-risking check — *is retrieval even feasible
on this benchmark, without training anything?*

### What it measures
For each out-of-KB mention in the `test-NIL` / `valid-NIL` splits of MM-S14-Disease, we embed
the mention and every candidate ontology edge with **zero-shot SapBERT**
(`cambridgeltl/SapBERT-from-PubMedBERT-fulltext` — the *un-fine-tuned* base model the Oxford
bi-encoder starts from) and check whether a gold insertion edge appears in the top-N nearest
edges by cosine similarity, for N in `{10, 50, 100, 200, 500, 1000}`.

Two scoring variants per split:
- **`endpoint_max`** — score(mention, edge) = max(cos(mention, parent-concept), cos(mention,
  child-concept)). Optimistic: lets the mention match either endpoint alone.
- **`edge_text`** — score(mention, edge) = cos(mention, embedding of `"child is a parent"`).
  Closer to how the real bi-encoder scores a full edge.

### Reading the numbers
- `results.<split>:<scoring>.<N>` = recall@N = fraction of mentions whose gold edge is retrieved
  somewhere in the top N candidates.
- Example (`test-NIL`): `edge_text` recall climbs 20.3% (@10) → 22.5% (@50) → 32.6% (@500),
  flattening around 33% by N=1000. `endpoint_max` is worse at low N (3.6% @10) but converges to
  about the same ceiling by N=1000.

### Why this number is a *lower bound*, and what it told us
Zero-shot SapBERT has never seen this task — it's the base model *before* the Oxford pipeline
fine-tunes it on edge-placement supervision. So these recall numbers are a conservative floor on
what the trained bi-encoder can do, not a claim about the trained model's actual recall.

The ~33% zero-shot ceiling (even letting the retrieval pool grow to N=1000) told us retrieval was
*learnable but not free* on this benchmark — reasonable enough to justify running the real
training pipeline, but not so easy that fine-tuning was pointless. That's why we went on to
actually retrain the Edge-Bi-encoder/Cross-encoder (see root `README.md`), which reaches
**49.6% recall@50** — well above this zero-shot floor at the same N, confirming the fine-tuning
step is doing real work.

No CPP counterpart was generated for this particular probe; the CPP leg went straight to the full
retrained pipeline (`phase2/precompute_wp_cpp.py`, `phase2/granularity_adaptive_cpp.py`).
