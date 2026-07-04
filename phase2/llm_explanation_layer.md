# LLM-explanation layer (scoping + worked demo)

**Status:** scoped, not built (needs API). This is the *surviving* LLM angle after the verifier
NO-GO (probe 3). It is a **complement** to the granularity-adaptive conformal core, NOT a standalone
method claim. Maps to the paper's future directions "LLM-generated explanations" + "assist human
terminologists."

## Design principle (why it's safe)
The conformal layer owns **correctness** (the guaranteed region); the LLM owns **readability** only.
The LLM is given the *already-decided* granularity-adaptive output and must explain/justify it — it
**never re-ranks or overrides** the placement. Therefore LLM hallucination cannot break the coverage
guarantee: worst case is a bad explanation, never a wrong placement. This sidesteps exactly the
failure that killed LLM-as-verifier (the model has no signal the mention lacks).

## I/O
- **In:** mention + context; emitted resolution w and its guaranteed region set (candidate edges with
  concept *names* + scores); the tier (auto / region / refer).
- **Out:** (a) one-line placement at the guaranteed resolution; (b) an *ambiguity note* naming the
  indistinguishable sibling children and why the text can't resolve them; (c) the tier flag.

## Evaluation plan
1. **Faithfulness (automatic):** explanation must name only concepts inside the guaranteed region —
   checkable by string-match against the region set (0 hallucinated parents = pass). Cheap, no humans.
2. **Usability (small expert study):** N≈30 cards, terminologist rates "would this speed my decision?"
   1-5. Qualitative; positions the contribution as curator-assist, not accuracy.

## Worked examples (generated WITHOUT an API, grounded in real test cards)

**Card #15 — "chronic kidney disease" (well-formed, AUTO/fine tier, guaranteed WP>=0.9)**
Region set: `renal impairment (disorder) -> {chronic kidney disease due to hypertension, hypertensive
chronic kidney disease}` (top-1 scores 0.50/0.31).
> Places under **renal impairment (disorder)** — high confidence, guaranteed region (WP>=0.9). The
> exact child is **not determined by the text**: the leading candidates are hypertension-specific
> subtypes ("CKD due to hypertension", "hypertensive chronic kidney disease") that the mention
> "chronic kidney disease" does not support. **Recommend:** accept parent = renal impairment; assign
> a generic CKD leaf rather than a hypertensive subtype. *Auto-tier.*

**Card #162 — "triple negative breast cancer" (well-formed, region tier)**
Region set: `{carcinoma of breast, malignant tumor of breast} -> NULL` (near-synonym parents).
> Places in the **breast-carcinoma region** (guaranteed WP>=0.9). Parent is ambiguous between the
> near-synonyms **"carcinoma of breast"** and **"malignant tumor of breast"** — TNBC is correctly
> both; choosing the leaf is a curator convention call, not a text-resolvable distinction. *Region-tier.*

**Card — "acute kidney injury" (degenerate / gold not retrieved, REFER tier)**
> **No confident placement.** The gold for this concept is a generic-root leaf (a version-diff
> artifact) and/or its specific parent was not retrieved. The model's weak suggestions are
> {…top-3…}; **refer to curator.** *Refer-tier.*

## Honest read
This layer makes the tool *usable* and hits two of the paper's stated future directions, but its
novelty is modest (it's a guarded explanation wrapper). It belongs in the paper as a usability
section + small study, NOT as the headline. Headline stays = granularity-adaptive conformal.
