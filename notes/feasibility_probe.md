# Feasibility probe (analytic, week 0) — 2026-06-23

Goal: before building the Python-3.8 model env, decide on paper whether "90% marginal coverage
at a usable (small) set size" is achievable with this backbone. Source: published InR tables,
Oxford LM-placement paper (arXiv:2402.17897v2, ar5iv).

## Published ceiling (best method = Edge-Cross-encoder, candidate pool = top-50 edge search)
Two numbers per cell = the paper's two settings (atomic/complex or directed/relaxed); magnitude
is what matters here, not the exact value.

MM-S14-Disease, Cross-encoder, InR_any@k:
  @1  = 7.3% / 7.6%
  @5  = 17.9% / 15.6%
  @10 = 25.8% / 26.5%
MM-S14-CPP, Cross-encoder, InR_any@10 = 24.8% / 26.6%

InR_all@k (ALL gold edges in set) is far lower (@10 ~ 9-14%).
Bi-encoder is weaker (@10 ~ 14-20%).

## The key inference: coverage is upper-bounded by retrieval recall
Conformal coverage cannot exceed the rate at which a gold edge is present in the pool the
conformal layer ranks over.
- At @10 the best model includes *any* gold edge only ~26% of the time.
- Growth is slow and roughly linear in the low range (+~1.6 pts / rank between @5 and @10).
- Naive extrapolation to the full top-50 pool lands ~well below 90% (order ~50-60%, unverified).

=> **Exact-edge 90% marginal coverage at small set size is almost certainly NOT achievable**
   with this backbone. Standard split-conformal would either (a) be infeasible (gold not in pool)
   or (b) demand sets ~= the whole pool. The brief's Phase-1 NO-GO is the expected outcome, as
   suspected. We've established this in week 0 on paper, as intended.

## This REFRAMES, it does not kill the project
A 26%-correct top-10 is exactly why a single guess is untrustworthy — that is the motivation,
not the refutation. The viable, honest contributions become:
1. **Risk-coverage / abstain headline**: method auto-places the confident slice (~the 25-30% it
   can stand behind) and refers the rest to curators. AURC is the headline metric. Large sets
   are fine because the value is *referral*, not small sets.
2. **Hierarchy-aware ancestor-collapse becomes ESSENTIAL, not optional**: exact-edge coverage is
   unreachable, but *ancestor-level* (lenient) coverage may be — guarantee the right region of
   the ontology even when the exact insertion edge is uncertain. This is the brief's core novelty
   and the Oxford authors' requested "lenient evaluation."
3. Report coverage-vs-set-size and pick alpha honestly (likely target 70-80% exact, higher for
   lenient), rather than promising 90% exact small sets.

## The ONE measurement that now gates everything
Candidate-retrieval **recall@(pool size)** = fraction of test mentions whose gold edge is anywhere
in the retrieved candidate pool. This is the hard ceiling on any conformal coverage.
- If recall@pool is high (>=90%) but ranking is bad -> conformal sets large but coverage reachable
  -> risk-coverage story strong.
- If recall@pool itself is low (<~70%) -> even abstain framing is capped; ancestor-collapse and/or
  enlarging the retrieval pool become mandatory.
This number is NOT in the paper tables; getting it requires running the bi-encoder edge search
(Phase 0) OR finding retrieved-candidate dumps in the Zenodo zip. <-- next action.

## Measured search-space size (from extracted Zenodo data, 2026-06-23)
| | Disease | CPP |
|---|---|---|
| candidate EDGE catalogue (all, atomic+complex) | **237,825** | **625,993** |
| edge catalogue (atomic only) | 232,828 | - |
| concepts (entity catalogue) | 64,899 | 175,894 |
| test-NIL mentions (test set) | 275 | 431 |
| test-NIL mention-edge pairs | 964 | - |
| valid-NIL mentions (**= calibration set**) | 328 | 567 |
| valid-NIL pairs | 671 | - |
| avg gold edges / mention (test) | ~3.5 (964/275) | - |

Implications:
- **Good news on calibration budget**: valid-NIL is a SEPARATE pool (328 Disease / 567 CPP), so we
  do NOT have to carve the 275/431 test set into calib+test. Split-conformal calibration n~328/567
  is workable (Mondrian per-subgroup will still be thin — the real squeeze is there, not marginal).
- **Confirms huge candidate space**: ~238k / ~626k edges. Conformalizing over the full catalogue
  guarantees the gold edge is *present* (recall=100% by construction) but pushes set sizes large,
  because the model ranks the gold edge in top-10 only ~26% of the time. The deliverable is the
  honest coverage-vs-set-size / risk-coverage curve over this space.
- ~3.5 gold edges/mention => InR_any (set contains >=1 gold) is the natural conformal target;
  InR_all is much harder.

## What still requires the model (no shortcut in released data)
The Zenodo zip has gold pairs + ontology + edge catalogue, but NO candidate dumps or scores.
The actual coverage/set-size curve needs per-mention edge scores over the catalogue (or top-N
retrieval), i.e., run the Edge-Bi-encoder (and ideally Cross-encoder) on valid-NIL + test-NIL.
=> Phase-0 env build (conda onto38, Python 3.8) is the real next cost.

## Decision
- GO/NO-GO #0 (small-set 90% exact): effectively NO -> drop that as the headline now, pre-build.
- Project proceeds under the risk-coverage + hierarchy-aware framing (already the brief's strength).
- Next: inspect Zenodo zip for candidate dumps; if absent, Phase-0 env build to compute recall@pool.
