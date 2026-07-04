#!/usr/bin/env python
"""
Phase 2 foundation: precompute, for every (mention, candidate-edge) pair, the FULL-EDGE
Wu-Palmer similarity of that candidate to the mention's BEST-matching gold edge.

Output per split -> phase2/wp_cache_<split>.npz with arrays [n_mentions, 50]:
  scores   : cross-encoder sigmoid score per candidate slot (from the edge dump)
  wp       : best full-edge WP of that candidate to any gold edge (0..1)
  exact    : 1.0 if the candidate edge exactly equals a gold edge
  has_gold : per-mention [n] 1.0 if the mention has >=1 gold edge (always 1 here)

This is the expensive (ontology) step; cache it so the conformal/risk-coverage analysis
(phase2/conformal_wp.py) iterates instantly. WP convention matches the repo: edge WP =
(parent_WP + child_WP)/2, NULL child -> parent (depth+1), complex -> best over atomic ids.
"""
import os, sys, re, json, time
sys.path.insert(0, "/home/kkolpetinou/calibrated-concept-placement/repo")
os.environ.setdefault("DEEPONTO_JVM_MEM", "8g")
import numpy as np
from preprocessing.onto_snomed_owl_util import (
    load_SNOMEDCT_deeponto, extract_SNOMEDCT_deeponto_taxonomy, calculate_wu_palmer_sim,
    get_iri_from_SCTID_id)

ONTO_FN = "/home/kkolpetinou/calibrated-concept-placement/data/MM-S14-CPP/ontology/SNOMEDCT-US-20140901-CPP-final.owl"
DUMPS = "/home/kkolpetinou/calibrated-concept-placement/phase1/dumps_cpp"
OUT = "/home/kkolpetinou/calibrated-concept-placement/phase2"

t0 = time.time()
print("loading ontology...", flush=True)
onto = load_SNOMEDCT_deeponto(ONTO_FN)
taxo = extract_SNOMEDCT_deeponto_taxonomy(onto)
print("  loaded in %.0fs" % (time.time() - t0), flush=True)

_snd, _lca = {}, {}
_SCTID = re.compile(r"<(?:http://snomed\.info/id/)?(\d+)>")
def atomic_ids(tok):
    if tok == "SCTID_NULL" or tok.endswith("_THING"): return []
    if tok.isdigit(): return [tok]
    return _SCTID.findall(tok)

def concept_wp(t1, t2, null1=False, null2=False):
    A, B = atomic_ids(t1), atomic_ids(t2)
    if not A or not B: return 0.0
    best = 0.0
    for a in A:
        for b in B:
            if a == b and not (null1 or null2):
                v = 1.0
            else:
                try:
                    v, _, _ = calculate_wu_palmer_sim(
                        taxo, get_iri_from_SCTID_id(a), get_iri_from_SCTID_id(b),
                        get_NULL_node_depth_iri1=null1, get_NULL_node_depth_iri2=null2,
                        dict_iri_to_snd=_snd, dict_iri_pair_to_lca=_lca)
                    v = float(v)
                except Exception:
                    v = 0.0
            if v > best: best = v
    return best

def edge_wp(pe, ge):
    pp, pc = pe; gp, gc = ge
    parent = concept_wp(pp, gp)
    n1 = pc == "SCTID_NULL"; n2 = gc == "SCTID_NULL"
    child = concept_wp(pp if n1 else pc, gp if n2 else gc, null1=n1, null2=n2)
    return (parent + child) / 2.0

for split in ["valid", "test"]:
    rows = json.load(open(os.path.join(DUMPS, "mm-%s-NIL_edges.json" % split)))
    n = len(rows); L = len(rows[0]["scores"])
    scores = np.zeros((n, L), np.float32); wp = np.zeros((n, L), np.float32)
    exact = np.zeros((n, L), np.float32); has_gold = np.zeros(n, np.float32)
    tcache = {}  # (tuple(pe)) cache within a mention is weak; rely on snd/lca global cache
    for i, r in enumerate(rows):
        golds = r["gold_edges"]; gold_set = {tuple(g) for g in golds}
        has_gold[i] = 1.0 if golds else 0.0
        scores[i] = np.array(r["scores"], np.float32)
        for j, pe in enumerate(r["pred_edges"]):
            exact[i, j] = 1.0 if tuple(pe) in gold_set else 0.0
            best = 0.0
            for g in golds:
                w = edge_wp(pe, g)
                if w > best: best = w
            wp[i, j] = best
        if (i + 1) % 50 == 0:
            print("  %s %d/%d  (%.0fs, |snd|=%d |lca|=%d)" % (split, i+1, n, time.time()-t0, len(_snd), len(_lca)), flush=True)
    np.savez(os.path.join(OUT, "wp_cache_cpp_%s.npz" % split), scores=scores, wp=wp, exact=exact, has_gold=has_gold)
    print("saved wp_cache_%s.npz  wp.mean=%.3f exact.sum=%d" % (split, wp.mean(), int(exact.sum())), flush=True)

print("DONE in %.0fs" % (time.time() - t0))
