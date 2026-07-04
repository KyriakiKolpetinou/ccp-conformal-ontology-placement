#!/usr/bin/env python
"""
Phase 1b -- hierarchy-aware LENIENT CEILING diagnostic.

The decisive go/no-go for the ancestor-aware pivot: when the model misses the EXACT gold edge
(~50% of mentions), are its predictions ontologically NEAR the gold (high Wu-Palmer over the
SNOMED taxonomy) or genuinely far? If near, lenient/ancestor coverage breaks the 49.6% exact
ceiling and rescues the confident-but-"wrong" predictions.

Per mention we compare each top-k predicted edge to each gold edge by PARENT Wu-Palmer similarity
(parent placement is the crux); WP = 2*depth(LCA)/(depth(c1)+depth(c2)) over the taxonomy.
Concepts: plain SCTID (taxonomy node), SCTID_NULL (leaf), or complex [EX.] role-group expr
(we extract the atomic SCTIDs inside and take the best WP). Reports exact vs lenient coverage.
"""
import os, sys, re, json, time
sys.path.insert(0, "/home/kkolpetinou/calibrated-concept-placement/repo")
os.environ.setdefault("DEEPONTO_JVM_MEM", "8g")
import numpy as np
from preprocessing.onto_snomed_owl_util import (
    load_SNOMEDCT_deeponto, extract_SNOMEDCT_deeponto_taxonomy, calculate_wu_palmer_sim,
    get_iri_from_SCTID_id)

ONTO_FN = "/home/kkolpetinou/calibrated-concept-placement/data/MM-S14-Disease/ontology/SNOMEDCT-US-20140901-Disease-final.owl"
DUMPS = "/home/kkolpetinou/calibrated-concept-placement/phase1/dumps"
TOPK = 10   # operationally relevant predicted set for the lenient ceiling

t0 = time.time()
print("loading ontology + taxonomy (deeponto, struct reasoner)...", flush=True)
onto = load_SNOMEDCT_deeponto(ONTO_FN)
taxo = extract_SNOMEDCT_deeponto_taxonomy(onto)
print("  loaded in %.0fs" % (time.time() - t0), flush=True)

_snd, _lca = {}, {}                       # caches for shortest-node-depth and lca pairs
_SCTID = re.compile(r"<(?:http://snomed\.info/id/)?(\d+)>")

def atomic_ids(tok):
    """Return list of atomic SCTIDs in a concept token (plain, or extracted from a complex expr)."""
    if tok == "SCTID_NULL":
        return []
    if tok.isdigit():
        return [tok]
    return _SCTID.findall(tok)            # complex [EX.](...) -> inner SCTIDs

def wp(a, b):
    if a == b:
        return 1.0
    try:
        v, _, _ = calculate_wu_palmer_sim(taxo, get_iri_from_SCTID_id(a), get_iri_from_SCTID_id(b),
                                          dict_iri_to_snd=_snd, dict_iri_pair_to_lca=_lca)
        return float(v)
    except Exception:
        return 0.0

def concept_wp(t1, t2, null1=False, null2=False):
    """Best WP between atomic ids of t1,t2; null flags use parent-depth+1 for NULL leaf children."""
    A, B = atomic_ids(t1), atomic_ids(t2)
    if not A or not B:                    # THING/unresolved complex -> conservative 0
        return 0.0
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
            best = max(best, v)
    return best

def full_edge_wp(pe, ge):
    """Canonical edge WP = (parent_WP + child_WP)/2, with NULL-child -> parent(depth+1)."""
    pp, pc = pe; gp, gc = ge
    parent_wp = concept_wp(pp, gp)
    n1 = pc == "SCTID_NULL"; n2 = gc == "SCTID_NULL"
    pc2 = pp if n1 else pc; gc2 = gp if n2 else gc
    child_wp = concept_wp(pc2, gc2, null1=n1, null2=n2)
    return (parent_wp + child_wp) / 2.0

def best_concept_wp(t1, t2):  # parent-only (kept for comparison)
    A, B = atomic_ids(t1), atomic_ids(t2)
    if not A or not B:
        return None
    return max(wp(a, b) for a in A for b in B)

def analyse(split):
    rows = json.load(open(os.path.join(DUMPS, "mm-%s-NIL_edges.json" % split)))
    exact_hit, bw_parent, bw_edge, edge_miss = [], [], [], []
    for r in rows:
        order = np.argsort(-np.array(r["scores"]))[:TOPK]
        preds = [r["pred_edges"][i] for i in order]
        golds = r["gold_edges"]
        gold_set = {tuple(g) for g in golds}
        exact = any(tuple(p) in gold_set for p in preds)
        bp = be = 0.0
        for p in preds:
            for g in golds:
                wpa = best_concept_wp(p[0], g[0])
                if wpa is not None and wpa > bp: bp = wpa
                we = full_edge_wp(p, g)
                if we > be: be = we
        exact_hit.append(exact); bw_parent.append(bp); bw_edge.append(be)
        if not exact: edge_miss.append(be)
    exact_hit = np.array(exact_hit); bp = np.array(bw_parent); be = np.array(bw_edge)
    miss = np.array(edge_miss)
    print("\n===== %s (n=%d, top-%d) =====" % (split, len(rows), TOPK))
    print("  EXACT-edge coverage@%d : %.3f" % (TOPK, exact_hit.mean()))
    print("  thr   parent-WP cov   FULL-EDGE-WP cov")
    for thr in [0.70, 0.80, 0.90, 0.95]:
        print("  %.2f      %.3f            %.3f" % (thr, (bp >= thr).mean(), (be >= thr).mean()))
    print("  best FULL-EDGE-WP overall: mean %.3f median %.3f | parent-only mean %.3f" %
          (be.mean(), np.median(be), bp.mean()))
    if len(miss):
        print("  AMONG EXACT MISSES (n=%d): full-edge-WP mean %.3f median %.3f | %%>=0.8 %.3f | %%>=0.9 %.3f"
              % (len(miss), miss.mean(), np.median(miss), (miss >= 0.8).mean(), (miss >= 0.9).mean()))
    return {"split": split, "n": len(rows), "topk": TOPK, "exact_cov": float(exact_hit.mean()),
            "parent_wp_cov": {str(t): float((bp >= t).mean()) for t in [0.7,0.8,0.9,0.95]},
            "full_edge_wp_cov": {str(t): float((be >= t).mean()) for t in [0.7,0.8,0.9,0.95]},
            "full_edge_wp_mean": float(be.mean()),
            "miss_full_edge_wp_mean": float(miss.mean()) if len(miss) else None,
            "miss_full_edge_ge_0.8": float((miss >= 0.8).mean()) if len(miss) else None,
            "miss_full_edge_ge_0.9": float((miss >= 0.9).mean()) if len(miss) else None}

out = [analyse("test"), analyse("valid")]
json.dump(out, open("/home/kkolpetinou/calibrated-concept-placement/phase1/lenient_ceiling.json", "w"), indent=2)
print("\nwritten: phase1/lenient_ceiling.json   (total %.0fs)" % (time.time() - t0))
