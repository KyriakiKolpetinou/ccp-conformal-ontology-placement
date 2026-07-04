#!/usr/bin/env python
"""
Phase 2 novelty probe -- HIERARCHY-AWARE BACK-OFF placement.
When uncertain, instead of a flat 50-edge set, return ONE ancestor concept = LCA of the top-m
predicted parents. Measure the tradeoff:
  coverage   = back-off ancestor subsumes a GOLD parent (we pointed at the right subtree)
  specificity= depth of that ancestor (root~3 = useless; deeper = more useful to a curator)
GO if there is an m where coverage is high (>=~0.8) AND the ancestor is meaningfully specific
(depth well above root, ideally near the gold-parent depth). Sweep m = top-1..top-10 parents.
"""
import sys, json, time, numpy as np
sys.path.insert(0, "/home/kkolpetinou/calibrated-concept-placement/repo")
import os; os.environ.setdefault("DEEPONTO_JVM_MEM", "8g")
from preprocessing.onto_snomed_owl_util import (
    load_SNOMEDCT_deeponto, extract_SNOMEDCT_deeponto_taxonomy, get_shortest_node_depth,
    get_lowest_common_ancestor, get_iri_from_SCTID_id, get_SCTID_id_from_iri)

DUMPS = "/home/kkolpetinou/calibrated-concept-placement/phase1/dumps"
t0 = time.time(); print("loading taxonomy...", flush=True)
taxo = extract_SNOMEDCT_deeponto_taxonomy(load_SNOMEDCT_deeponto(
    "/home/kkolpetinou/calibrated-concept-placement/repo/ontologies/SNOMEDCT-US-20140901-Disease-final.owl"))
print("  %.0fs" % (time.time()-t0), flush=True)

_snd, _lca = {}, {}
def depth(iri):
    try: d,_ = get_shortest_node_depth(taxo, iri, dict_iri_to_snd=_snd); return int(d)
    except: return -1
def lca(a, b):
    if a == b: return a
    key = (a, b) if a < b else (b, a)
    if key in _lca: return _lca[key]
    try: r = get_lowest_common_ancestor(taxo, a, b)
    except: r = None
    _lca[key] = r; return r
def subsumes(anc, g):       # anc is ancestor-or-equal of g  <=>  LCA(anc,g)==anc
    if anc is None or g is None: return False
    if anc == g: return True
    return lca(anc, g) == anc

def parents_iris(edges, scores=None):
    """distinct atomic parent IRIs, ordered by score desc if scores given."""
    idx = np.argsort(-np.array(scores)) if scores is not None else range(len(edges))
    out = []
    for j in idx:
        p = edges[j][0]
        if p.isdigit():
            iri = get_iri_from_SCTID_id(p)
            if iri not in out: out.append(iri)
    return out

def analyse(split):
    rows = json.load(open(os.path.join(DUMPS, "mm-%s-NIL_edges.json" % split)))
    res = {}
    gold_depths = []
    for m in [1, 2, 3, 5, 8, 10]:
        covs, deps = [], []
        for r in rows:
            pars = parents_iris(r["pred_edges"], r["scores"])[:m]
            golds = [get_iri_from_SCTID_id(g[0]) for g in r["gold_edges"] if g[0].isdigit()]
            if not pars or not golds: continue
            anc = pars[0]
            for p in pars[1:]: anc = lca(anc, p)
            covs.append(any(subsumes(anc, g) for g in golds))
            deps.append(depth(anc))
            if m == 1: gold_depths.append(max(depth(g) for g in golds))
        covs = np.array(covs); deps = np.array(deps[: len(covs)])
        res[m] = {"coverage": float(covs.mean()), "mean_depth": float(deps[deps>=0].mean()),
                  "median_depth": float(np.median(deps[deps>=0]))}
    gd = np.array(gold_depths)
    print("\n===== %s (n eval=%d) =====" % (split, len(rows)))
    print("  reference: gold-parent depth mean %.1f median %.0f | root depth ~3" % (gd.mean(), np.median(gd)))
    print("  top-m parents -> LCA back-off ancestor:")
    print("   m   coverage   mean_depth  median_depth")
    for m, v in res.items():
        print("  %2d    %.3f       %.1f         %.0f" % (m, v["coverage"], v["mean_depth"], v["median_depth"]))
    return {"split": split, "gold_parent_depth_mean": float(gd.mean()), "by_m": res}

out = [analyse("test"), analyse("valid")]
json.dump(out, open("/home/kkolpetinou/calibrated-concept-placement/phase2/backoff_probe.json", "w"), indent=2)
print("\nwritten: phase2/backoff_probe.json  (%.0fs)" % (time.time()-t0))
