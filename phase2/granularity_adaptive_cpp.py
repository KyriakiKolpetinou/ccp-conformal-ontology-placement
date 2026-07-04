#!/usr/bin/env python
"""
CROSS-DATASET replication of the granularity-adaptive conformal triage frontier on MM-S14-CPP.
Mirrors granularity_adaptive.py (Disease). Reads phase2/wp_cache_cpp_{valid,test}.npz (built by
precompute_wp_cpp.py) + phase1/dumps_cpp for the leaf/non-leaf split.

CPP note: unlike Disease (one dominant degenerate root "disease(disorder)->NULL" = 64%), CPP gold
parents are spread across branches. So we split by the dataset-agnostic LEAF criterion instead of a
single-root match: a mention is LEAF/degenerate if ALL its gold edges have a NULL child (placed as a
bare leaf, no informative child) -- parallels the paper's own leaf vs non-leaf reporting.
"""
import os, json, numpy as np
P = "/home/kkolpetinou/calibrated-concept-placement/phase2"
D = "/home/kkolpetinou/calibrated-concept-placement/phase1/dumps_cpp"
def load(s):
    d = np.load(os.path.join(P, "wp_cache_cpp_%s.npz" % s)); return d["scores"], np.maximum(d["wp"], d["exact"])
scal, wcal = load("valid"); stest, wtest = load("test")
LEVELS = [("exact", 1.0), ("fine", 0.9), ("region", 0.8), ("coarse", 0.7)]

def leaf_mask(split):
    j = json.load(open(os.path.join(D, "mm-%s-NIL_edges.json" % split)))
    return np.array([all("NULL" in str(e[1]).upper() for e in m["gold_edges"]) if m["gold_edges"]
                     else True for m in j])
LEAF = leaf_mask("test"); NONLEAF = ~LEAF

def frontier(mask):
    ss = stest[mask]; ww = wtest[mask]; n = len(ss)
    top = ss.argmax(1); order = np.argsort(-ss.max(1)); out = {}
    for p in [0.90, 0.80, 0.70]:
        out["p>=%.2f" % p] = {}
        for _, wl in LEVELS:
            q = (ww[np.arange(n), top] >= wl).astype(float)[order]
            cum = np.cumsum(q) / np.arange(1, n + 1)
            ok = np.where(cum >= p)[0]
            out["p>=%.2f" % p]["w=%.1f" % wl] = float((ok[-1] + 1) / n) if len(ok) else 0.0
    return {"n": int(n), "frontier": out}

FRONT = {"all": frontier(np.ones(len(stest), bool)),
         "non_leaf": frontier(NONLEAF), "leaf": frontier(LEAF)}
json.dump(FRONT, open(os.path.join(P, "granularity_adaptive_cpp_results.json"), "w"), indent=2)

print("=" * 82)
print("CPP CROSS-DATASET TRIAGE FRONTIER  (test n=%d | non-leaf=%d leaf=%d)"
      % (len(stest), int(NONLEAF.sum()), int(LEAF.sum())))
print("max auto-place fraction at precision>=p, per resolution")
print("=" * 82)
for name in ["all", "non_leaf", "leaf"]:
    f = FRONT[name]
    print("\n%s (n=%d):   exact(1.0)  fine(0.9)  region(0.8)  coarse(0.7)" % (name.upper(), f["n"]))
    for p in ["p>=0.90", "p>=0.80", "p>=0.70"]:
        r = f["frontier"][p]
        print("  %-8s        %.2f        %.2f       %.2f         %.2f"
              % (p, r["w=1.0"], r["w=0.9"], r["w=0.8"], r["w=0.7"]))
print("\nCompare vs Disease (granularity_adaptive_results.json) to claim cross-dataset generality.")
print("written: phase2/granularity_adaptive_cpp_results.json")
