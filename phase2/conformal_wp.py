#!/usr/bin/env python
"""
Phase 2 -- HIERARCHY-AWARE conformal layer (continuous Wu-Palmer).

Reads phase2/wp_cache_{valid,test}.npz (scores + per-candidate full-edge WP-to-gold).
valid-NIL = calibration, test-NIL = test.

Produces:
 (A) WP risk-coverage: rank mentions by a confidence signal, auto-accept the top fraction, report
     the MEAN WP-quality of accepted predictions (continuous risk = 1-WP) + AURC. Tests whether the
     abstain story is RESCUED once "near" predictions earn partial credit (Phase-1 binary risk failed).
 (B) Coverage-guaranteed CORRECT-REGION sets: for a target ontological quality w (the set must
     contain an edge with WP>=w to gold) and target coverage 1-alpha, split-conformal calibrates a
     score threshold on valid-NIL; report empirical test coverage + set size. Sweep w to show the
     (quality, coverage, set-size) tradeoff -- the curator-usability headline. w=1.0 == exact (Phase-1
     blow-up) for contrast.
"""
import os, json, numpy as np
P = "/home/kkolpetinou/calibrated-concept-placement/phase2"
def load(s):
    d = np.load(os.path.join(P, "wp_cache_%s.npz" % s)); return d["scores"], d["wp"], d["exact"]
scal, wcal, ecal = load("valid")
stest, wtest, etest = load("test")
# An edge identical to a gold edge is perfectly correct (WP=1.0); the repo's NULL-node convention
# scores exact LEAF edges <1, which undercounts exact matches at high WP. Force exact -> WP=1.0.
wcal = np.maximum(wcal, ecal); wtest = np.maximum(wtest, etest)
res = {"meta": {"n_cal": int(len(scal)), "n_test": int(len(stest))}}

# ---------- (A) continuous-WP risk-coverage (selective prediction) ----------
def risk_coverage(s, w):
    top = s.argmax(1)
    q = w[np.arange(len(s)), top]          # WP-quality of the top-1 predicted edge
    conf = s.max(1)                        # confidence = top-1 score
    order = np.argsort(-conf)              # most confident first
    cum = np.cumsum(q[order]) / np.arange(1, len(q) + 1)
    cov = np.arange(1, len(q) + 1) / len(q)
    aurc = float(np.mean(1 - cum))         # risk = 1 - WP
    pts = {("%.0f%%" % (100*c)): float(cum[int(c*len(q))-1]) for c in [0.1,0.2,0.3,0.5,1.0]}
    return {"mean_WP_top1": float(q.mean()), "AURC_riskcov": aurc, "selective_meanWP_at": pts}
res["risk_coverage_test"] = risk_coverage(stest, wtest)

# ---------- (B) coverage-guaranteed correct-region sets ----------
def conformal_threshold(E, alpha):
    n = len(E); k = int(np.floor(alpha * (n + 1)))
    if k <= 0: return -np.inf
    if k > n: return np.inf
    return float(np.sort(E)[k - 1])

def best_score_at_quality(s, w, wlevel):
    """per mention, highest score among candidates with WP>=wlevel (0 if none)."""
    masked = np.where(w >= wlevel, s, -1.0)
    e = masked.max(1); e[e < 0] = 0.0; return e

def region_sets(wlevel, target):
    alpha = 1 - target
    Ecal = best_score_at_quality(scal, wcal, wlevel)
    tau = conformal_threshold(Ecal, alpha)
    inset = stest >= tau
    set_sizes = inset.sum(1)
    covered = ((inset) & (wtest >= wlevel)).sum(1) > 0      # set contains a WP>=wlevel edge
    return {"wlevel": wlevel, "target_cov": target, "tau": float(tau),
            "emp_coverage": float(covered.mean()), "mean_set_size": float(set_sizes.mean()),
            "median_set_size": float(np.median(set_sizes)), "frac_full_pool": float((set_sizes>=stest.shape[1]).mean())}

res["region_sets"] = []
for w in [1.0, 0.9, 0.8, 0.7]:
    for tgt in [0.70, 0.80, 0.90]:
        res["region_sets"].append(region_sets(w, tgt))

# also: the lenient ceiling per quality (oracle, top-all) for reference
res["ceiling_test"] = {str(w): float(((wtest >= w).any(1)).mean()) for w in [1.0,0.95,0.9,0.8,0.7]}

json.dump(res, open(os.path.join(P, "conformal_wp_results.json"), "w"), indent=2)

# ---------- console ----------
print("="*76)
print("PHASE 2 -- HIERARCHY-AWARE CONFORMAL (continuous Wu-Palmer), MM-S14-Disease")
print("="*76)
print("(A) WP RISK-COVERAGE (confidence=top-1 score, quality=full-edge WP of top-1):")
rc = res["risk_coverage_test"]
print("    mean WP@top1 (accept all) = %.3f | AURC(1-WP) = %.3f" % (rc["mean_WP_top1"], rc["AURC_riskcov"]))
for k, v in rc["selective_meanWP_at"].items():
    print("      accept %4s most-confident -> mean WP = %.3f" % (k, v))
print("\n    (Phase-1 binary risk-cov AURC was 0.957; here risk is continuous WP-distance.)")
print("\n(B) COVERAGE-GUARANTEED CORRECT-REGION SETS (calib=valid, eval=test):")
print("    WP>=w  target  emp.cov  mean|med set  %full")
for r in res["region_sets"]:
    print("    %.2f    %.2f     %.3f    %5.1f|%2d     %.2f" % (
        r["wlevel"], r["target_cov"], r["emp_coverage"], r["mean_set_size"],
        int(r["median_set_size"]), r["frac_full_pool"]))
print("\n    oracle ceiling (top-all) coverage by quality:",
      {k: round(v,2) for k,v in res["ceiling_test"].items()})
print("="*76); print("written: phase2/conformal_wp_results.json")
