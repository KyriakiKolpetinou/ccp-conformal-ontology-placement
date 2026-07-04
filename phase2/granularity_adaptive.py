#!/usr/bin/env python
"""
Phase 2 -- GRANULARITY-ADAPTIVE CONFORMAL (the unified mechanism / core methodological novelty).

Idea: standard conformal fixes ONE label granularity. Here the OUTPUT RESOLUTION is per-mention
adaptive: emit the FINEST ontological level (exact edge -> fine region -> region -> coarse region)
whose conformal set fits a curator size-budget B, and certify "correct at the emitted resolution".
Fine where the model is confident, coarse where it is not -- one coverage statement over a
heterogeneous-resolution output.

Granularity ladder via full-edge Wu-Palmer-to-gold (from phase2/wp_cache_*.npz, no JVM):
    exact  w=1.0  | fine  w=0.9 | region w=0.8 | coarse w=0.7
For each level w: split-conformal threshold tau_w on calib (valid-NIL) so the set {score>=tau_w}
contains a WP>=w edge with prob >= 1-alpha (marginal). Coarser w => easier => higher tau => SMALLER
set. Adaptive rule: per mention pick the finest w whose set size <= B; emit that set + its level.

Reports, at matched target coverage & budget, vs two fixed-granularity baselines:
  fixed-EXACT  (w=1.0 for all)  -> sets explode (the Phase-1 failure)
  fixed-COARSE (w=0.7 for all)  -> small sets but everyone gets a coarse answer
  ADAPTIVE                      -> small sets AND a fine answer for the easy fraction
Headline = mean achieved RESOLUTION (mean emitted WP-level) at matched coverage+budget.
"""
import os, json, numpy as np
P = "/home/kkolpetinou/calibrated-concept-placement/phase2"
def load(s):
    d = np.load(os.path.join(P, "wp_cache_%s.npz" % s)); return d["scores"], d["wp"], d["exact"], d["has_gold"]
scal, wcal, ecal, hcal = load("valid")
stest, wtest, etest, htest = load("test")
# exact gold edge is perfectly correct (repo NULL convention scores exact leaf <1) -> force WP=1.
wcal = np.maximum(wcal, ecal); wtest = np.maximum(wtest, etest)
hcal = hcal.astype(bool); htest = htest.astype(bool)
LEVELS = [("exact", 1.0), ("fine", 0.9), ("region", 0.8), ("coarse", 0.7)]
B = 10                       # curator size budget (edges shown)

def conformal_threshold(E, alpha):
    n = len(E); k = int(np.floor(alpha * (n + 1)))
    if k <= 0: return -np.inf
    if k > n: return np.inf
    return float(np.sort(E)[k - 1])

def best_at(s, w, wl):       # per mention: best score among candidates with WP>=wl (-1 if none)
    return np.where(w >= wl, s, -1.0).max(1)

def taus(alpha):             # calibrate one threshold per level at this alpha
    return {nm: conformal_threshold(best_at(scal, wcal, wl), alpha) for nm, wl in LEVELS}

def sets_at(tau, s):         # boolean membership [n,50] for threshold tau
    return s >= tau

def covered_at(inset, w, wl):  # set contains a WP>=wl edge
    return ((inset) & (w >= wl)).sum(1) > 0

def run(target):
    alpha = 1 - target; T = taus(alpha)
    n = len(stest)
    wl = dict(LEVELS)
    # ---- fixed baselines ----
    def fixed(name):
        inset = sets_at(T[name], stest); sz = inset.sum(1)
        cov = covered_at(inset, wtest, wl[name])
        return dict(cov=float(cov.mean()), size=float(sz.mean()), med=float(np.median(sz)),
                    full=float((sz >= 50).mean()), res=wl[name])
    fx_exact = fixed("exact"); fx_coarse = fixed("coarse")
    # ---- adaptive: finest level whose set fits budget B ----
    insets = {nm: sets_at(T[nm], stest) for nm, _ in LEVELS}
    sizes  = {nm: insets[nm].sum(1) for nm, _ in LEVELS}
    emit_lvl = np.empty(n, object); emit_w = np.zeros(n); emit_size = np.zeros(n); cov = np.zeros(n, bool)
    for i in range(n):
        chosen = LEVELS[-1][0]                       # default coarsest
        for nm, _ in LEVELS:                          # finest first
            if sizes[nm][i] <= B:
                chosen = nm; break
        emit_lvl[i] = chosen; emit_w[i] = wl[chosen]; emit_size[i] = sizes[chosen][i]
        cov[i] = covered_at(insets[chosen][i:i+1], wtest[i:i+1], wl[chosen])[0]
    dist = {nm: float((emit_lvl == nm).mean()) for nm, _ in LEVELS}
    adaptive = dict(cov=float(cov.mean()), cov_hasgold=float(cov[htest].mean()),
                    size=float(emit_size.mean()), med=float(np.median(emit_size)),
                    mean_resolution=float(emit_w.mean()), dist=dist)
    return dict(target=target, taus=T, fixed_exact=fx_exact, fixed_coarse=fx_coarse, adaptive=adaptive)

# ---- RESOLUTION-PRECISION TRIAGE FRONTIER (the curator-usability headline) ----
# For each resolution w: rank mentions by confidence (top-1 score), accept the most-confident
# prefix; "correct@w" = top-1 edge has WP>=w to gold. Report the LARGEST auto-place fraction phi
# whose running precision stays >= p. This is selective placement at each ontological resolution.
def degen_mask(split):
    j = json.load(open("/home/kkolpetinou/calibrated-concept-placement/phase1/dumps/mm-%s-NIL_edges.json" % split))
    return np.array([any(str(e[0]) == "64572001" for e in m["gold_edges"]) for m in j])
DEG = degen_mask("test"); WF = ~DEG

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
FRONT = {"all": frontier(np.ones(len(stest), bool)), "well_formed": frontier(WF), "degenerate": frontier(DEG)}

OUT = {"by_target": [run(t) for t in [0.70, 0.80, 0.90]], "triage_frontier": FRONT,
       "split": {"well_formed": int(WF.sum()), "degenerate": int(DEG.sum())}}
json.dump(OUT, open(os.path.join(P, "granularity_adaptive_results.json"), "w"), indent=2)
OUT = OUT["by_target"]

print("="*82)
print("GRANULARITY-ADAPTIVE CONFORMAL  (MM-S14-Disease, calib=valid-NIL n=%d, test=test-NIL n=%d)"
      % (len(scal), len(stest)))
print("budget B=%d edges | resolution = WP-level emitted (1.0 exact ... 0.7 coarse region)" % B)
print("="*82)
for r in OUT:
    print("\nTARGET COVERAGE %.0f%%" % (100*r["target"]))
    print("  %-16s cov    mean|med size   %%full   resolution" % "")
    fe, fc, ad = r["fixed_exact"], r["fixed_coarse"], r["adaptive"]
    print("  fixed-EXACT      %.3f   %5.1f|%2d      %.2f    %.2f" % (fe["cov"], fe["size"], int(fe["med"]), fe["full"], fe["res"]))
    print("  fixed-COARSE     %.3f   %5.1f|%2d      %.2f    %.2f" % (fc["cov"], fc["size"], int(fc["med"]), fc["full"], fc["res"]))
    print("  ADAPTIVE         %.3f   %5.1f|%2d       --     %.2f   (mean achieved resolution)"
          % (ad["cov"], ad["size"], int(ad["med"]), ad["mean_resolution"]))
    print("    adaptive coverage on has-gold subset = %.3f" % ad["cov_hasgold"])
    print("    emitted-resolution mix:", {k: round(v,2) for k,v in ad["dist"].items()})
print("\n" + "="*82)
print("READ: adaptive matches fixed-COARSE coverage at a similar small set, but RETURNS A FINER")
print("answer (higher mean resolution) by giving the confident fraction an exact/near-exact edge")
print("set -- while fixed-EXACT either explodes (%full) or under-covers. Coverage on non-retrieved")
print("gold is still capped (has-gold subset) -> the ancestor-LCA rung (backoff_probe) is the")
print("coarsest fallback that lifts THAT, integrated next.")
print("\n" + "="*82)
print("RESOLUTION-PRECISION TRIAGE FRONTIER  (max auto-place fraction at precision>=p)")
print("="*82)
for name in ["all", "well_formed", "degenerate"]:
    f = FRONT[name]
    print("\n%s (n=%d):   exact(1.0)  fine(0.9)  region(0.8)  coarse(0.7)" % (name.upper(), f["n"]))
    for p in ["p>=0.90", "p>=0.80", "p>=0.70"]:
        r = f["frontier"][p]
        print("  %-8s        %.2f        %.2f       %.2f         %.2f"
              % (p, r["w=1.0"], r["w=0.9"], r["w=0.8"], r["w=0.7"]))
print("\nHEADLINE: exact-edge auto-placement = 0 everywhere (info-capped, as the 3 negatives showed),")
print("but WELL-FORMED mentions auto-place at FINE (WP>=0.9) resolution at high precision, and the")
print("mixed stream yields a usable region/coarse triage tier + a referral tier. CAVEAT: well-formed")
print("n=100 is low concept-diversity (CKD x20) -> CPP + cross-concept eval needed to de-optimise.")
print("written: phase2/granularity_adaptive_results.json")
