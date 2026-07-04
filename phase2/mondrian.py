#!/usr/bin/env python
"""
Phase 2 (brief core novelty): CLASS-CONDITIONAL / MONDRIAN conformal for region-set placement.

Marginal split-conformal gives coverage averaged over all mentions; fig4 showed it UNDERSHOOTS on
test (valid->test drift). Mondrian conformal calibrates a SEPARATE threshold per subgroup g(x), where
g is observable at test time, giving per-group coverage. We use curator-relevant, prediction-defined
groups: (1) leaf vs non-leaf (top-1 predicted child == NULL), (2) ontology-depth band of the top-1
predicted parent. We report, at a fixed quality w and target 1-alpha: per-group coverage under MARGINAL
vs MONDRIAN calibration, plus overall. Honest test: if the drift is group-COMPOSITION, Mondrian fixes
per-group coverage; if it is WITHIN-group shift (valid uniformly easier), Mondrian only partly helps --
we report which.
"""
import os, sys, re, json, time, numpy as np
sys.path.insert(0, "/home/kkolpetinou/calibrated-concept-placement/repo")
os.environ.setdefault("DEEPONTO_JVM_MEM", "8g")
from preprocessing.onto_snomed_owl_util import (
    load_SNOMEDCT_deeponto, extract_SNOMEDCT_deeponto_taxonomy, get_shortest_node_depth,
    get_iri_from_SCTID_id)

P = "/home/kkolpetinou/calibrated-concept-placement/phase2"
DUMPS = "/home/kkolpetinou/calibrated-concept-placement/phase1/dumps"
ONTO_FN = "/home/kkolpetinou/calibrated-concept-placement/data/MM-S14-Disease/ontology/SNOMEDCT-US-20140901-Disease-final.owl"
import os as _os
W = float(_os.environ.get("CCP_W", "0.7"))        # quality level
TARGET = float(_os.environ.get("CCP_TARGET", "0.80"))  # target coverage (non-saturated regime)

def load(s):
    d = np.load(os.path.join(P, "wp_cache_%s.npz" % s)); return d["scores"], np.maximum(d["wp"], d["exact"])
scal, wcal = load("valid"); stest, wtest = load("test")

# top-1 predicted edge per mention (by score) -> parent SCTID + leaf flag, from the edge dumps
def top1_meta(split):
    rows = json.load(open(os.path.join(DUMPS, "mm-%s-NIL_edges.json" % split)))
    parents, leaf = [], []
    for r in rows:
        j = int(np.argmax(r["scores"])); pe = r["pred_edges"][j]
        parents.append(pe[0]); leaf.append(pe[1] == "SCTID_NULL")
    return parents, np.array(leaf)
pc_cal, leaf_cal = top1_meta("valid"); pc_test, leaf_test = top1_meta("test")

print("loading taxonomy for depths...", flush=True); t0 = time.time()
taxo = extract_SNOMEDCT_deeponto_taxonomy(load_SNOMEDCT_deeponto(ONTO_FN))
print("  %.0fs" % (time.time()-t0), flush=True)
_sndc = {}
_SCTID = re.compile(r"<(?:http://snomed\.info/id/)?(\d+)>")
def depth(tok):
    ids = ([tok] if tok.isdigit() else _SCTID.findall(tok))
    if not ids: return -1
    best = 999
    for i in ids:
        try:
            d, _ = get_shortest_node_depth(taxo, get_iri_from_SCTID_id(i), dict_iri_to_snd=_sndc); best = min(best, int(d))
        except Exception: pass
    return best if best < 999 else -1
dep_cal = np.array([depth(p) for p in pc_cal]); dep_test = np.array([depth(p) for p in pc_test])

# depth tertiles from valid (resolved depths only)
qs = np.quantile(dep_cal[dep_cal >= 0], [1/3, 2/3])
def depth_band(d): return 0 if d < 0 else (1 if d <= qs[0] else (2 if d <= qs[1] else 3))
db_cal = np.array([depth_band(d) for d in dep_cal]); db_test = np.array([depth_band(d) for d in dep_test])

def thr(E, alpha):
    n=len(E); k=int(np.floor(alpha*(n+1)))
    if k<=0: return -np.inf
    if k>n: return np.inf
    return float(np.sort(E)[k-1])
def E_at(s,w,wl):
    e=np.where(w>=wl, s, -1.0).max(1); e[e<0]=0.; return e
Ec, Et = E_at(scal,wcal,W), E_at(stest,wtest,W)
covered_if = lambda mask_test, t: (((stest>=t)&(wtest>=W)).sum(1)>0)[mask_test]
setsize_if = lambda mask_test, t: (stest>=t).sum(1)[mask_test]

def run_grouping(name, gcal, gtest, labels):
    alpha = 1-TARGET
    tau_marg = thr(Ec, alpha)
    rows=[]
    for gid, lab in labels.items():
        mc, mt = (gcal==gid), (gtest==gid)
        if mt.sum()==0: continue
        cov_marg = covered_if(mt, tau_marg).mean()
        # mondrian: calibrate within group on valid
        if mc.sum()>=10:
            tau_g = thr(E_at(scal[mc],wcal[mc],W), alpha)
        else:
            tau_g = tau_marg
        cov_mond = covered_if(mt, tau_g).mean()
        rows.append({"group":lab,"n_cal":int(mc.sum()),"n_test":int(mt.sum()),
                     "cov_marginal":round(float(cov_marg),3),"cov_mondrian":round(float(cov_mond),3),
                     "set_marg":round(float(setsize_if(mt,tau_marg).mean()),1),
                     "set_mond":round(float(setsize_if(mt,tau_g).mean()),1)})
    nT_tot = sum(r["n_test"] for r in rows)
    pooled_marg = sum(r["cov_marginal"]*r["n_test"] for r in rows)/nT_tot
    pooled_mond = sum(r["cov_mondrian"]*r["n_test"] for r in rows)/nT_tot
    set_marg_tot = sum(r["set_marg"]*r["n_test"] for r in rows)/nT_tot
    set_mond_tot = sum(r["set_mond"]*r["n_test"] for r in rows)/nT_tot
    worst_marg = min(r["cov_marginal"] for r in rows); worst_mond = min(r["cov_mondrian"] for r in rows)
    print("\n## grouping: %s  (W>=%.1f, target=%.2f)"%(name, W, TARGET))
    print("  group                 nC  nT  cov_marg cov_mond  set_marg set_mond")
    for r in rows:
        print("  %-20s %3d %3d   %.3f    %.3f     %5.1f   %5.1f"%(
            r["group"],r["n_cal"],r["n_test"],r["cov_marginal"],r["cov_mondrian"],r["set_marg"],r["set_mond"]))
    print("  -> POOLED cov: marginal %.3f / mondrian %.3f (target %.2f) | WORST-group: marg %.3f / mond %.3f"
          " | mean set: marg %.1f / mond %.1f"%(pooled_marg,pooled_mond,TARGET,worst_marg,worst_mond,set_marg_tot,set_mond_tot))
    return rows

out = {"W":W,"target":TARGET}
out["leaf"] = run_grouping("leaf vs non-leaf",
    np.where(leaf_cal,1,0), np.where(leaf_test,1,0), {1:"leaf (pred child=NULL)",0:"non-leaf"})
out["depth"] = run_grouping("predicted-parent depth band",
    db_cal, db_test, {1:"shallow",2:"mid",3:"deep",0:"unresolved"})
json.dump(out, open(os.path.join(P,"mondrian_results.json"),"w"), indent=2)
print("\nwritten: phase2/mondrian_results.json")

if _os.environ.get("CCP_FIG"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = out["depth"]; g=[r["group"] for r in rows]; x=np.arange(len(g)); wbar=0.38
    plt.figure(figsize=(5.6,4))
    plt.bar(x-wbar/2,[r["cov_marginal"] for r in rows],wbar,label="marginal conformal",color="#bbbbbb")
    plt.bar(x+wbar/2,[r["cov_mondrian"] for r in rows],wbar,label="Mondrian (per-group)",color="#1f77b4")
    plt.axhline(TARGET,ls="--",color="k",lw=1,label="target %.2f"%TARGET)
    plt.xticks(x,g); plt.ylim(0,1.05); plt.ylabel("empirical coverage (test)")
    plt.title("Marginal hides depth-band gaps; Mondrian rebalances\n(W≥%.1f) — but global drift caps both"%W)
    plt.legend(fontsize=8); plt.grid(axis="y",alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(P,"figures","fig5_mondrian_depth.png"),dpi=150); plt.close()
    print("written: phase2/figures/fig5_mondrian_depth.png")
