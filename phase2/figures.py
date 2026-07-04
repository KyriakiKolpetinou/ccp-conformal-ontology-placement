#!/usr/bin/env python
"""
Phase 2 publication figures (reads caches, no ontology recompute). Writes PNGs to phase2/figures/.
 Fig1 risk_coverage.png  : selective-prediction curve, continuous WP quality vs exact correctness.
 Fig2 calibration.png    : reliability diagram (cross-enc score vs empirical exact-rate and mean WP) + ECE.
 Fig3 setsize_coverage.png: coverage-guaranteed region sets -- coverage vs mean set size per WP level.
 Fig4 coverage_calib.png : target vs empirical coverage (shows the valid/test drift undershoot).
Also writes phase2/figures/metrics.json with AURC / ECE / key operating points.
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "/home/kkolpetinou/calibrated-concept-placement/phase2"
FIG = os.path.join(P, "figures"); os.makedirs(FIG, exist_ok=True)
def load(s):
    d = np.load(os.path.join(P, "wp_cache_%s.npz" % s)); return d["scores"], np.maximum(d["wp"], d["exact"]), d["exact"]
scal, wcal, ecal = load("valid"); stest, wtest, etest = load("test")
L = stest.shape[1]; M = {}

# ---------- Fig 1: risk-coverage (continuous WP vs exact) ----------
top = stest.argmax(1); conf = stest.max(1); order = np.argsort(-conf)
q_wp = wtest[np.arange(len(stest)), top][order]
q_ex = etest[np.arange(len(stest)), top][order]
cov = np.arange(1, len(order)+1)/len(order)
cum_wp = np.cumsum(q_wp)/np.arange(1, len(order)+1)
cum_ex = np.cumsum(q_ex)/np.arange(1, len(order)+1)
M["AURC_wp"] = float(np.mean(1-cum_wp)); M["AURC_exact"] = float(np.mean(1-cum_ex))
M["meanWP_all"] = float(q_wp.mean()); M["exact_top1_all"] = float(q_ex.mean())
plt.figure(figsize=(5.2,4))
plt.plot(cov, cum_wp, "-", lw=2, color="#1f77b4", label="region quality (Wu–Palmer)")
plt.plot(cov, cum_ex, "-", lw=2, color="#d62728", label="exact-edge correctness")
plt.xlabel("coverage (fraction auto-accepted, by confidence)"); plt.ylabel("mean placement quality of accepted")
plt.title("Risk–coverage: confidence tracks region, not exact edge"); plt.ylim(0,1); plt.grid(alpha=.3)
plt.legend(loc="center right", fontsize=9); plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig1_risk_coverage.png"), dpi=150); plt.close()

# ---------- Fig 2: calibration / ECE ----------
def reliability(s, lab, nb=10):
    sc=s.ravel(); y=lab.ravel(); edges=np.linspace(0,1,nb+1); xs=[]; ys=[]; ws=[]; ece=0.; tot=len(sc)
    for b in range(nb):
        m=(sc>=edges[b])&((sc<edges[b+1]) if b<nb-1 else (sc<=edges[b+1]))
        if m.sum()==0: continue
        xs.append(sc[m].mean()); ys.append(y[m].mean()); ws.append(m.sum())
        ece+=m.sum()/tot*abs(sc[m].mean()-y[m].mean())
    return np.array(xs), np.array(ys), np.array(ws), float(ece)
xe,ye,we,ece_ex = reliability(stest, etest)        # score vs exact-gold rate
xw,yw,_,_       = reliability(stest, wtest)         # score vs mean WP quality
M["ECE_exact"]=ece_ex
plt.figure(figsize=(5.2,4))
plt.plot([0,1],[0,1],"k--",lw=1,label="perfect calibration")
plt.plot(xe,ye,"o-",color="#d62728",label="emp. exact-gold rate (ECE=%.3f)"%ece_ex)
plt.plot(xw,yw,"s-",color="#1f77b4",label="emp. mean WP quality")
plt.xlabel("cross-encoder score (predicted P(correct))"); plt.ylabel("empirical outcome")
plt.title("Calibration: scores overstate exact, track region"); plt.grid(alpha=.3)
plt.legend(fontsize=8,loc="upper left"); plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig2_calibration.png"),dpi=150); plt.close()

# ---------- conformal helpers ----------
def thr(E, alpha):
    n=len(E); k=int(np.floor(alpha*(n+1)))
    if k<=0: return -np.inf
    if k>n: return np.inf
    return float(np.sort(E)[k-1])
def E_at(s,w,wl):
    e=np.where(w>=wl, s, -1.0).max(1); e[e<0]=0.; return e

# ---------- Fig 3: set-size vs coverage frontier, per WP level ----------
plt.figure(figsize=(5.2,4))
colors={1.0:"#d62728",0.9:"#ff7f0e",0.8:"#2ca02c",0.7:"#1f77b4"}
M["frontier"]={}
for wl in [1.0,0.9,0.8,0.7]:
    Ec=E_at(scal,wcal,wl); sizes=[]; covs=[]
    for tgt in np.linspace(0.50,0.97,24):
        t=thr(Ec,1-tgt); inset=stest>=t
        sizes.append(inset.sum(1).mean()); covs.append((((inset)&(wtest>=wl)).sum(1)>0).mean())
    lab = "exact" if wl==1.0 else ("WP≥%.1f"%wl)
    plt.plot(sizes, covs, "o-", ms=3, color=colors[wl], label=lab)
    M["frontier"][("exact" if wl==1.0 else "WP%.1f"%wl)]={"set_size":[round(x,1) for x in sizes],"coverage":[round(c,3) for c in covs]}
plt.xlabel("mean set size (edges a curator reviews)"); plt.ylabel("empirical coverage (test)")
plt.title("Coverage-guaranteed region sets: quality vs effort"); plt.grid(alpha=.3); plt.legend(title="set must contain",fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig3_setsize_coverage.png"),dpi=150); plt.close()

# ---------- Fig 4: coverage calibration (drift) ----------
plt.figure(figsize=(5.2,4)); plt.plot([0.5,0.97],[0.5,0.97],"k--",lw=1,label="ideal (target=empirical)")
for wl in [0.7,0.8,0.9]:
    Ec=E_at(scal,wcal,wl); tg=np.linspace(0.50,min(0.95,((wtest>=wl).any(1)).mean()),12); emp=[]
    for t in tg:
        th=thr(Ec,1-t); emp.append(((( stest>=th)&(wtest>=wl)).sum(1)>0).mean())
    plt.plot(tg,emp,"o-",ms=3,color={0.7:"#1f77b4",0.8:"#2ca02c",0.9:"#ff7f0e"}[wl],label="WP≥%.1f"%wl)
plt.xlabel("target coverage (1−α)"); plt.ylabel("empirical coverage (test)")
plt.title("Coverage undershoot from valid→test drift (motivates Mondrian)"); plt.grid(alpha=.3); plt.legend(fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig4_coverage_calib.png"),dpi=150); plt.close()

json.dump(M, open(os.path.join(FIG,"metrics.json"),"w"), indent=2)
print("Figures + metrics written to phase2/figures/")
print("  AURC  WP=%.3f  exact=%.3f | meanWP_all=%.3f exact@1_all=%.3f | ECE_exact=%.3f"%(
    M["AURC_wp"],M["AURC_exact"],M["meanWP_all"],M["exact_top1_all"],M["ECE_exact"]))
for f in sorted(os.listdir(FIG)):
    if f.endswith(".png"): print("  -", f)
