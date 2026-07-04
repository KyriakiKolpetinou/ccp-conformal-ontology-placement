#!/usr/bin/env python
"""
Phase 1 -- split-conformal layer over Edge-Cross-encoder scores for concept placement.

Inputs (from the patched main_dense_plus dump):
  phase1/dumps/mm-valid-NIL.npz  -> calibration  (y, yhat_raw) each [n_cal, 50]
  phase1/dumps/mm-test-NIL.npz   -> test         (y, yhat_raw) each [n_test, 50]
  y[i,j]=1 if candidate edge j is a gold edge for mention i; yhat_raw[i,j]=cross-enc sigmoid prob.

Produces split-conformal "any"-coverage edge SETS, the coverage/set-size tradeoff, the
risk-coverage (selective-prediction) curve + AURC, and ECE. Writes conformal_results.json.

GO/NO-GO #2 read: if achieving useful (>=80-90%) coverage forces sets ~= full pool, the
small-set story is dead and we pivot to risk-coverage/abstain (still the brief's strength).
The hard bound on any-coverage is recall@pool (fraction of mentions with a gold edge retrieved).
"""
import numpy as np, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
def load(split):
    d = np.load(os.path.join(HERE, "dumps", "mm-%s-NIL.npz" % split))
    return d["y"].astype(np.float32), d["yhat_raw"].astype(np.float32)

ycal, scal = load("valid")   # calibration
ytest, stest = load("test")  # test
n_cal, L = ycal.shape

def recall_at_pool(y):  # fraction of mentions with >=1 gold edge in the candidate pool
    return float((y.sum(1) > 0).mean())

ceil_cal, ceil_test = recall_at_pool(ycal), recall_at_pool(ytest)

# ----- max gold score per mention (0 if no gold retrieved) -> conformal nonconformity basis -----
def max_gold_score(y, s):
    masked = np.where(y > 0, s, -1.0)        # gold scores, -1 where not gold
    e = masked.max(1)                        # highest-scoring gold per mention
    e[e < 0] = 0.0                           # no gold retrieved -> 0 (uncoverable by any tau>0)
    return e
Ecal = max_gold_score(ycal, scal)

def split_conformal_threshold(E, alpha):
    """Lower tau s.t. >= (1-alpha) of calib mentions have max-gold-score >= tau (finite-sample)."""
    n = len(E)
    k = int(np.floor(alpha * (n + 1)))       # number we allow to fall below tau
    if k <= 0:
        return -np.inf                       # include everything (tau below all scores)
    if k > n:
        return np.inf
    Es = np.sort(E)                          # ascending
    return float(Es[k - 1])                  # k-th smallest -> ~ (1-alpha) lie at/above it

def evaluate_sets(y, s, tau):
    sets = s >= tau                          # boolean [n, L]: which candidates are in the set
    set_sizes = sets.sum(1)
    covered = ((sets & (y > 0)).sum(1) > 0)  # set contains >=1 gold edge
    return {
        "tau": float(tau),
        "empirical_coverage": float(covered.mean()),
        "mean_set_size": float(set_sizes.mean()),
        "median_set_size": float(np.median(set_sizes)),
        "max_set_size": int(set_sizes.max()),
        "frac_full_pool": float((set_sizes >= L).mean()),
        "frac_empty_set": float((set_sizes == 0).mean()),
    }

results = {
    "meta": {"n_cal": int(n_cal), "n_test": int(len(ytest)), "L": int(L),
             "recall_at_pool_cal": ceil_cal, "recall_at_pool_test": ceil_test,
             "note": "any-coverage is hard-bounded by recall_at_pool_test=%.3f" % ceil_test},
    "split_conformal": [],
}
for target in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
    alpha = 1 - target
    tau = split_conformal_threshold(Ecal, alpha)
    row = {"target_coverage": target, **evaluate_sets(ytest, stest, tau)}
    results["split_conformal"].append(row)

# ----- risk-coverage / selective prediction (the abstain reframing) -----
# confidence = top-1 candidate score per mention; accept the most-confident fraction f.
# selective metric: InR_any@1 (is the single top edge a gold edge) among accepted.
conf = stest.max(1)
top1_is_gold = np.array([ytest[i, stest[i].argmax()] > 0 for i in range(len(ytest))], dtype=float)
order = np.argsort(-conf)                    # most confident first
acc_sorted = top1_is_gold[order]
cum_acc = np.cumsum(acc_sorted) / np.arange(1, len(acc_sorted) + 1)  # selective accuracy @ coverage f
coverages = np.arange(1, len(acc_sorted) + 1) / len(acc_sorted)
aurc = float(np.mean(1 - cum_acc))           # area under risk(=1-acc) vs coverage
rc_curve = [{"coverage": float(coverages[i]), "selective_InR_any@1": float(cum_acc[i])}
            for i in [int(0.1*len(coverages))-1, int(0.2*len(coverages))-1, int(0.3*len(coverages))-1,
                      int(0.5*len(coverages))-1, len(coverages)-1] if i >= 0]
results["risk_coverage"] = {"AURC_top1": aurc, "InR_any@1_full": float(top1_is_gold.mean()),
                            "curve_samples": rc_curve}

# ----- ECE: treat each candidate score as P(edge is gold); reliability over score bins -----
def ece(y, s, n_bins=10):
    sc = s.ravel(); lab = (y.ravel() > 0).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0; tot = len(sc); rows = []
    for b in range(n_bins):
        m = (sc >= bins[b]) & (sc < bins[b + 1] if b < n_bins - 1 else sc <= bins[b + 1])
        if m.sum() == 0: continue
        conf_b, acc_b, w = sc[m].mean(), lab[m].mean(), m.sum() / tot
        e += w * abs(conf_b - acc_b)
        rows.append({"bin": [float(bins[b]), float(bins[b+1])], "n": int(m.sum()),
                     "mean_score": float(conf_b), "emp_gold_rate": float(acc_b)})
    return float(e), rows
ece_val, ece_rows = ece(ytest, stest)
results["ECE_test"] = {"ECE": ece_val, "bins": ece_rows}

with open(os.path.join(HERE, "conformal_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ----- console summary -----
print("="*78)
print("PHASE 1 -- SPLIT-CONFORMAL over Edge-Cross-encoder scores (MM-S14-Disease)")
print("="*78)
print("calib(valid-NIL) n=%d  recall@pool=%.3f | test n=%d  recall@pool=%.3f" %
      (n_cal, ceil_cal, len(ytest), ceil_test))
print("  -> ANY-coverage is HARD-BOUNDED by test recall@pool = %.1f%%\n" % (100*ceil_test))
print("Split-conformal 'any'-coverage edge sets (calibrated on valid-NIL, eval on test-NIL):")
print("  target   emp.cov   mean|med set   %%full-pool   %%empty")
for r in results["split_conformal"]:
    print("   %.2f     %.3f     %5.1f|%2d        %.2f         %.2f" % (
        r["target_coverage"], r["empirical_coverage"], r["mean_set_size"],
        int(r["median_set_size"]), r["frac_full_pool"], r["frac_empty_set"]))
print("\nRisk-coverage (selective, confidence=top-1 score):")
print("  InR_any@1 (accept all) = %.3f | AURC = %.3f" %
      (results["risk_coverage"]["InR_any@1_full"], aurc))
for c in rc_curve:
    print("   accept %.0f%% most-confident -> selective InR_any@1 = %.3f" %
          (100*c["coverage"], c["selective_InR_any@1"]))
print("\nECE (test, score-as-P(gold)) = %.3f" % ece_val)
print("="*78)
print("written: phase1/conformal_results.json")
