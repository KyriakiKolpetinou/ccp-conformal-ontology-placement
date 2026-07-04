#!/usr/bin/env python
"""
Zero-shot SapBERT recall@N probe for ontology concept placement.

Question: for a new (out-of-KB) mention, how often is a GOLD insertion edge present
in the top-N retrieved candidate edges? This bounds the max achievable conformal
coverage. We use the *base* model the Oxford bi-encoder fine-tunes from
(SapBERT-from-PubMedBERT-fulltext), so these numbers are a conservative LOWER BOUND
on their fine-tuned bi-encoder's recall.

Two retrieval scorings per edge (truth likely between / above these):
  (a) endpoint-max : score = max(cos(m,parent), cos(m,child))   [optimistic]
  (b) edge-text    : score = cos(m, emb("child <is-a> parent")) [bi-encoder-like]

Usage: python recall_probe.py --dataset Disease   (or CPP)
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
DATA = "/home/kkolpetinou/calibrated-concept-placement/data"
OUT = "/home/kkolpetinou/calibrated-concept-placement/results"

def load_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(l) for l in f if l.strip()]

@torch.no_grad()
def embed(texts, tok, model, device, bs=256, maxlen=32):
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    for i in range(0, len(texts), bs):
        b = texts[i:i+bs]
        enc = tok(b, padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            h = model(**enc).last_hidden_state[:, 0]   # [CLS], SapBERT convention
        v = h.float().cpu().numpy()
        out[i:i+len(b)] = v
        if (i // bs) % 50 == 0:
            print(f"  embed {i+len(b)}/{len(texts)}", flush=True)
    # L2 normalize -> cosine via dot
    out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-8)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Disease", choices=["Disease", "CPP"])
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda"
    ds = args.dataset
    base = f"{DATA}/MM-S14-{ds}"
    ont = f"{base}/ontology"
    # locate edge + entity catalogues (names differ only by Disease/CPP)
    edge_path = f"{ont}/SNOMEDCT-US-20140901-{ds}-edges-all.jsonl"
    ent_path  = f"{ont}/SNOMEDCT-US-20140901-{ds}_syn_attr_hyp-all.jsonl"

    print(f"[{ds}] loading catalogues ...", flush=True)
    edges = load_jsonl(edge_path)
    ents  = load_jsonl(ent_path)
    print(f"  edges={len(edges)}  concepts={len(ents)}", flush=True)

    # concept idx -> title, and a contiguous index
    title_by_idx = {e["idx"]: (e["title"] or e.get("entity") or "") for e in ents}
    concept_idxs = list(title_by_idx.keys())
    cpos = {idx: i for i, idx in enumerate(concept_idxs)}
    concept_titles = [title_by_idx[i] for i in concept_idxs]

    # edge endpoints as concept positions (skip edges whose endpoints arent in catalogue)
    par_pos = np.full(len(edges), -1, dtype=np.int64)
    chi_pos = np.full(len(edges), -1, dtype=np.int64)
    edge_texts = []
    for i, e in enumerate(edges):
        p, c = e.get("parent_idx"), e.get("child_idx")
        par_pos[i] = cpos.get(p, -1)
        chi_pos[i] = cpos.get(c, -1)
        edge_texts.append(f"{e.get('child','')} is a {e.get('parent','')}")

    # mentions: pair-level test-NIL (gold) + valid-NIL (calibration pool) for context
    splits = {"test-NIL": f"{base}/mention-edge-pair-level/test-NIL.jsonl",
              "valid-NIL": f"{base}/mention-edge-pair-level/valid-NIL.jsonl"}
    mentions = {}  # split -> {key: {"text":.., "gold":set(edge ids)}}
    for sp, path in splits.items():
        if not os.path.exists(path):
            print(f"  (missing {path})"); continue
        d = {}
        for p in load_jsonl(path):
            key = (p["context_left"], p["mention"], p["context_right"], p["label_concept_ori"])
            if key not in d:
                d[key] = {"text": p["mention"], "gold": set()}
            eid = p["edge_label_id"]
            if eid not in ("", None):
                d[key]["gold"].add(int(eid))
        mentions[sp] = d
        print(f"  {sp}: {len(d)} mentions", flush=True)

    print("loading model ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval().half()

    t0 = time.time()
    print("embedding concepts ...", flush=True)
    C = embed(concept_titles, tok, model, device)        # [n_concepts, d]
    print("embedding edge-texts ...", flush=True)
    E = embed(edge_texts, tok, model, device, maxlen=48)  # [n_edges, d]
    print(f"  embed done in {time.time()-t0:.0f}s", flush=True)

    Ns = [10, 50, 100, 200, 500, 1000]
    report = {"dataset": ds, "model": MODEL, "n_edges": len(edges),
              "n_concepts": len(ents), "Ns": Ns, "results": {}}

    valid_parent = par_pos >= 0
    valid_child = chi_pos >= 0
    for sp, d in mentions.items():
        keys = list(d.keys())
        mtexts = [d[k]["text"] for k in keys]
        M = embed(mtexts, tok, model, device)            # [n_mentions, d]

        # (a) endpoint-max scoring
        cs = M @ C.T                                      # [n_mentions, n_concepts]
        sp_par = np.where(valid_parent, cs[:, np.clip(par_pos, 0, None)], -1.0)
        sp_chi = np.where(valid_child, cs[:, np.clip(chi_pos, 0, None)], -1.0)
        score_max = np.maximum(sp_par, sp_chi)            # [n_mentions, n_edges]
        # (b) edge-text scoring
        score_txt = M @ E.T                               # [n_mentions, n_edges]

        for name, scores in [("endpoint_max", score_max), ("edge_text", score_txt)]:
            rec = {}
            maxN = max(Ns)
            # argpartition top-maxN once, then check each N
            top = np.argpartition(-scores, kth=maxN-1, axis=1)[:, :maxN]
            # sort those maxN by score for prefix checks
            for r, k in enumerate(keys):
                row = top[r]
                row = row[np.argsort(-scores[r, row])]
                top[r] = row
            for N in Ns:
                hits = 0
                for r, k in enumerate(keys):
                    gold = d[k]["gold"]
                    if gold & set(top[r, :N].tolist()):
                        hits += 1
                rec[N] = hits / len(keys)
            report["results"][f"{sp}:{name}"] = rec
            print(f"[{ds}] {sp} {name} recall@N: " +
                  " ".join(f"@{N}={rec[N]*100:.1f}%" for N in Ns), flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/recall_probe_{ds}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved {OUT}/recall_probe_{ds}.json", flush=True)

if __name__ == "__main__":
    main()
