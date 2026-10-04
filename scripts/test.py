#!/usr/bin/env python3
"""Generates the full Business Entity Resolution submission package and zips it.

Usage:
    python create_project.py --team myteam
    python create_project.py --team myteam --outputs path/to/student_resource/output
"""
import argparse
import shutil
import zipfile
from pathlib import Path

# ---------------------------------------------------------------- file contents

CONFIG = r'''from pathlib import Path

SEED = 42

DATA_DIR = Path("dataset")          # run from student_resource/ root
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUT_DIR = Path("output")
ARTIFACTS = Path("artifacts")

# --- blocking ---
NGRAM_RANGE = (3, 5)
TFIDF_TOPK = 15
TFIDF_MIN_SIM = 0.30
RT_TOP = 20
RT_MIN_W = 1.0
RT_DF_MAX = 0.02
RT_MAX_TOKENS_PER_DOC = 10
KEY_CAP = 100
PRUNE_TOPM = 64

# --- model / calibration ---
LGB_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=63,
    min_child_samples=40, feature_fraction=0.9,
    bagging_fraction=0.8, bagging_freq=1, n_jobs=-1,
    verbosity=-1, seed=SEED,
)
NUM_ROUNDS = 600
N_FOLDS = 5
N_THRESHOLDS = 150
'''

NORMALIZE = r'''import re
import unicodedata

WORD_RE = re.compile(r"\w+", re.UNICODE)

ABBREV = {
    "pvt": "private", "ltd": "limited", "corp": "corporation",
    "inc": "incorporated", "co": "company", "bros": "brothers",
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "hwy": "highway", "ln": "lane",
    "fl": "floor", "ste": "suite", "apt": "apartment", "bldg": "building",
    "no": "number", "nr": "near", "opp": "opposite", "mkt": "market",
}
LEGAL = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "company", "co", "pvt", "private", "llp", "lp", "plc", "gmbh", "ag",
    "sas", "sarl", "sa", "srl", "spa", "bv", "nv", "pte", "kg", "oy", "ab",
    "pvtltd", "holdings",
}
STOP = {"and", "the", "of", "for", "a", "an", "at", "in", "near", "opposite",
        "beside", "behind", "above", "below", "number"}

POSTAL_RE = re.compile(r"(?<![0-9])(\d{4,6}(?:-\d{4})?)(?![0-9])")
HOUSE_RE = re.compile(r"(?<![a-z0-9])(\d{1,5}[a-z]?)(?![a-z0-9])")


def basic(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    return re.sub(r"\s+", " ", s).strip()


def tokens(s: str):
    return [t for t in (ABBREV.get(w, w) for w in WORD_RE.findall(s))
            if t not in STOP]


def name_core_tokens(s: str):
    return [t for t in tokens(s) if t not in LEGAL]


def core_str(toks) -> str:
    return " ".join(toks)


def key_sorted(toks) -> str:
    return " ".join(sorted(set(toks)))


def extract_postal(addr: str) -> str:
    m = POSTAL_RE.search(addr)
    return m.group(1) if m else ""


def extract_house(addr: str) -> str:
    m = HOUSE_RE.search(addr)
    return m.group(1) if m else ""
'''

DATA_IO = r'''from pathlib import Path

import pandas as pd

import normalize as N


def load_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for col in ("entity_id", "business_name", "business_address", "country"):
        if col not in df.columns:
            raise ValueError(f"{path}: missing column {col}")
    name_b = df["business_name"].map(N.basic)
    addr_b = df["business_address"].map(N.basic)

    df = df.copy()
    df["name_b"] = name_b
    df["addr_b"] = addr_b
    df["name_tok"] = name_b.map(N.tokens)
    df["name_core"] = name_b.map(N.name_core_tokens)
    df["name_core_str"] = df["name_core"].map(N.core_str)
    df["name_key"] = df["name_core"].map(N.key_sorted)
    df["addr_tok"] = addr_b.map(N.tokens)
    df["addr_core"] = addr_b.map(N.core_str)
    df["postal"] = addr_b.map(N.extract_postal)
    df["house"] = addr_b.map(N.extract_house)
    df["country_n"] = df["country"].map(N.basic)
    return df


def load_ground_truth(path: Path) -> dict:
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    out = {}
    for row in gt.itertuples(index=False):
        ids = frozenset(x for x in row.matched_entity_ids.split(",") if x)
        out[row.source1_entity_id] = ids
    return out


def write_pairs(path: Path, header: str, mapping: dict):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(header + "\n")
        for k, v in mapping.items():
            ids = ",".join(sorted(set(v)))
            f.write(f"{k}\t{ids}\n")
'''

BLOCKING = r'''import math
from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import config as C


def _prune_topm(X, m: int):
    X = X.tocsr()
    for i in range(X.shape[0]):
        seg = X.data[X.indptr[i]:X.indptr[i + 1]]
        if seg.size > m:
            thr = np.partition(seg, seg.size - m)[seg.size - m]
            seg[seg < thr] = 0
    X.eliminate_zeros()
    return normalize(X, norm="l2", copy=False)


def _topk_matches(Xs, Xt, k: int, min_sim: float, chunk: int = 1024):
    XtT = Xt.T.tocsr()
    out = [[] for _ in range(Xs.shape[0])]
    for start in range(0, Xs.shape[0], chunk):
        end = min(start + chunk, Xs.shape[0])
        sims = (Xs[start:end] @ XtT).tocoo()
        if sims.nnz == 0:
            continue
        rows, cols, vals = sims.row, sims.col, sims.data
        bounds = np.flatnonzero(np.diff(rows)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [len(rows)]))
        for s, e in zip(starts, ends):
            r = int(rows[s])
            c, v = cols[s:e], vals[s:e]
            m = v >= min_sim
            c, v = c[m], v[m]
            if c.size == 0:
                continue
            if c.size > k:
                keep = np.argpartition(v, -k)[-k:]
                c = c[keep]
            out[start + r].extend(c.tolist())
    return [np.unique(np.asarray(x, dtype=np.int64)) for x in out]


class Blocker:
    """Union of: tfidf char-ngram cosine, rare-token idf index, exact keys."""

    def fit(self, s1, tgt):
        self.n_tgt = len(tgt)
        corpus = (s1["name_core_str"] + " " + s1["addr_core"]).tolist() + \
                 (tgt["name_core_str"] + " " + tgt["addr_core"]).tolist()
        self.vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=C.NGRAM_RANGE, min_df=2,
            max_df=0.5, sublinear_tf=True, dtype=np.float32,
            max_features=300_000,
        )
        mat = self.vec.fit_transform(corpus)
        self.Xs = _prune_topm(mat[: len(s1)], C.PRUNE_TOPM)
        self.Xt = _prune_topm(mat[len(s1):], C.PRUNE_TOPM)

        tok_lists = tgt["name_core"].map(set).tolist() + \
                    tgt["addr_tok"].map(set).tolist()
        df = defaultdict(int)
        for toks in tok_lists:
            for t in toks:
                df[t] += 1
        df_max = max(1.0, C.RT_DF_MAX * self.n_tgt)
        rare = {t: math.log(1 + self.n_tgt / c)
                for t, c in df.items() if c <= df_max}

        RT_VOCAB = {t: j for j, t in enumerate(sorted(rare))}
        rows, cols, vals = [], [], []

        def build(list_of_sets):
            rows.clear(); cols.clear(); vals.clear()
            for i, toks in enumerate(list_of_sets):
                best = sorted(((rare[t], t) for t in toks if t in rare),
                              reverse=True)[: C.RT_MAX_TOKENS_PER_DOC]
                for w, t in best:
                    rows.append(i)
                    cols.append(RT_VOCAB[t])
                    vals.append(w)
            M = sparse.csr_matrix(
                (vals, (rows, cols)),
                shape=(len(list_of_sets), len(RT_VOCAB)), dtype=np.float32)
            return normalize(M, norm="l2", copy=False)

        self.Rs = build(s1["name_core"].map(set).tolist() +
                        s1["addr_tok"].map(set).tolist())
        self.Rt = build(tok_lists)

        self.post_full, self.post_postal, self.post_house = {}, {}, {}
        keys = tgt["name_key"].tolist()
        postals = tgt["postal"].tolist()
        houses = tgt["house"].tolist()
        first_tok = [t[0] if t else "" for t in tgt["name_core"].tolist()]
        for j in range(self.n_tgt):
            if keys[j]:
                self.post_full.setdefault(keys[j], []).append(j)
            ft = first_tok[j]
            if ft:
                if postals[j]:
                    self.post_postal.setdefault(postals[j] + "|" + ft, []).append(j)
                if houses[j]:
                    self.post_house.setdefault(houses[j] + "|" + ft, []).append(j)

    def candidates(self, s1) -> dict:
        cands = {}
        tfidf_hits = _topk_matches(self.Xs, self.Xt, C.TFIDF_TOPK, C.TFIDF_MIN_SIM)
        rt_hits = _topk_matches(self.Rs, self.Rt, C.RT_TOP, C.RT_MIN_W)
        keys = s1["name_key"].tolist()
        postals = s1["postal"].tolist()
        houses = s1["house"].tolist()
        s1_first = [t[0] if t else "" for t in s1["name_core"].tolist()]
        for i in range(len(s1)):
            found = set(tfidf_hits[i].tolist()) | set(rt_hits[i].tolist())
            ft = s1_first[i]
            klist = [keys[i]] if keys[i] else []
            if ft:
                if postals[i]:
                    klist.append(postals[i] + "|" + ft)
                if houses[i]:
                    klist.append(houses[i] + "|" + ft)
            for key in klist:
                for post in (self.post_full, self.post_postal, self.post_house):
                    hits = post.get(key)
                    if hits:
                        found.update(hits[: C.KEY_CAP])
            if found:
                cands[i] = np.fromiter(found, dtype=np.int64)
        return cands


def flatten(cands: dict) -> np.ndarray:
    rows = [(i, int(j)) for i, arr in cands.items() for j in arr]
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)
'''

FEATURES = r'''import numpy as np
from rapidfuzz import fuzz

FEAT_NAMES = [
    "name_ratio", "name_token_sort", "name_token_set", "name_partial",
    "name_jaccard", "name_contain", "name_exact",
    "addr_ratio", "addr_token_set", "addr_jaccard", "addr_contain",
    "addr_exact", "addr_both_missing", "addr_one_missing",
    "postal_eq", "postal_both_missing", "house_eq", "country_eq",
    "name_len_diff", "name_either_missing",
]


def build_features(s1, tgt, pairs: np.ndarray) -> np.ndarray:
    n1 = s1["name_core_str"].tolist(); a1 = s1["addr_core"].tolist()
    n2 = tgt["name_core_str"].tolist(); a2 = tgt["addr_core"].tolist()
    p1 = s1["postal"].tolist(); p2 = tgt["postal"].tolist()
    h1 = s1["house"].tolist(); h2 = tgt["house"].tolist()
    c1 = s1["country_n"].tolist(); c2 = tgt["country_n"].tolist()
    T1 = s1["name_core"].map(set).tolist(); T2 = tgt["name_core"].map(set).tolist()
    A1 = s1["addr_tok"].map(set).tolist(); A2 = tgt["addr_tok"].map(set).tolist()

    X = np.zeros((len(pairs), len(FEAT_NAMES)), dtype=np.float32)
    ratio, tsort, tset, part = fuzz.ratio, fuzz.token_sort_ratio, \
        fuzz.token_set_ratio, fuzz.partial_ratio

    for i, (a, b) in enumerate(pairs):
        a, b = int(a), int(b)
        sa, sb = n1[a], n2[b]
        ta, tb = T1[a], T2[b]
        aa, ab_ = a1[a], a2[b]
        xa, xb = A1[a], A2[b]
        f = X[i]

        inter = len(ta & tb)
        union = len(ta | tb)
        f[0] = ratio(sa, sb) / 100.0
        f[1] = tsort(sa, sb) / 100.0
        f[2] = tset(sa, sb) / 100.0
        f[3] = part(sa, sb) / 100.0
        f[4] = inter / union if union else 0.0
        f[5] = inter / min(len(ta), len(tb)) if ta and tb else 0.0
        f[6] = 1.0 if sa and sa == sb else 0.0
        f[7] = ratio(aa, ab_) / 100.0
        f[8] = tset(aa, ab_) / 100.0
        ia = len(xa & xb)
        f[9] = ia / len(xa | xb) if (xa | xb) else 0.0
        f[10] = ia / min(len(xa), len(xb)) if xa and xb else 0.0
        f[11] = 1.0 if aa and aa == ab_ else 0.0
        f[12] = 1.0 if (not aa and not ab_) else 0.0
        f[13] = 1.0 if (not aa) != (not ab_) else 0.0
        f[14] = 1.0 if p1[a] and p1[a] == p2[b] else 0.0
        f[15] = 1.0 if (not p1[a]) and (not p2[b]) else 0.0
        f[16] = 1.0 if h1[a] and h1[a] == h2[b] else 0.0
        f[17] = 1.0 if c1[a] == c2[b] else 0.0
        f[18] = abs(len(ta) - len(tb))
        f[19] = 1.0 if (not sa) or (not sb) else 0.0
    return X
'''

METRIC = r'''def f05_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 1.25 * tp + 0.25 * fn + fp
    if denom == 0:
        return 1.0          # empty truth, empty prediction
    return 1.25 * tp / denom


def entity_scores(gt_map: dict, pred_map: dict):
    for s1_id in set(gt_map) | set(pred_map):
        gt = gt_map.get(s1_id, frozenset())
        pred = set(pred_map.get(s1_id, ()))
        tp = len(gt & pred)
        fp = len(pred) - tp
        fn = len(gt) - tp
        yield s1_id, f05_from_counts(tp, fp, fn)


def macro_f05(gt_map: dict, pred_map: dict) -> float:
    scores = list(entity_scores(gt_map, pred_map))
    return sum(s for _, s in scores) / max(1, len(scores))
'''

MODEL = r'''import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

import config as C
from metric import f05_from_counts


def fit_lgb(X, y):
    dtrain = lgb.Dataset(X, label=y)
    return lgb.train(C.LGB_PARAMS, dtrain, num_boost_round=C.NUM_ROUNDS)


def oof_predict(X, y, groups):
    oof = np.zeros(len(y), dtype=np.float64)
    gkf = GroupKFold(n_splits=C.N_FOLDS)
    for tr, va in gkf.split(X, y, groups=groups):
        m = lgb.train(C.LGB_PARAMS, lgb.Dataset(X[tr], label=y[tr]),
                      num_boost_round=C.NUM_ROUNDS)
        oof[va] = m.predict(X[va])
    return oof


def calibrate_threshold(pairs, probs, n_s1, s1_ids, tgt_ids, gt_map,
                        n_thresholds=C.N_THRESHOLDS):
    order = np.argsort(pairs[:, 0], kind="stable")
    p_sorted = pairs[order]
    pr_sorted = probs[order]
    boundaries = np.searchsorted(p_sorted[:, 0], np.arange(n_s1 + 1))

    groups = {}
    for i in range(n_s1):
        s, e = boundaries[i], boundaries[i + 1]
        ids = tgt_ids[p_sorted[s:e, 1]]
        pr = pr_sorted[s:e]
        o = np.argsort(pr, kind="stable")
        groups[i] = (ids[o], pr[o], {t: float(p) for t, p in zip(ids, pr)})

    thresholds = np.unique(np.quantile(probs, np.linspace(0, 1, n_thresholds)))
    thresholds = np.concatenate([thresholds, [probs.max() + 1e-6]])

    best_t, best_s = 0.5, -1.0
    for t in thresholds:
        total, n = 0.0, 0
        for i in range(n_s1):
            s1_id = s1_ids[i]
            gt = gt_map.get(s1_id, frozenset())
            ids, pr, prob_of = groups.get(i, (np.array([]), np.array([]), {}))
            n_ge = len(pr) - int(np.searchsorted(pr, t, side="left")) \
                if len(pr) else 0
            gt_probs = np.array([prob_of[g] for g in gt if g in prob_of])
            tp = int((gt_probs >= t).sum())
            fp = n_ge - tp
            fn = len(gt) - tp
            total += f05_from_counts(tp, fp, fn)
            n += 1
        score = total / n
        if score > best_s:
            best_s, best_t = score, float(t)
    return best_t, best_s
'''

PIPELINE = r'''import json
from collections import defaultdict

import numpy as np
import lightgbm as lgb

import config as C
import data_io as D
from blocking import Blocker, flatten
from features import FEAT_NAMES, build_features
from metric import macro_f05
from model import calibrate_threshold, fit_lgb, oof_predict


def _load_targets(paths):
    return D.__dict__ and None  # placeholder, replaced below


def _concat_targets(paths):
    import pandas as pd
    tgt = pd.concat([D.load_source(p) for p in paths], ignore_index=True)
    return tgt.reset_index(drop=True)


def _labels(pairs, s1_ids, tgt_ids, gt_map):
    y = np.zeros(len(pairs), dtype=np.int32)
    for k, (a, b) in enumerate(pairs):
        if tgt_ids[b] in gt_map.get(s1_ids[a], frozenset()):
            y[k] = 1
    return y


def run_train():
    C.ARTIFACTS.mkdir(exist_ok=True)
    s1 = D.load_source(C.TRAIN_DIR / "train_source1.tsv")
    tgt = _concat_targets([C.TRAIN_DIR / "train_source2.tsv",
                           C.TRAIN_DIR / "train_source3.tsv"])
    gt = D.load_ground_truth(C.TRAIN_DIR / "train_ground_truth.tsv")

    blocker = Blocker()
    blocker.fit(s1, tgt)
    pairs = flatten(blocker.candidates(s1))
    tgt_ids = tgt["entity_id"].to_numpy()
    s1_ids = s1["entity_id"].to_numpy()

    n_gt_pairs = sum(len(v) for v in gt.values())
    hit = sum(1 for a, b in pairs
              if tgt_ids[b] in gt.get(s1_ids[a], frozenset()))
    recall = hit / max(1, n_gt_pairs)
    avg_c = len(pairs) / max(1, len(s1))
    reduction = 1 - len(pairs) / max(1, len(s1) * len(tgt))
    print(f"[blocking] recall={recall:.4f}  cand/S1={avg_c:.1f}  "
          f"reduction={reduction:.6f}")

    X = build_features(s1, tgt, pairs)
    y = _labels(pairs, s1_ids, tgt_ids, gt)
    print(f"[data] pairs={len(pairs)} positives={int(y.sum())}")

    oof = oof_predict(X, y, groups=pairs[:, 0])
    thr, oof_f05 = calibrate_threshold(pairs, oof, len(s1), s1_ids,
                                       tgt_ids, gt)
    print(f"[calibration] threshold={thr:.4f}  OOF macro-F0.5={oof_f05:.4f}")

    model = fit_lgb(X, y)
    model.save_model(str(C.ARTIFACTS / "model.txt"))

    pred_map = defaultdict(list)
    for (a, b), p in zip(pairs, oof):
        if p >= thr:
            pred_map[s1_ids[a]].append(tgt_ids[b])
    val_f05 = macro_f05(gt, pred_map)
    print(f"[validation] macro-F0.5 (OOF @thr) = {val_f05:.4f}")

    (C.ARTIFACTS / "config.json").write_text(json.dumps(
        {"threshold": thr, "features": FEAT_NAMES}), encoding="utf-8")
    (C.ARTIFACTS / "train_report.json").write_text(json.dumps(
        {"blocking_recall": recall, "avg_candidates_per_s1": avg_c,
         "reduction_ratio": reduction, "threshold": thr,
         "oof_macro_f05": oof_f05, "val_macro_f05": val_f05,
         "n_pairs": int(len(pairs)), "n_positives": int(y.sum())},
        indent=2), encoding="utf-8")


def run_predict():
    cfg = json.loads((C.ARTIFACTS / "config.json").read_text())
    thr = cfg["threshold"]
    model = lgb.Booster(model_file=str(C.ARTIFACTS / "model.txt"))

    s1 = D.load_source(C.TEST_DIR / "test_source1.tsv")
    tgt = _concat_targets([C.TEST_DIR / "test_source2.tsv",
                           C.TEST_DIR / "test_source3.tsv"])

    blocker = Blocker()
    blocker.fit(s1, tgt)
    pairs = flatten(blocker.candidates(s1))
    tgt_ids = tgt["entity_id"].to_numpy()
    s1_ids = s1["entity_id"].to_numpy()

    X = build_features(s1, tgt, pairs)
    probs = model.predict(X)

    matching = {s: [] for s in s1_ids}
    candidates = {s: [] for s in s1_ids}
    for (a, b), p in zip(pairs, probs):
        candidates[s1_ids[a]].append(tgt_ids[b])
        if p >= thr:
            matching[s1_ids[a]].append(tgt_ids[b])

    C.OUT_DIR.mkdir(exist_ok=True)
    D.write_pairs(C.OUT_DIR / "matching_results.tsv",
                  "source1_entity_id\tmatched_entity_ids", matching)
    D.write_pairs(C.OUT_DIR / "candidate_pairs.tsv",
                  "source1_entity_id\tcandidate_entity_ids", candidates)
    print(f"[output] {len(matching)} S1 rows, "
          f"{sum(len(v) for v in matching.values())} matches "
          f"@threshold={thr:.4f} -> {C.OUT_DIR}")
'''

RUN = r'''import argparse

import pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "predict", "all"],
                    default="all")
    args = ap.parse_args()
    if args.mode in ("train", "all"):
        pipeline.run_train()
    if args.mode in ("predict", "all"):
        pipeline.run_predict()


if __name__ == "__main__":
    main()
'''

REQUIREMENTS = '''pandas==2.2.3
numpy==1.26.4
scipy==1.13.1
scikit-learn==1.5.2
lightgbm==4.5.0
rapidfuzz==3.10.1
'''

README = '''# Business Entity Resolution — LightGBM + multi-key blocking

## Run (from the student_resource/ root)
```bash
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/run.py --mode all
# -> output/matching_results.tsv, output/candidate_pairs.tsv

python3 utils/validate_submission.py \\
  --matching output/matching_results.tsv \\
  --candidate output/candidate_pairs.tsv --test-dir dataset/test
  '''
