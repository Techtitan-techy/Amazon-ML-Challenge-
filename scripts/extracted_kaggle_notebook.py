
# ============================================================
# CELL 0 [markdown]
# ============================================================
"""
# Cross-encoder re-scoring on top of the GBDT (GPU)
**Settings: Accelerator = GPU (T4 x2 or P100), Internet = On** (**second run, seed 1, training capped at 50 min, for an ensemble with CE_v5_base**; downloads **intfloat/multilingual-e5-base**, MIT license, 278M params; ~2.5x slower than e5-small).
**Inputs:** challenge data, kaggle_b (wheels), **ber-v5-T** output, **ber-v5-S** output, **both F5 outputs** (India, US+France).
Trains a multilingual pair classifier on train candidates (GBDT holdout S1 excluded), picks the blend weight and
threshold on the GBDT holdout, re-scores the uncertain test pairs, writes `ce_out/` (validated). ~1.5 h.
"""

# ============================================================
# CELL 1 [code]
# ============================================================
import glob, os, sys, subprocess
whl = sorted({os.path.dirname(w) for w in glob.glob("/kaggle/input/**/*.whl", recursive=True)})
try:
    import anyascii, rapidfuzz
except ImportError:
    args = ["--no-index"] + sum([["--find-links", d] for d in whl], []) if whl else []
    print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args, "anyascii", "rapidfuzz"], capture_output=True, text=True).stderr[-500:])
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", transformers.__version__)
assert torch.cuda.is_available(), "turn on a GPU accelerator (Settings -> Accelerator)"
os.makedirs("/kaggle/working/ber/src", exist_ok=True); os.makedirs("/kaggle/working/ber/scripts", exist_ok=True)
%cd /kaggle/working/ber

# ============================================================
# CELL 2 [code]
# ============================================================
%%writefile src/__init__.py
# business entity resolution pipeline

# ============================================================
# CELL 3 [code]
# ============================================================
%%writefile src/blocking.py
"""Candidate generation: IDF-weighted multi-key token blocking, fully vectorized.

Every record emits a set of hashed keys (country-scoped):
  name  : phonetic-skeleton tokens, 5-char prefixes, skeleton token pairs
  addr  : skeleton tokens, (house-number, token) pairs
An S1 record and a pool record share a key -> they accumulate idf(key) into a name or
address score. Keys that are too common (pool df > cap) are dropped. For every S1 we
keep the union of the top-K by total / name / address score. All joins are numpy
range-expansions over sorted key arrays (no Python loop per pair).
"""
import os
import multiprocessing as mp
import numpy as np
import pandas as pd

from .normalize import skeleton, phon

STREET_TYPES = {"st", "ave", "rd", "dr", "ln", "cir", "ct", "blvd", "pl", "pkwy", "hwy", "ter", "trl",
                "sq", "cv", "way", "loop", "n", "s", "e", "w", "ne", "nw", "se", "sw", "ste", "apt", "flr",
                "bldg", "sec", "ngr", "col", "marg", "rue", "all", "chem", "imp", "rte", "fbg", "quai",
                "crs", "res", "main", "new", "old"}
MAX_PAIR_TOKS = 6


MAX_CROSS_NAME, MAX_CROSS_ADDR, MAX_ADDR_PAIR = 4, 6, 6
EXTRA_KEYS = True
PHON_KEYS = False          # --phon: full sound-keyed name (alone, x address word, x house number)
MORE_KEYS = False          # --extra-keys: joined-name suffix + name-pair x house number (set before KeyIndex)
SINGLE_KINDS = set("trpjaz")


def record_keys(country, n_clean, a_clean, nums):
    """Returns (name_keys, addr_keys, cross_keys) as sets of strings."""
    c = country
    nk, ak, xk = set(), set(), set()
    ntoks = n_clean.split()
    sk = []
    for t in ntoks:
        s = skeleton(t)
        if len(s) >= 2:
            nk.add(f"t{c}|{s}")
            sk.append(s)
        elif len(t) >= 2:
            nk.add(f"r{c}|{t}")
        if len(t) >= 5:
            nk.add(f"p{c}|{t[:5]}")
    joined = "".join(ntoks)
    if len(joined) >= 8:                       # 'southernexports.com' <-> 'Southern Exports'
        nk.add(f"j{c}|{joined[:8]}")
        if MORE_KEYS:                          # survives injected leading tokens ('@paramounttrading', '>> sion...')
            nk.add(f"z{c}|{joined[-8:]}")
    if PHON_KEYS and len(ntoks) >= 2:
        ph = sorted({p for p in (phon(t) for t in ntoks) if len(p) >= 2})[:5]
        if len(ph) >= 2:
            nk.add(f"v{c}|{'_'.join(ph)}")
    sk = list(dict.fromkeys(sk))[:MAX_PAIR_TOKS]
    if EXTRA_KEYS:
        # typo-tolerant name keys: pairs of 4-char prefixes ('consulatscy' ~ 'consultancy')
        pf = list(dict.fromkeys(t[:4] for t in ntoks if len(t) >= 3))[:MAX_PAIR_TOKS]
        for i in range(len(pf)):
            for j in range(i + 1, len(pf)):
                a, b = (pf[i], pf[j]) if pf[i] < pf[j] else (pf[j], pf[i])
                nk.add(f"c{c}|{a}|{b}")
    for i in range(len(sk)):
        for j in range(i + 1, len(sk)):
            a, b = (sk[i], sk[j]) if sk[i] < sk[j] else (sk[j], sk[i])
            nk.add(f"b{c}|{a}|{b}")
    if a_clean or nums:
        ask = []
        for t in a_clean.split():
            if t in STREET_TYPES:
                continue
            s = skeleton(t) if t.isalpha() else t
            if len(s) >= 2:
                ask.append(s)
        ask = list(dict.fromkeys(ask))
        for s in ask:
            ak.add(f"a{c}|{s}")
        nl = nums.split()[:2]
        for n in nl:
            for s in ask:
                ak.add(f"n{c}|{n}|{s}")
        ap = ask[:MAX_ADDR_PAIR]
        for i in range(len(ap)):
            for j in range(i + 1, len(ap)):
                a, b = (ap[i], ap[j]) if ap[i] < ap[j] else (ap[j], ap[i])
                ak.add(f"q{c}|{a}|{b}")
        for s_ in sk[:MAX_CROSS_NAME]:
            for a in ask[:MAX_CROSS_ADDR] + nl:
                xk.add(f"x{c}|{s_}|{a}")
        if MORE_KEYS:                          # common names made specific by the house number
            npair = sk[:MAX_CROSS_NAME]
            for i in range(len(npair)):
                for j in range(i + 1, len(npair)):
                    a, b = (npair[i], npair[j]) if npair[i] < npair[j] else (npair[j], npair[i])
                    for n in nl:
                        xk.add(f"y{c}|{a}|{b}|{n}")
        if PHON_KEYS:                          # common India names in any script, made specific by the address
            ph = sorted({p for p in (phon(t) for t in ntoks) if len(p) >= 2})[:5]
            if ph:
                fn = "_".join(ph)
                for s_ in ask[:MAX_CROSS_ADDR] + nl[:1]:
                    xk.add(f"w{c}|{fn}|{s_}")
        if EXTRA_KEYS:
            nu = list(dict.fromkeys(nums.split()))[:4]
            for i in range(len(nu)):          # number pairs: '5534 10480' survives a replaced name
                for j in range(i + 1, len(nu)):
                    a, b = (nu[i], nu[j]) if nu[i] < nu[j] else (nu[j], nu[i])
                    ak.add(f"u{c}|{a}|{b}")
    return nk, ak, xk


def _keys_chunk(args):
    off, countries, names, addrs, nums = args
    recs, keys, fams = [], [], []
    for i, (c, n, a, u) in enumerate(zip(countries, names, addrs, nums)):
        for fam, ks in enumerate(record_keys(c.lower(), n, a, u)):
            for k in ks:
                recs.append(off + i); keys.append(k)
                # fam code: +0 name / +1 addr / +2 cross ; +4 if single-token key (t,r,p,j,a)
                fams.append(fam + (4 if k[0] in SINGLE_KINDS else 0))
    h = pd.util.hash_array(np.array(keys, dtype=object), categorize=False)
    return np.array(recs, np.int32), h, np.array(fams, np.int8)


def build_keys(df, n_jobs=None, chunk=100_000):
    n_jobs = n_jobs or os.cpu_count()
    cols = [df.country.tolist(), df.n_clean.tolist(), df.a_clean.tolist(), df.nums.tolist()]
    parts = [(i, *[c[i:i + chunk] for c in cols]) for i in range(0, len(df), chunk)]
    if n_jobs > 1 and len(parts) > 1:
        with mp.get_context("fork").Pool(n_jobs) as pool:
            res = pool.map(_keys_chunk, parts)
    else:
        res = [_keys_chunk(p) for p in parts]
    return (np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res]),
            np.concatenate([r[2] for r in res]))


def _group_rank(g, score):
    """rank (0 = best) of each row within its group g, by descending score.
    Single int64 argsort on (group, -score) instead of lexsort (~5x faster)."""
    v = np.asarray(score, np.float32)
    bits = v.view(np.int32).astype(np.int64)
    bits = np.where(bits < 0, np.int64(0x7FFFFFFF) - bits, bits)   # order-preserving for negatives
    key = (np.asarray(g, np.int64) << 32) | (np.int64(0xFFFFFFFF) - (bits + np.int64(0x80000000)) & np.int64(0xFFFFFFFF))
    order = np.argsort(key, kind="stable")
    gs = g[order]
    first = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    rank_sorted = np.arange(len(g)) - np.repeat(first, np.diff(np.r_[first, len(g)]))
    rank = np.empty(len(g), np.int32)
    rank[order] = rank_sorted
    return rank


class KeyIndex:
    """Sorted pool-key index + S1 key lists; build once, run blocking passes many times."""

    def __init__(self, s1, pool, cap=600, df_scale=1.0, n_jobs=None, verbose=True, cap_composite=None):
        import time
        t0 = time.time()
        pr, pk, pf = build_keys(pool, n_jobs)
        sr, sk, sf = build_keys(s1, n_jobs)
        if verbose:
            print(f"[block] keys pool={len(pk):,} s1={len(sk):,} ({time.time() - t0:.0f}s)", flush=True)
        order = np.argsort(pk)
        pk = pk[order]
        self.pr = pr[order]
        pf = pf[order]
        del order, pr
        newk = np.empty(len(pk), bool)
        newk[0] = True
        np.not_equal(pk[1:], pk[:-1], out=newk[1:])
        self.starts = np.flatnonzero(newk)
        uk = pk[self.starts]
        del pk, newk
        self.counts = np.diff(np.r_[self.starts, len(self.pr)]).astype(np.int64)
        self.idf = np.log((len(pool) + 1) / (self.counts + 1)).astype(np.float32)
        # per-record idf mass of usable keys per family (name/addr/cross) -> cosine-normalized scores:
        # summed idf favours verbose records over true matches with short / empty addresses
        self.pmass = np.zeros((len(pool), 3))
        for a0 in range(0, len(self.pr), 50_000_000):
            a1 = min(a0 + 50_000_000, len(self.pr))
            kr = np.searchsorted(self.starts, np.arange(a0, a1), side="right") - 1
            f = pf[a0:a1]
            capr = np.where(f >= 4, cap, cap_composite or cap)
            w = np.where(self.counts[kr] * df_scale <= capr, self.idf[kr], 0).astype(np.float64)
            for fam_k in range(3):
                self.pmass[:, fam_k] += np.bincount(self.pr[a0:a1], weights=w * ((f & 3) == fam_k), minlength=len(pool))
        self.pmass = self.pmass.astype(np.float32)
        del pf
        pos = np.searchsorted(uk, sk)
        pos[pos == len(uk)] = 0
        capv = np.where(sf >= 4, cap, cap_composite or cap)
        ok = (uk[pos] == sk) & (self.counts[pos] * df_scale <= capv)
        self.sr, self.sf, self.pos = sr[ok], (sf[ok] & 3).astype(np.int8), pos[ok]
        self.n_s1 = len(s1)
        self.smass = np.stack([np.bincount(self.sr, weights=self.idf[self.pos].astype(np.float64) * (self.sf == k),
                                           minlength=len(s1)) for k in range(3)], 1).astype(np.float32)
        del sk, ok, uk
        if verbose:
            print(f"[block] cap={cap} usable s1 keys={len(self.sr):,} "
                  f"expansion={self.counts[self.pos].sum():,} ({time.time() - t0:.0f}s)", flush=True)


def generate_candidates(s1, pool, cap=600, df_scale=1.0, k_total=30, k_name=10, k_addr=10, k_cross=0,
                        budget=60_000_000, n_jobs=None, verbose=True, prerank=None, s1_subset=None, index=None,
                        k_cos=0, k_prior=0, prior=None):
    """Returns DataFrame [s1_idx, pool_idx, bs_*] (row indices into s1/pool).
    index: a prebuilt KeyIndex (reuse across passes). s1_subset: bool mask of S1 rows to process.
    prerank: callable(DataFrame)->DataFrame applied to every chunk's kept pairs (learned pruning)."""
    import time
    t0 = time.time()
    ix = index or KeyIndex(s1, pool, cap, df_scale, n_jobs, verbose)
    pr, starts, counts, idf = ix.pr, ix.starts, ix.counts, ix.idf
    sr, sf, pos = ix.sr, ix.sf, ix.pos
    if s1_subset is not None:
        m = s1_subset[sr]
        sr, sf, pos = sr[m], sf[m], pos[m]

    out = []
    # chunk S1 records so each chunk expands to <= budget pairs (bounded memory)
    rec_exp = np.bincount(sr, weights=counts[pos], minlength=len(s1))
    cum = np.cumsum(rec_exp)
    rb = np.unique(np.r_[0, np.searchsorted(cum, np.arange(budget, cum[-1] + budget, budget)), len(s1)])
    bounds = np.searchsorted(sr, rb)
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        if b1 <= b0:
            continue
        r, f, p = sr[b0:b1], sf[b0:b1], pos[b0:b1]
        cnt = counts[p]
        tot = int(cnt.sum())
        rep_idx = np.repeat(np.arange(len(r)), cnt)
        offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(starts[p], cnt)
        pairs = (r[rep_idx].astype(np.int64) << 24) | pr[offs].astype(np.int64)
        w = idf[p][rep_idx]
        fam = f[rep_idx]
        del offs, rep_idx
        up, inv = np.unique(pairs, return_inverse=True)
        del pairs
        s_name = np.bincount(inv, weights=w * (fam == 0), minlength=len(up)).astype(np.float32)
        s_addr = np.bincount(inv, weights=w * (fam == 1), minlength=len(up)).astype(np.float32)
        s_cross = np.bincount(inv, weights=w * (fam == 2), minlength=len(up)).astype(np.float32)
        n_nk = np.bincount(inv, weights=(fam == 0), minlength=len(up)).astype(np.float32)
        n_ak = np.bincount(inv, weights=(fam == 1), minlength=len(up)).astype(np.float32)
        n_xk = np.bincount(inv, weights=(fam == 2), minlength=len(up)).astype(np.float32)
        nkeys = np.bincount(inv, minlength=len(up)).astype(np.int16)
        g = (up >> 24).astype(np.int32)
        s_tot = s_name + s_addr + s_cross
        keep = (_group_rank(g, s_tot) < k_total) | (_group_rank(g, s_name) < k_name) | \
               (_group_rank(g, s_addr) < k_addr) | (_group_rank(g, s_cross) < k_cross)
        extra = {}
        pc = None
        if k_prior and prior is not None:
            # reserved slots for pool records that are almost surely true copies of SOME S1 but miss a key
            # family: empty address (99.3% owned) by name score, invented name (97-98% owned) by address score
            pc = prior[(up & ((1 << 24) - 1)).astype(np.int64)]
            for code, sc in ((1, s_name), (2, s_addr)):
                m = pc == code
                if m.any():
                    keep |= m & (_group_rank(g, np.where(m, sc, np.float32(-1))) < k_prior)
        if k_cos:
            pidx = (up & ((1 << 24) - 1)).astype(np.int64)
            ms, mp_ = ix.smass[g], ix.pmass[pidx]
            cos = s_tot / np.sqrt((ms.sum(1) + 1.0) * (mp_.sum(1) + 1.0))
            ncos = s_name / np.sqrt((ms[:, 0] + 1.0) * (mp_[:, 0] + 1.0))
            acos = s_addr / np.sqrt((ms[:, 1] + 1.0) * (mp_[:, 1] + 1.0))
            keep |= _group_rank(g, cos) < k_cos
            extra = {"bs_cos": cos[keep].astype(np.float32), "bs_ncos": ncos[keep].astype(np.float32),
                     "bs_acos": acos[keep].astype(np.float32)}
            del pidx, ms, mp_, cos, ncos, acos
        if pc is not None:
            extra["bs_prior"] = pc[keep].astype(np.float32)
            del pc
        blk = pd.DataFrame({"s1_idx": g[keep], "pool_idx": (up[keep] & ((1 << 24) - 1)).astype(np.int32),
                            "bs_name": s_name[keep], "bs_addr": s_addr[keep], "bs_cross": s_cross[keep],
                            "bs_nkeys": nkeys[keep], "bs_nnk": n_nk[keep], "bs_nak": n_ak[keep],
                            "bs_nxk": n_xk[keep], **extra})
        if prerank is not None:
            blk = prerank(blk)
        out.append(blk)
        del inv, w, fam, s_name, s_addr, s_cross, n_nk, n_ak, n_xk, nkeys, g, s_tot
        if verbose:
            print(f"[block] s1 rows {r[0]:,}-{r[-1]:,}: expanded {tot:,} -> uniq {len(up):,} -> kept {len(out[-1]):,}"
                  f" ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(out, ignore_index=True)

# ============================================================
# CELL 4 [code]
# ============================================================
%%writefile src/features.py
"""Pairwise features for (S1, pool) candidate pairs.

* String similarities: rapidfuzz.process.cpdist (C++, multithreaded, element-wise pairs).
* Set / noise-model / number features: lean Python in forked workers over chunks.
* Competition context (ranks / margins among candidates): vectorized numpy/pandas.
"""
import math
import os
import multiprocessing as mp
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from .normalize import phon, LEGAL as _LEGAL
from .noise import token_noise_feats, number_feats, number_diff_feats, edit_feats, number_sign_feats, TOKEN_COLS, NUM_COLS, NUMD_COLS, EDIT_COLS, NUMS_COLS

_G = {}


# ------------------------------------------------------------------ IDF tables
def build_idf(series_list):
    c = Counter()
    n = 0
    for s in series_list:
        for v in s:
            n += 1
            c.update(set(v.split()))
    return {k: math.log((n + 1) / (v + 1)) for k, v in c.items()}, math.log(n + 1)


# ------------------------------------------------------------------ vectorized string sims
CPD = [  # (name, column, scorer, joined-without-spaces)
    ("n_ratio", "n_clean", fuzz.ratio, False),
    ("n_tset", "n_clean", fuzz.token_set_ratio, False),
    ("n_tsort", "n_clean", fuzz.token_sort_ratio, False),
    ("n_partial", "n_clean", fuzz.partial_ratio, False),
    ("n_jw_join", "n_clean", JaroWinkler.normalized_similarity, True),
    ("n_ratio_join", "n_clean", fuzz.ratio, True),
    ("n_partial_join", "n_clean", fuzz.partial_ratio, True),
    ("n_skel_tset", "nsk", fuzz.token_set_ratio, False),
    ("a_ratio", "a_clean", fuzz.ratio, False),
    ("a_tset", "a_clean", fuzz.token_set_ratio, False),
    ("a_partial", "a_clean", fuzz.partial_ratio, False),
    ("a_skel_tset", "ask", fuzz.token_set_ratio, False),
    ("u_tset", "nums", fuzz.token_set_ratio, False),
]


def cpdist_features(A, B, workers=-1):
    out = {}
    cache = {}
    for name, col, scorer, join in CPD:
        k = (col, join)
        if k not in cache:
            a, b = A[col], B[col]
            if join:
                cache[k] = ([x.replace(" ", "") for x in a], [x.replace(" ", "") for x in b])
            else:
                cache[k] = (list(a), list(b))
        a, b = cache[k]
        v = cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)
        empty = np.fromiter((not x or not y for x, y in zip(a, b)), bool, len(a))
        v[empty] = -1
        out[name] = v
    return out


# ------------------------------------------------------------------ python per-pair features
def _wjacc(sa, sb, idf, dflt):
    if not sa or not sb:
        return -1.0, -1.0, 0.0
    inter = sa & sb
    wi = sum(idf.get(t, dflt) for t in inter)
    wu = wi + sum(idf.get(t, dflt) for t in sa ^ sb)
    miss = max((idf.get(t, dflt) for t in sa - sb), default=0.0)
    return wi / wu if wu else 0.0, len(inter) / len(sa | sb), miss


def _num_old(n1, n2):
    a, b = n1.split(), n2.split()
    if not a or not b:
        return -1.0, -1.0, -1.0, -1.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    near = 0.0
    if not inter:
        near = float(any(Levenshtein.distance(x, y) <= 1 for x in sa for y in sb if len(x) >= 3 or len(y) >= 3))
    return inter / len(sa | sb), float(a[0] == b[0]), float(inter == 0), near


PY_SRC = ["nsk", "ask", "legal", "nums", "state", "rn", "ra", "ru", "country"]
PY_COLS = (["n_wjacc", "n_jacc", "n_miss_idf", "n_miss_idf_r", "n_first_eq", "n_ntok1", "n_ntok2",
            "legal_eq", "legal_ov", "a_wjacc", "a_jacc", "a_miss_idf", "a_miss_idf_r",
            "num_jacc", "num_first_eq", "num_conflict", "num_near", "state_eq"]
           + ["nn_" + c for c in TOKEN_COLS] + ["an_" + c for c in TOKEN_COLS] + NUM_COLS)


EXTRA_COLS = {"numdiff": NUMD_COLS, "skel": ["ns_" + c for c in TOKEN_COLS],
              "edits": ["ce_" + c for c in EDIT_COLS] + ["cea_" + c for c in EDIT_COLS],
              "numsign": NUMS_COLS, "phon": ["ph_jacc", "ph_cov", "ph_full"]}
EXTRA_ORDER = ("numdiff", "skel", "edits", "numsign", "phon")


def _phon_set(tokens):
    return {phon(t) for t in tokens if len(t) >= 2 and t.isalpha() and t not in _LEGAL} - {""}


def extra_cols(extras):
    return [c for e in EXTRA_ORDER if e in extras for c in EXTRA_COLS[e]]


def _py_row(k1, a1, l1, u1, st1, rn1, ra1, ru1, ct1, k2, a2, l2, u2, st2, rn2, ra2, ru2, ct2):
    idf_n, dn, idf_a, da, nadd, ndrop, aadd, adrop, fq_n, fq_a = _G["t"]
    t1, t2 = k1.split(), k2.split()
    s1, s2 = set(t1), set(t2)
    wj, jac, miss = _wjacc(s1, s2, idf_n, dn)
    _, _, miss_r = _wjacc(s2, s1, idf_n, dn)
    x1, x2 = set(a1.split()), set(a2.split())
    if x1 and x2:
        awj, ajac, amiss = _wjacc(x1, x2, idf_a, da)
        _, _, amiss_r = _wjacc(x2, x1, idf_a, da)
    else:
        awj = ajac = amiss = amiss_r = -1.0
    row = ((wj, jac, miss, miss_r, float(t1[:1] == t2[:1]), float(len(t1)), float(len(t2)),
             -1.0 if (not l1 or not l2) else float(l1 == l2),
             -1.0 if (not l1 or not l2) else float(bool(set(l1.split()) & set(l2.split()))),
             awj, ajac, amiss, amiss_r)
            + _num_old(u1, u2)
            + (-1.0 if (not st1 or not st2) else float(st1 == st2),)
            + token_noise_feats(rn1.split(), rn2.split(), nadd, ndrop, ct1.lower(), fq_n)
            + token_noise_feats(ra1.split(), ra2.split(), aadd, adrop, ct1.lower(), fq_a)
            + number_feats(tuple(ru1.split()), tuple(ru2.split())))
    ex = _G.get("extras", ())
    if "numdiff" in ex:
        row += number_diff_feats(u1, u2)
    if "skel" in ex:          # script-independent noise model: add/drop on phonetic-skeleton tokens
        sadd, sdrop, fq_s = _G["skel"]
        row += token_noise_feats(t1, t2, sadd, sdrop, ct1.lower(), fq_s)
    if "edits" in ex:         # one-letter edit class log-odds: typo (true copy) vs distractor edit
        en, ea = _G["edits"]
        row += edit_feats(rn1.split(), rn2.split(), en) + edit_feats(ra1.split(), ra2.split(), ea)
    if "numsign" in ex:       # signed house-number delta + learned log-odds of the exact delta
        row += number_sign_feats(u1, u2, ct1.lower(), _G["numsign"])
    if "phon" in ex:          # name overlap on sound keys: English vs native-script romanization
        p1, p2 = _phon_set(rn1.split()), _phon_set(rn2.split())
        if p1 and p2:
            inter = len(p1 & p2)
            row += (inter / len(p1 | p2), inter / len(p1), float(p1 == p2))
        else:
            row += (-1.0, -1.0, -1.0)
    return row


def _py_chunk(cols):
    return np.array([_py_row(*r) for r in zip(*cols)], dtype=np.float32)


def _init(tables, extras=(), skel=None, edits=None, numsign=None):
    _G["t"] = tables
    _G["extras"] = tuple(extras)
    _G["skel"] = skel
    _G["edits"] = edits
    _G["numsign"] = numsign


def string_feature_cols(extras=()):
    return [c for c, *_ in CPD] + PY_COLS + extra_cols(extras)


def string_features(cands, s1, pool, tables, n_jobs=None, chunk=1_000_000, sub=100_000, extras=(), skel=None,
                    edits=None, numsign=None, out=None):
    """tables = (idf_n, dn, idf_a, da, name_add_lo, name_drop_lo, addr_add_lo, addr_drop_lo,
                  frequent_name_tokens, frequent_addr_tokens).
    Returns a float32 DataFrame (one preallocated block, no intermediate copies).
    out: optional (n, len(string_feature_cols(extras))) float32 view to fill in place, e.g. the left block of
    the training matrix, so the string features never exist twice in memory."""
    n_jobs = n_jobs or os.cpu_count()
    ii, jj = cands.s1_idx.values, cands.pool_idx.values
    need = sorted({c for _, c, _, _ in CPD} | set(PY_SRC))
    S1 = {c: s1[c].array for c in need}
    PL = {c: pool[c].array for c in need}
    cols = string_feature_cols(extras)
    if out is None:
        out = np.empty((len(ii), len(cols)), np.float32)
    elif out.shape != (len(ii), len(cols)):
        raise ValueError(f"out has shape {out.shape}, expected {(len(ii), len(cols))}")
    ncpd = len(CPD)
    with mp.get_context("fork").Pool(n_jobs, initializer=_init, initargs=(tables, extras, skel, edits, numsign)) as pp:
        for k in range(0, len(ii), chunk):
            i, j = ii[k:k + chunk], jj[k:k + chunk]
            A = {c: np.asarray(S1[c].take(i), dtype=object) for c in need}
            B = {c: np.asarray(PL[c].take(j), dtype=object) for c in need}
            f = cpdist_features(A, B)
            for q, (name, *_) in enumerate(CPD):
                out[k:k + len(i), q] = f[name]
            parts = [[A[c][q:q + sub].tolist() for c in PY_SRC] + [B[c][q:q + sub].tolist() for c in PY_SRC]
                     for q in range(0, len(i), sub)]
            out[k:k + len(i), ncpd:] = np.vstack(pp.map(_py_chunk, parts))
            del A, B, f, parts
    return pd.DataFrame(out, columns=cols, index=cands.index, copy=False)


# ------------------------------------------------------------------ competition context
def _top2(g, v):
    """Per-row: best and second-best value within the row's group (second=-1 if none)."""
    order = np.lexsort((-v, g))
    gs, vs = g[order], v[order]
    first = np.r_[True, gs[1:] != gs[:-1]]
    grp = np.cumsum(first) - 1
    starts = np.flatnonzero(first)
    sizes = np.diff(np.r_[starts, len(g)])
    b1 = vs[starts]
    b2 = np.where(sizes > 1, vs[np.minimum(starts + 1, len(g) - 1)], -1.0)
    t1, t2 = np.empty(len(g), vs.dtype), np.empty(len(g), vs.dtype)
    t1[order], t2[order] = b1[grp], b2[grp]
    return t1, t2


CTX = ["rank_s1", "gap_s1", "ratio_s1", "top2gap_s1", "rank_pool", "gap_pool", "margin_pool"]


def _context_into(out, col0, g1, g2, v, rows):
    """Write the 7 competition-context features of score v into out[:, col0:col0+7] (rows subset)."""
    from .blocking import _group_rank
    sel = (lambda x: x) if rows is None else (lambda x: x[rows])
    out[:, col0 + 0] = sel(_group_rank(g1, v))
    t1, t2 = _top2(g1, v)
    vv = sel(v)
    out[:, col0 + 1] = sel(t1) - vv
    out[:, col0 + 2] = vv / (sel(t1) + 1e-6)
    out[:, col0 + 3] = sel(t1) - sel(t2)
    del t1, t2
    out[:, col0 + 4] = sel(_group_rank(g2, v))
    p1, p2 = _top2(g2, v)
    p1s, p2s = sel(p1), sel(p2)
    del p1, p2
    out[:, col0 + 5] = p1s - vv
    out[:, col0 + 6] = vv - np.where(vv >= p1s, p2s, p1s)


def context_features(c, score_col, prefix):
    v = c[score_col].values.astype(np.float32)
    out = np.empty((len(c), len(CTX)), np.float32)
    _context_into(out, 0, c.s1_idx.values, c.pool_idx.values, v, None)
    return pd.DataFrame(out, columns=[f"{prefix}_{k}" for k in CTX], index=c.index)


BASE_CHEAP = ["bs_name", "bs_addr", "bs_cross", "bs_nkeys", "bs_nnk", "bs_nak", "bs_nxk", "pr_score", "pr_rank",
              "bs_cos", "bs_ncos", "bs_acos"]


PF_BITS = ["allcaps", "lower", "leadjunk", "bracket", "dblspace", "domain", "nonascii", "digit", "alias", "idtag"]
PA_BITS = ["empty", "allcaps", "masked", "nulltok", "dblspace", "zeropad", "landmark"]
FMT_FEATS = ["pf_" + b for b in PF_BITS] + ["pa_" + b for b in PA_BITS] + ["fmt_noise", "raw_name_eq", "raw_addr_eq"]


def fmt_features(c, s1, pool, rows=None, out=None):
    """Raw-format fingerprints of the pool record + raw-string equality with the S1 record.
    out: optional (n, len(FMT_FEATS)) float32 view to fill in place (no second full-size block)."""
    si, pj = c.s1_idx.values, c.pool_idx.values
    if rows is not None:
        si, pj = si[rows], pj[rows]
    fn = np.asarray(pool.fn.values)[pj].astype(np.int32)
    fa = np.asarray(pool.fa.values)[pj].astype(np.int32)
    if out is None:
        out = np.empty((len(si), len(FMT_FEATS)), np.float32)
    for b in range(len(PF_BITS)):
        out[:, b] = (fn >> b) & 1
    for b in range(len(PA_BITS)):
        out[:, len(PF_BITS) + b] = (fa >> b) & 1
    k = len(PF_BITS) + len(PA_BITS)
    noisy_n = sum(((fn >> b) & 1) for b in (0, 1, 2, 3, 4, 5, 6, 9))
    noisy_a = sum(((fa >> b) & 1) for b in (1, 2, 3, 4, 5))
    out[:, k] = noisy_n + noisy_a
    out[:, k + 1] = np.asarray(s1.hn.values)[si] == np.asarray(pool.hn.values)[pj]
    out[:, k + 2] = np.asarray(s1.ha.values)[si] == np.asarray(pool.ha.values)[pj]
    return out


OOV_FEATS = ["pn_oov_frac", "pn_oov_n", "pn_oov_all"]


def name_oov(s1, pool):
    """Per pool record: share and count of its clean-name tokens that no S1 name of the same country uses.
    The generator replaces some true copies' names with invented words ('Jaxhalo', 'Belodelta'); 97-98% of
    pool records named by one unseen word are true copies (ERR-02), distractors reuse real vocabulary."""
    frac = np.zeros(len(pool), np.float32)
    cnt = np.zeros(len(pool), np.float32)
    pc = np.asarray(pool.country.values)
    sc = np.asarray(s1.country.values)
    for ct in pd.unique(pc):
        voc = {t for n in pd.unique(np.asarray(s1.n_clean.values)[sc == ct]) if isinstance(n, str) for t in n.split()}
        rows = np.flatnonzero(pc == ct)
        names = np.asarray(pool.n_clean.values)[rows]
        memo = {}
        for r, n in zip(rows, names):
            v = memo.get(n)
            if v is None:
                t = n.split() if isinstance(n, str) else []
                k = sum(x not in voc for x in t)
                v = memo[n] = (k / len(t) if t else 0.0, float(k))
            frac[r], cnt[r] = v
    return frac, cnt


def cheap_features(c, s1, pool, rows=None, fmt=False, oov=False):
    """Features that need the FULL candidate set (competition context), materialized only for `rows`
    (bool mask or index array; None = all). One preallocated float32 block, no DataFrame copies:
    at 77M candidate pairs this is the difference between ~5 GB and an OOM kill."""
    if rows is not None and np.asarray(rows).dtype == bool:
        rows = np.flatnonzero(rows)
    n = len(c) if rows is None else len(rows)
    sel = (lambda x: x) if rows is None else (lambda x: x[rows])
    base = [b for b in BASE_CHEAP if b in c]
    has_pr = "pr_score" in c
    cols = (["src", "n_cands_s1", "n_cands_pool", "a1_empty", "a2_empty"] + base + ["bs_total"]
            + [f"bst_{k}" for k in CTX] + ([f"prs_{k}" for k in CTX] if has_pr else [])
            + (FMT_FEATS if fmt else []) + (OOV_FEATS if oov else []))
    out = np.empty((n, len(cols)), np.float32)
    g1, g2 = c.s1_idx.values, c.pool_idx.values
    si, pj = sel(g1), sel(g2)
    out[:, 0] = pool.src.values[pj]
    out[:, 1] = np.bincount(g1, minlength=len(s1))[si]
    out[:, 2] = np.bincount(g2, minlength=len(pool))[pj]
    out[:, 3] = (np.asarray(s1.a_clean.str.len().fillna(0).values) == 0)[si]
    out[:, 4] = (np.asarray(pool.a_clean.str.len().fillna(0).values) == 0)[pj]
    k = 5
    for b in base:
        out[:, k] = sel(c[b].values)
        k += 1
    tot = (c.bs_name.values + c.bs_addr.values + c.bs_cross.values).astype(np.float32)
    out[:, k] = sel(tot)
    k += 1
    _context_into(out, k, g1, g2, tot, rows)
    del tot
    k += len(CTX)
    if has_pr:
        _context_into(out, k, g1, g2, c.pr_score.values.astype(np.float32), rows)
        k += len(CTX)
    if fmt:
        fmt_features(c, s1, pool, rows, out=out[:, k:k + len(FMT_FEATS)])
        k += len(FMT_FEATS)
    if oov:
        of, on = np.asarray(pool.oov_f.values)[pj], np.asarray(pool.oov_n.values)[pj]
        out[:, k], out[:, k + 1], out[:, k + 2] = of, on, (of >= 1.0) & (on > 0)
    return pd.DataFrame(out, columns=cols, copy=False)

# ============================================================
# CELL 5 [code]
# ============================================================
%%writefile src/france.py
"""Normalization fixes for a country that is absent from training (France), applied to that country's rows only.
Other countries keep the cached normalization, so the model trained on US/India sees unchanged inputs there.

1. Admin components: region / departement names that the normalizer does not know as 'states' stay in the address
   ('hauts france', 'pas calais', 'gironde'), and true copies swap region <-> departement. They are LEARNED from the
   data (no table): components that dominate the last comma field of S1 or pool addresses of that country
   (the FR-ADMIN-01 rule, LB-confirmed in our first pipeline), and are removed from every address component.
2. Extra legal / generic forms (cie, ets, etablissements, freres, associes, ei, eirl, earl, gaec, scea, sem) are
   recognized as legal forms during that country's name normalization.
3. (off by default) removing the record's own country token from names: it hides the strongest decoy marker
   (a name that GAINS its country word is a decoy 100% of the time in train); on France it added 64k pairs, 42% of
   them with an added 'France'."""
import re

import numpy as np
import pandas as pd

from . import normalize as N
from .prep import NORM_COLS, normalize_df, read_tsv

EXTRA_LEGAL = {"cie": "cie", "ets": "ets", "etablissements": "ets", "freres": "freres", "associes": "associes",
               "ei": "ei", "eirl": "eirl", "earl": "earl", "gaec": "gaec", "scea": "scea", "sem": "sem"}
_NONALNUM = re.compile(r"[^a-z0-9 ]+")


def comp_clean(c):
    return " ".join(_NONALNUM.sub(" ", N.to_ascii(c or "").lower().replace(".", " ")).split())


def learn_admin(s1_addr, pool_addr, min_share=0.005):
    last = lambda a: comp_clean(a.split(",")[-1]) if (a or "").strip() else ""
    a = pd.Series([last(x) for x in s1_addr]).value_counts(normalize=True)
    b = pd.Series([last(x) for x in pool_addr]).value_counts(normalize=True)
    both = pd.DataFrame({"s1": a, "pool": b}).fillna(0)
    keep = both[(both.s1 >= min_share) | ((both.pool >= min_share) & (both.s1 < both.pool / 10))]
    return {k for k in keep.index if k and not any(ch.isdigit() for ch in k)}


def strip_admin(addr, admin):
    return ",".join(p for p in (addr or "").split(",") if comp_clean(p) not in admin)


def strip_country(name, country):
    out = re.sub(rf"\(?\b{re.escape(country)}\b\)?", " ", name or "", flags=re.I)
    out = " ".join(out.split())
    return out if re.search(r"[A-Za-z0-9]", out) else name


def _replace(frame, raw, cols, n_jobs):
    """Re-normalize the rows of `raw` (entity_id, business_name, business_address, ...) and write NORM_COLS into
    `frame` at the matching entity_id positions. Formatting columns (fn, fa, hn, ha) keep the ORIGINAL raw values."""
    pos = pd.Index(frame.entity_id.values).get_indexer(raw.entity_id.values)
    ok = pos >= 0
    raw, pos = raw[ok], pos[ok]
    saved = dict(N.LEGAL)
    N.LEGAL.update(EXTRA_LEGAL)                   # visible to the forked normalization workers
    try:
        nz = normalize_df(raw[["entity_id", "business_name", "business_address"]].copy(), n_jobs)
    finally:
        N.LEGAL.clear()
        N.LEGAL.update(saved)
    for c in cols:
        v = frame[c].to_numpy(dtype=object)
        v[pos] = nz[c].to_numpy(dtype=object)
        frame[c] = pd.array(v, dtype="string[pyarrow]")
        del v
    return int(ok.sum())


def apply_bundle(data_dir, split, s1, pool, countries, n_jobs=None, log=print, strip_country_token=False):
    """In-place: re-normalize the rows of `countries` in the cached s1 / pool frames with the bundle."""
    d = f"{data_dir}/{split}"
    raw1 = read_tsv(f"{d}/{split}_source1.tsv")
    rawp = pd.concat([read_tsv(f"{d}/{split}_source{k}.tsv") for k in (2, 3)], ignore_index=True)
    for ct in countries:
        r1, rp = raw1[raw1.country == ct].copy(), rawp[rawp.country == ct].copy()
        if not len(r1):
            log(f"[france] {ct}: no S1 rows, skipped")
            continue
        admin = learn_admin(r1.business_address.tolist(), rp.business_address.tolist())
        log(f"[france] {ct}: {len(admin)} learned admin components, e.g. {sorted(admin)[:12]}")
        for r in (r1, rp):
            r["business_address"] = [strip_admin(x, admin) for x in r.business_address]
            if strip_country_token:
                r["business_name"] = [strip_country(x, ct) for x in r.business_name]
        cols = [c for c in NORM_COLS if c in s1.columns]
        n1 = _replace(s1, r1, cols, n_jobs)
        n2 = _replace(pool, rp, cols, n_jobs)
        log(f"[france] {ct}: re-normalized {n1:,} S1 and {n2:,} pool rows")
    return s1, pool

# ============================================================
# CELL 6 [code]
# ============================================================
%%writefile src/matching.py
"""Pair classifier, set selection, and the competition metric."""
import numpy as np
import pandas as pd
import lightgbm as lgb

NON_FEATS = {"y", "s1_idx", "pool_idx", "fold"}

LGB_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=255, verbose=-1, num_threads=0)


def feat_cols(X):
    return [c for c in X.columns if c not in NON_FEATS]


def train_lgb(X, y, rounds=600, params=None, valid=None, feature_name="auto"):
    p = dict(LGB_PARAMS, **(params or {}))
    dtr = lgb.Dataset(X, y, free_raw_data=True, feature_name=feature_name)
    vs = [lgb.Dataset(valid[0], valid[1], reference=dtr)] if valid is not None else []
    cb = [lgb.log_evaluation(100)] + ([lgb.early_stopping(50, verbose=False)] if vs else [])
    return lgb.train(p, dtr, num_boost_round=rounds, valid_sets=vs, callbacks=cb)


def select(s1_idx, pool_idx, p, thr=0.5, one_to_one=True, min_top=None):
    """Pick matches. Each pool record goes to at most one S1 (its best-scoring one).
    `min_top`: if set, an S1 whose best candidate has p >= min_top keeps it even if < thr."""
    df = pd.DataFrame({"s1": s1_idx, "pool": pool_idx, "p": p})
    if one_to_one:
        best = df.groupby("pool").p.transform("max")
        df = df[df.p >= best]
        df = df.drop_duplicates("pool")
    keep = df.p >= thr
    if min_top is not None:
        top = df.groupby("s1").p.transform("max")
        keep |= (df.p >= top) & (df.p >= min_top)
    return df[keep][["s1", "pool", "p"]]


def f05_macro(pred_pairs, true_pairs, all_s1):
    """pred_pairs/true_pairs: DataFrames with columns s1, pool (ids or idx). all_s1: iterable of s1 keys."""
    all_s1 = pd.Index(pd.unique(np.asarray(all_s1)))
    tp = pred_pairs.merge(true_pairs, on=["s1", "pool"]).groupby("s1").size()
    npred = pred_pairs.groupby("s1").size()
    ntrue = true_pairs.groupby("s1").size()
    tp, npred, ntrue = (s.reindex(all_s1, fill_value=0).values.astype(float) for s in (tp, npred, ntrue))
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(npred > 0, tp / npred, 0.0)
        rec = np.where(ntrue > 0, tp / ntrue, 0.0)
        f = np.where(tp > 0, 1.25 * prec * rec / (0.25 * prec + rec), 0.0)
    f = np.where((ntrue == 0) & (npred == 0), 1.0, f)
    return float(f.mean()), dict(precision=float(prec[npred > 0].mean()) if (npred > 0).any() else 0.0,
                                 recall=float(rec[ntrue > 0].mean()),
                                 singleton_acc=float(((npred == 0)[ntrue == 0]).mean()) if (ntrue == 0).any() else 1.0)

# ============================================================
# CELL 7 [code]
# ============================================================
%%writefile src/noise.py
"""Learned noise-model features.

The data generator perturbs true copies and distractors differently:
  * distractors ADD tokens from a specific vocabulary (public, private, group, holdings, india, ...)
    and SUBSTITUTE house-number digits (111 -> 115);
  * true copies add alias markers (dba, formerly, aka, nee), drop/mask/zero-pad digits (16282 -> 6282, ##1).
We learn per-token log-odds of being ADDED / DROPPED in true pairs vs. negative candidates (from
training labels, out-of-fold) and classify the edit relation between house numbers.
"""
import re
from collections import Counter

import numpy as np
from .translit import to_ascii as unidecode   # own romanizer (no GPL)

from .normalize import skeleton, collapse, LEGAL

SK = "\u00a7"   # prefix for phonetic-stem keys inside the log-odds dicts


from functools import lru_cache


@lru_cache(maxsize=2_000_000)
def stem_key(t: str):
    """Language-agnostic stem: singularize + consonant skeleton ('groupe'/'group' -> 'jrp',
    'developpement'/'development' -> 'dblpmnt'). None when too short to be distinctive."""
    if len(t) > 4 and t.endswith("s"):
        t = t[:-1]
    k = skeleton(collapse(t))
    return k if len(k) >= 3 else None


LEGAL_KEY, COUNTRY_KEY = SK + "LEGAL", SK + "COUNTRY"


def _special_key(t, country):
    if country and t == country:
        return COUNTRY_KEY
    if t in LEGAL and LEGAL[t]:
        return LEGAL_KEY
    return None


def lookup(lo, t, country="", freq=None):
    """exact token -> [only for corpus-frequent tokens] special class (own country name / legal form)
    -> phonetic stem -> 0. Frequency gate: distractor markers come from a vocabulary (frequent),
    typos are random (rare) and must not inherit a stem's statistics."""
    v = lo.get(t)
    if v is not None:
        return v
    if freq is None or t not in freq:
        return 0.0
    sp = _special_key(t, country)
    if sp:
        return lo.get(sp, 0.0)
    k = stem_key(t)
    return lo.get(SK + k, 0.0) if k else 0.0


_TOK = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"[0-9#]+")


def raw_tokens(s: str):
    if not s:
        return ()
    if not s.isascii():
        s = unidecode(s)
    return tuple(sorted(set(_TOK.findall(s.lower()))))


def raw_numbers(s: str):
    if not s:
        return ()
    return tuple(t for t in _NUM.findall(s) if any(c.isdigit() for c in t))


# ------------------------------------------------------------------ token log-odds
class LogOddsCounter:
    """Incremental version of learn_logodds (feed chunks, then .result())."""

    def __init__(self):
        self.add_p, self.add_n, self.drop_p, self.drop_n = Counter(), Counter(), Counter(), Counter()
        self.npos = self.nneg = 0

    def update(self, pairs_a, pairs_b, y, countries=None):
        countries = countries if countries is not None else [""] * len(y)
        for a, b, t, ct in zip(pairs_a, pairs_b, y, countries):
            sa, sb = set(a.split()), set(b.split())
            add, drop = sb - sa, sa - sb
            ct = (ct or "").lower()
            add = list(add) + [k for k in (_special_key(x, ct) for x in add) if k]
            drop = list(drop) + [k for k in (_special_key(x, ct) for x in drop) if k]
            if t:
                self.npos += 1
                self.add_p.update(add)
                self.drop_p.update(drop)
            else:
                self.nneg += 1
                self.add_n.update(add)
                self.drop_n.update(drop)

    def result(self, min_count=25, prior=1.0):
        npos, nneg = self.npos, self.nneg

        def lo(cp, cn):
            out = {k: float(np.log((cp[k] + prior) / (npos + prior)) - np.log((cn[k] + prior) / (nneg + prior)))
                   for k in set(cp) | set(cn) if cp[k] + cn[k] >= min_count}
            # phonetic-stem aggregate: lets unseen spellings in another language ('groupe') inherit
            # the statistics of their cognate ('group'); used only when the exact token is unknown
            sp, sn = Counter(), Counter()
            for c_, d_ in ((cp, sp), (cn, sn)):
                for k, v in c_.items():
                    if k.startswith(SK):
                        continue
                    sk = stem_key(k)
                    if sk:
                        d_[sk] += v
            for k in set(sp) | set(sn):
                if sp[k] + sn[k] >= min_count:
                    out[SK + k] = float(np.log((sp[k] + prior) / (npos + prior)) - np.log((sn[k] + prior) / (nneg + prior)))
            return out

        return lo(self.add_p, self.add_n), lo(self.drop_p, self.drop_n)


def learn_logodds(pairs_a, pairs_b, y, min_count=25, prior=1.0):
    """pairs_a/pairs_b: sequences of token tuples (S1 side, pool side). y: labels.
    Returns (add_lo, drop_lo) dicts: log P(token added | match) / P(token added | non-match)."""
    add_p, add_n, drop_p, drop_n = Counter(), Counter(), Counter(), Counter()
    npos = nneg = 0
    for a, b, t in zip(pairs_a, pairs_b, y):
        sa, sb = set(a), set(b)
        if t:
            npos += 1
            add_p.update(sb - sa)
            drop_p.update(sa - sb)
        else:
            nneg += 1
            add_n.update(sb - sa)
            drop_n.update(sa - sb)

    def lo(cp, cn):
        out = {}
        for k in set(cp) | set(cn):
            if cp[k] + cn[k] >= min_count:
                out[k] = float(np.log((cp[k] + prior) / (npos + prior)) - np.log((cn[k] + prior) / (nneg + prior)))
        return out

    return lo(add_p, add_n), lo(drop_p, drop_n)


def token_noise_feats(a, b, add_lo, drop_lo, country="", freq=None):
    sa, sb = set(a), set(b)
    add_t, drop_t = sb - sa, sa - sb
    add = [lookup(add_lo, t, country, freq) for t in add_t]
    drop = [lookup(drop_lo, t, country, freq) for t in drop_t]
    return (min(add, default=0.0), sum(add), max(add, default=0.0), float(sum(v < -2 for v in add)),
            min(drop, default=0.0), sum(drop), float(len(add_t)), float(len(drop_t)))


def frequent_tokens(series_list, min_df=50):
    c = Counter()
    for s_ in series_list:
        for v in s_:
            if v:
                c.update(v.split())
    return frozenset(k for k, n in c.items() if n >= min_df)


TOKEN_COLS = ["add_min", "add_sum", "add_max", "add_nbad", "drop_min", "drop_sum", "n_add", "n_drop"]

# ------------------------------------------------------------------ number edit relation
REL = ["equal", "zero_pad", "masked", "deletion", "insertion", "transpose", "substitution", "unrelated", "missing"]
_RANK = {r: i for i, r in enumerate(REL)}


def _rel(x, y):
    if x == y:
        return 0
    if x.lstrip("0") == y.lstrip("0"):
        return 1
    if "#" in y:
        core = y.replace("#", "")
        if (len(y) == len(x) and all(c == d or c == "#" for d, c in zip(x, y))) or (core and core.lstrip("0") in x):
            return 2
    lx, ly = len(x), len(y)
    if ly == lx - 1 and any(x[:i] + x[i + 1:] == y for i in range(lx)):
        return 3
    if ly == lx + 1 and any(y[:i] + y[i + 1:] == x for i in range(ly)):
        return 4
    if lx == ly:
        d = sum(c != e for c, e in zip(x, y))
        if d == 2 and sorted(x) == sorted(y):
            return 5
        if d == 1:
            return 6
    return 7


def number_feats(a, b):
    """a: S1 numbers, b: pool numbers. Returns best relation, counts of equal/substitution, coverage."""
    if not a or not b:
        return (8.0, 0.0, 0.0, 0.0, 0.0, float(len(a)), float(len(b)))
    best_for_a = [min(_rel(x, y) for y in b) for x in a]
    best = min(best_for_a)
    n_eq = sum(r <= 1 for r in best_for_a)
    n_soft = sum(2 <= r <= 5 for r in best_for_a)
    n_sub = sum(r == 6 for r in best_for_a)
    first = min(_rel(a[0], y) for y in b)
    return (float(best), float(first), n_eq / len(a), n_soft / len(a), n_sub / len(a), float(len(a)), float(len(b)))


NUM_COLS = ["num_rel_best", "num_rel_first", "num_eq_frac", "num_soft_frac", "num_sub_frac", "num_n1", "num_n2"]


# ------------------------------------------------------------------ number difference structure (ERR-01)
NUMD_COLS = ["numd_state", "numd_logabs", "numd_pos", "numd_ndig", "numd_rel"]


def number_diff_feats(n1, n2):
    """First S1 house number vs the closest same-length pool number (normalized digits).
    Distractors sit next door (small |d|, last digits differ); true copies carry typos in any digit
    (|d| >= 100 with one wrong digit: P(true) = 0.98 on train). numd_state: 2 equal, 1 same-length diff,
    0 no same-length number, -1 missing."""
    a, b = n1.split(), n2.split()
    if not a or not b:
        return (-1.0, -1.0, -1.0, -1.0, -1.0)
    x = a[0]
    if x in b:
        return (2.0, 0.0, 0.0, 0.0, 0.0)
    same = [v for v in b if len(v) == len(x)]
    if not same or not x.isdigit():
        return (0.0, -1.0, -1.0, -1.0, -1.0)
    xi = int(x)
    v = min(same, key=lambda z: abs(int(z) - xi) if z.isdigit() else 10 ** 12)
    if not v.isdigit():
        return (0.0, -1.0, -1.0, -1.0, -1.0)
    d = abs(int(v) - xi)
    dp = [i for i, (c1, c2) in enumerate(zip(x, v)) if c1 != c2]
    return (1.0, float(np.log10(d + 1)), float(len(x) - dp[0]) if dp else 0.0, float(len(dp)), d / max(xi, 1))


# ------------------------------------------------------------------ one-letter edit fingerprint (FORENSICS-02)
_VOW = frozenset("aeiou")
EDIT_COLS = ["n", "min", "sum", "pos_min", "len_min"]


def edit_key(x, z):
    """(op, position, letter class, length) of a one-edit change x -> z (Damerau distance 1), else None.
    The generator's typos in true copies hit the middle of a word or swap two letters; its distractor
    edits insert/delete/replace the FIRST or LAST letter, swap vowels, or change 1-3 letter words
    (dev train: first-letter insert/delete P(true)=0.02, last-letter insert 0.10, middle 0.73-0.81)."""
    lx, lz = len(x), len(z)
    if abs(lx - lz) > 1 or x == z or not x.isalpha() or not z.isalpha():
        return None
    m = min(lx, lz)
    i = 0
    while i < m and x[i] == z[i]:
        i += 1
    if lx == lz:
        if x[i + 1:] == z[i + 1:]:
            op, a, b = "sub", x[i] in _VOW, z[i] in _VOW
            ch = "vv" if a and b else ("cc" if not a and not b else "vc")
            last = lx - 1
        elif i + 1 < lx and x[i] == z[i + 1] and x[i + 1] == z[i] and x[i + 2:] == z[i + 2:]:
            op, ch, last = "tr", "", lx - 2
        else:
            return None
    else:
        long_, short = (z, x) if lz > lx else (x, z)
        if short[i:] != long_[i + 1:]:
            return None
        op, c = ("ins" if lz > lx else "del"), long_[i]
        ch = "dup" if (i > 0 and long_[i - 1] == c) or (i + 1 < len(long_) and long_[i + 1] == c) else ("v" if c in _VOW else "c")
        last = lx if op == "ins" else lx - 1
    pos = "first" if i == 0 else ("last" if i >= last else "mid")
    return op, pos, ch, min(lx, 7)


EDIT_CTX = False           # --edits-ctx: separate statistics when the edit is the ONLY name difference
MORPH = False              # --morph: also 2-3 letter suffix/prefix appends ('indraksh' -> 'indrakshyn')


def _morph_key(x, z):
    """2-3 letter append/removal at either end of an alphabetic token of >= 3 letters, keyed by the letters
    themselves so the log-odds table can price each one (distractors append pseudo-suffixes like 'yn')."""
    if not (x.isalpha() and z.isalpha()) or min(len(x), len(z)) < 3 or not 2 <= abs(len(z) - len(x)) <= 3:
        return None
    L = min(len(x), 7)
    if len(z) > len(x):
        if z.startswith(x):
            return "suf", z[len(x):], "", L
        if z.endswith(x):
            return "pre", z[:len(z) - len(x)], "", L
    else:
        if x.startswith(z):
            return "dsuf", x[len(z):], "", L
        if x.endswith(z):
            return "dpre", x[:len(x) - len(z)], "", L
    return None


def _edits(a, b):
    """Edit keys of every (dropped S1 token, added pool token) pair at Damerau distance 1."""
    sa, sb = set(a), set(b)
    drop, add = sa - sb, sb - sa
    if not drop or not add:
        return []
    ks = [k for x in drop for z in add if abs(len(x) - len(z)) <= 1 for k in (edit_key(x, z),) if k]
    if MORPH:
        ks += [k for x in drop for z in add for k in (_morph_key(x, z),) if k]
    if EDIT_CTX:              # a lone one-letter change is the distractor's move; typos come with other noise
        c = "1" if len(drop) == 1 and len(add) == 1 else "m"
        ks = [(c + k[0],) + k[1:] for k in ks]
    return ks


def _edit_names(k):
    op, pos, ch, L = k
    return (f"{op}|{pos}|{ch}|{L}", f"{op}|{pos}", f"{op}|L{L}", f"{op}|{ch}")


class EditCounter:
    """Log-odds (true vs negative candidate) of one-letter edit classes; learned on the dictionary slice."""

    def __init__(self):
        self.p, self.n = Counter(), Counter()
        self.npos = self.nneg = 0

    def update(self, pairs_a, pairs_b, y):
        for a, b, t in zip(pairs_a, pairs_b, y):
            keys = {s for k in _edits(a.split(), b.split()) for s in _edit_names(k)}
            if t:
                self.npos += 1
                self.p.update(keys)
            else:
                self.nneg += 1
                self.n.update(keys)

    def result(self, min_count=20, prior=1.0):
        return {k: float(np.log((self.p[k] + prior) / (self.npos + prior)) - np.log((self.n[k] + prior) / (self.nneg + prior)))
                for k in set(self.p) | set(self.n) if self.p[k] + self.n[k] >= min_count}


def edit_feats(a, b, lo):
    """Count of one-edit token changes and their log-odds: min/sum over edits of the most specific known
    class, plus the (op, position) and (op, length) classes alone."""
    ks = _edits(a, b)
    if not ks:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    best, pos, ln = [], [], []
    for k in ks:
        fine, p_, l_, c_ = _edit_names(k)
        v = lo.get(fine)
        best.append(v if v is not None else lo.get(p_, 0.0))
        pos.append(lo.get(p_, 0.0))
        ln.append(lo.get(l_, 0.0))
    return (float(len(ks)), min(best), sum(best), min(pos), min(ln))


# ------------------------------------------------------------------ signed house-number delta (--numsign)
NUMS_COLS = ["nums_d", "nums_lo", "nums_sign", "nums_nun"]


def _closest_delta(n1, n2):
    """Signed delta (pool - S1) of the closest same-length pair of numbers the records do NOT share,
    and the count of unshared S1 numbers. number_diff_feats keeps only |d| of the first S1 number: the sign
    matters (US distractors shift the house number UP by +1..+5, +7, +9, +11, +13, +21; P(true) is 0.76-0.83
    at d = -1/-2 but 0.08-0.09 at d = +1/+2 on dev, name Jaccard >= 0.5)."""
    a, b = n1.split(), n2.split()
    if not a or not b:
        return None, 0
    sa, sb = set(a), set(b)
    ua = [x for x in a if x not in sb and x.isdigit()]
    ub = [v for v in b if v not in sa and v.isdigit()]
    best = None
    for x in ua:
        xi = int(x)
        for v in ub:
            if len(v) == len(x):
                dd = int(v) - xi
                if best is None or abs(dd) < abs(best):
                    best = dd
    return best, len(ua)


def _delta_key(d):
    return str(d) if abs(d) <= 50 else ("big+" if d > 0 else "big-")


class DeltaCounter:
    """Log-odds (true vs negative candidate) of the exact signed delta, per country with a pooled back-off."""

    def __init__(self):
        self.p, self.n = Counter(), Counter()
        self.npos = self.nneg = 0

    def update(self, pairs_a, pairs_b, y, countries):
        for a, b, t, ct in zip(pairs_a, pairs_b, y, countries):
            if t:
                self.npos += 1
            else:
                self.nneg += 1
            d, _ = _closest_delta(a, b)
            if d is None:
                continue
            k = _delta_key(d)
            keys = (f"{(ct or '').lower()}|{k}", f"|{k}")
            (self.p if t else self.n).update(keys)

    def result(self, min_count=20, prior=1.0):
        return {k: float(np.log((self.p[k] + prior) / (self.npos + prior)) - np.log((self.n[k] + prior) / (self.nneg + prior)))
                for k in set(self.p) | set(self.n) if self.p[k] + self.n[k] >= min_count}


def number_sign_feats(n1, n2, country, lo):
    d, nun = _closest_delta(n1, n2)
    if d is None:
        return (-99999.0, 0.0, 0.0, float(nun))
    k = _delta_key(d)
    v = lo.get(f"{country}|{k}")
    if v is None:
        v = lo.get(f"|{k}", 0.0)
    return (float(max(min(d, 10000), -10000)), v, float(np.sign(d)), float(nun))

# ============================================================
# CELL 8 [code]
# ============================================================
%%writefile src/normalize.py
"""Text normalization for business names and addresses.

Country-agnostic: every rule is language-level (Latin transliteration, punctuation,
abbreviation folding), never keyed on a fixed country list, so unseen countries
(France in test) go through the same path.
"""
import re
from .translit import to_ascii as _romanize

# ----------------------------------------------------------------------------- tokens
LEGAL = {
    # EN / IN
    "pvt": "pvt", "private": "pvt", "pte": "pvt", "prvt": "pvt",
    "ltd": "ltd", "limited": "ltd", "ltda": "ltd", "lt": "ltd",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc",
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cos": "co", "tbk": "tbk", "gmbh": "gmbh", "ag": "ag",
    "public": "public", "opc": "opc", "holdings": None, "group": None,
    # FR (and generic civil-law)
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci",
    "snc": "snc", "sca": "sca", "scop": "scop", "selarl": "selarl", "scp": "scp", "gie": "gie",
}
LEGAL_FORMS = {v for v in LEGAL.values() if v}
NAME_STOP = {"the", "and", "of", "de", "des", "du", "la", "le", "les", "d", "l", "et", "a",
             "an", "www", "com", "net", "org", "in", "co.in", "biz", "info", "fr", "us", "dba",
             "nee", "id", "m/s", "ms", "mr", "mrs", "sri", "shri", "shree"}

ADDR_MAP = {
    "street": "st", "str": "st", "st": "st", "avenue": "ave", "av": "ave", "ave": "ave", "avn": "ave",
    "road": "rd", "rd": "rd", "drive": "dr", "dr": "dr", "lane": "ln", "ln": "ln",
    "circle": "cir", "cir": "cir", "court": "ct", "ct": "ct", "boulevard": "blvd", "blvd": "blvd",
    "bd": "blvd", "bld": "blvd", "place": "pl", "pl": "pl", "parkway": "pkwy", "pkwy": "pkwy",
    "highway": "hwy", "hwy": "hwy", "terrace": "ter", "ter": "ter", "trail": "trl", "trl": "trl",
    "square": "sq", "sq": "sq", "cove": "cv", "cv": "cv", "way": "way", "loop": "loop",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "suite": "ste", "ste": "ste", "apartment": "apt", "apt": "apt",
    "floor": "flr", "flr": "flr", "fl": "flr", "etage": "flr", "building": "bldg", "bldg": "bldg",
    "sector": "sec", "sec": "sec", "nagar": "ngr", "ngr": "ngr", "colony": "col", "marg": "marg",
    # FR
    "rue": "rue", "r": "rue", "allee": "all", "all": "all", "chemin": "chem", "chem": "chem",
    "impasse": "imp", "imp": "imp", "route": "rte", "rte": "rte", "faubourg": "fbg", "fbg": "fbg",
    "quai": "quai", "cours": "crs", "crs": "crs", "residence": "res", "res": "res",
}
ADDR_STOP = {"no", "num", "number", "nr", "of", "the", "de", "des", "du", "la", "le", "les", "d", "l",
             "city", "cdp", "borough", "village", "town", "township", "bis", "ter", "unit", "near",
             "opp", "opposite", "behind", "landmark", "house", "door", "flat", "plot", "shop", "h",
             "hno", "dno", "p", "s/o", "so", "at", "po", "post", "dist", "district", "tehsil", "taluka",
             "eme", "er", "nd", "rd_", "th"}

# state / region names -> canonical code, matched by phonetic skeleton so that
# "Maharashtra", "महाराष्ट्र" (-> "mhaaraassttr") and typos land on the same code.
_STATES = {
    # US
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "dc": "district of columbia",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho", "il": "illinois",
    "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon",
    "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
    "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia",
    "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
}
_IN_STATES = {
    "mh": "maharashtra", "dl": "delhi", "up": "uttar pradesh", "ka": "karnataka", "tn": "tamil nadu",
    "gj": "gujarat", "wb": "west bengal", "tg": "telangana", "ts": "telangana", "hr": "haryana",
    "kl": "kerala", "rj": "rajasthan", "br": "bihar", "mp": "madhya pradesh", "ap": "andhra pradesh",
    "od": "odisha", "or": "orissa", "pb": "punjab", "jh": "jharkhand", "ch": "chandigarh",
    "ga": "goa", "as": "assam", "uk": "uttarakhand", "hp": "himachal pradesh", "jk": "jammu and kashmir",
    "ct": "chhattisgarh", "cg": "chhattisgarh", "py": "puducherry",
}

# ----------------------------------------------------------------------------- helpers
_RE_ID = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)", re.I)
_RE_DOT_SPLIT = re.compile(r"\.(?=[a-z0-9]{2,})")
_RE_NONALNUM = re.compile(r"[^a-z0-9/ ]+")
_RE_SPACES = re.compile(r"\s+")
_RE_ORD = re.compile(r"\b(\d+)(st|nd|rd|th|eme|er|e)\b")
_RE_NUM = re.compile(r"\d+")
_OCR = str.maketrans({"0": "o", "1": "l", "5": "s", "3": "e", "4": "a", "8": "b", "7": "t"})


def to_ascii(s: str) -> str:
    if not s:
        return ""
    if not s.isascii():
        s = _romanize(s)
    return s.lower()


def _basic(s: str) -> str:
    s = to_ascii(s)
    s = _RE_ID.sub(" ", s)
    s = s.replace("&", " and ").replace("+", " plus ").replace("@", " at ")
    s = _RE_DOT_SPLIT.sub(" ", s)      # "pvt.ltd" -> "pvt ltd", "ironl.com" -> "ironl com"
    s = s.replace(".", "")             # "s.a.r.l." -> "sarl", "inc." -> "inc"
    s = s.replace("'", "")
    s = _RE_NONALNUM.sub(" ", s)
    return _RE_SPACES.sub(" ", s).strip()


def collapse(t: str) -> str:
    out = []
    for c in t:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


_DIGRAPHS = [("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"),
             ("sh", "s"), ("ch", "c"), ("ck", "k"), ("nb", "mb"), ("np", "mp"), ("x", "ks"),
             ("q", "k"), ("c", "k"), ("v", "b"), ("w", ""), ("z", "j"), ("g", "j"), ("h", "")]
_VOWELS = set("aeiouy")


def skeleton(t: str) -> str:
    """Phonetic consonant skeleton; makes transliterations / vowel typos collide.
    'private'/'praaivett'/'praibhet' -> 'prbt';  'management'/'myaanejmentt' -> 'mnjmnt'."""
    t = collapse(t)
    for a, b in _DIGRAPHS:
        if a in t:
            t = t.replace(a, b)
    return collapse("".join(c for c in t if c not in _VOWELS))


def _fix_ocr(tok: str) -> str:
    # tokens mixing letters and digits with a majority of letters: '0ne' -> 'one'
    if tok.isalnum() and not tok.isalpha() and not tok.isdigit():
        n_alpha = sum(c.isalpha() for c in tok)
        if n_alpha >= len(tok) / 2:
            return tok.translate(_OCR)
    return tok


_STATE_FULL = {}   # exact full name -> code
_STATE_SKEL = {}   # skeleton -> code, only for skeletons that are unambiguous and >= 2 chars
_skel_codes = {}
for _code, _name in list(_STATES.items()) + list(_IN_STATES.items()):
    _code = "od" if _name == "orissa" else _code          # orissa/odisha are one state
    _STATE_FULL.setdefault(_name.replace(" ", ""), _code)
    _skel_codes.setdefault(skeleton(_name.replace(" ", "")), set()).add(_code)
for _sk, _codes in _skel_codes.items():
    # '' (hawaii/iowa/ohio), 'd', 't', 'j' match arbitrary words; 'rjn' is arizona AND oregon
    if len(_sk) >= 2 and len(_codes) == 1:
        _STATE_SKEL[_sk] = next(iter(_codes))


def state_of(part: str):
    """Map one comma-separated address component to a state code if it is one."""
    p = part.replace(" ", "")
    if len(p) == 2 and p.isalpha():
        return p if (p in _STATES or p in _IN_STATES) else None
    if p in _STATE_FULL:
        return _STATE_FULL[p]
    return _STATE_SKEL.get(skeleton(p)) if len(p) >= 4 else None


# ----------------------------------------------------------------------------- public API
_LEGAL_SKEL = {skeleton(k): v for k, v in LEGAL.items() if v and len(k) >= 5}


def _legal(t: str):
    """Legal-form lookup that survives transliteration ('praivet', 'limittedd')."""
    if t in LEGAL:
        return LEGAL[t]
    if len(t) >= 5:
        c = collapse(t)
        if c in LEGAL:
            return LEGAL[c]
        return _LEGAL_SKEL.get(skeleton(t))
    return None


def norm_name(s: str):
    """Returns (clean_string, core_tokens, legal_forms)."""
    b = _basic(s)
    toks = [_fix_ocr(t) for t in b.replace("/", " ").split()]
    core, legal = [], []
    for t in toks:
        v = _legal(t)
        if v:
            legal.append(v)
            continue
        if t in NAME_STOP:
            continue
        core.append(collapse(t) if len(t) > 3 else t)
    return " ".join(core), core, sorted(set(legal))


def norm_addr(s: str):
    """Returns (clean_string, alpha_tokens, numbers, state_code)."""
    if not s:
        return "", [], [], ""
    a = to_ascii(s).replace("#", " ")
    state, keep = "", []
    for part in a.split(","):
        p = _RE_NONALNUM.sub(" ", part.replace(".", " ")).strip()
        st = state_of(p) if p else None
        if st:
            state = st
        else:
            keep.append(part)
    b = _basic(",".join(keep))
    b = _RE_ORD.sub(r"\1", b)
    b = b.replace("/", " ")
    alpha, nums = [], []
    for t in b.split():
        if t.isdigit():
            t = t.lstrip("0") or "0"
            nums.append(t)
            continue
        t = _fix_ocr(t)
        if any(c.isdigit() for c in t):          # 1715b, 8c5 -> keep leading number too
            m = _RE_NUM.match(t)
            if m:
                nums.append(m.group(0).lstrip("0") or "0")
            alpha.append(t)
            continue
        t = ADDR_MAP.get(t, t)
        if t in ADDR_STOP or len(t) < 2:
            continue
        alpha.append(collapse(t))
    return " ".join(alpha), alpha, nums, state



# ------------------------------------------------------------------ sound key (TRANSLIT-02)
_PH_SUB = [("tion", "sn"), ("sion", "sn"), ("ph", "f"), ("gh", ""), ("ck", "k"), ("ch", "k"), ("sh", "s"), ("zh", "s"),
           ("th", "t"), ("dh", "d"), ("bh", "b"), ("kh", "k"), ("wh", "v"), ("w", "v"), ("q", "k"), ("x", "ks"), ("z", "s")]
_PH_C = re.compile(r"c(?=[eiy])")
_PH_G = re.compile(r"g(?=[eiy])")
_PH_V = re.compile(r"[aeiouy]")
_PH_D = re.compile(r"(.)\1+")


def phon(t):
    """Consonant sound key shared by an English spelling and its phonetic romanization from an Indic script
    ('technologies'/'teknolajis' -> tknljs, 'international'/'intaraneshanal' -> ntrnsnl, 'foundation'/'phaundeshan'
    -> fndsn). The consonant skeleton misses these (ch/k, ph/f, ti/sh, soft g/j, w/v)."""
    if not t.isalpha():
        return t
    t = _PH_G.sub("j", _PH_C.sub("s", t))
    for a, b in _PH_SUB:
        t = t.replace(a, b)
    t = _PH_V.sub("", t.replace("c", "k").replace("h", ""))
    return _PH_D.sub(r"\1", t)

# ============================================================
# CELL 9 [code]
# ============================================================
%%writefile src/prep.py
"""Load the three sources, normalize in parallel, cache as parquet."""
import os
import re
import zlib
import multiprocessing as mp
import numpy as np
import pandas as pd

from .normalize import norm_name, norm_addr, skeleton
from .noise import raw_tokens, raw_numbers

NORM_COLS = ["n_clean", "legal", "a_clean", "nums", "state", "nsk", "ask", "rn", "ra", "ru"]
FMT_COLS = ["fn", "fa", "hn", "ha"]      # raw-format bitmasks + raw-string hashes (formatting is lost by normalization)
READ_KW = dict(sep="\t", dtype=str, keep_default_na=False, quoting=3)


def read_tsv(path):
    df = pd.read_csv(path, **READ_KW)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.replace("\r", "", regex=False)
    return df


_ALIAS = re.compile(r"\b(dba|d/b/a|formerly|aka|a\.k\.a|nee|n\u00e9e|fka|f/k/a|trading as|t/a)\b", re.I)
_NULL = re.compile(r"\b(null|none|nan|n/a)\b|<null>", re.I)
_DOMAIN = re.compile(r"\.(com|in|net|org|co|biz|info|fr)\b|www\.", re.I)
_LANDMARK = re.compile(r"\b(near|opp|opposite|behind|beside)\b", re.I)
_IDTAG = re.compile(r"\bid\s*:?\s*\d", re.I)
_LEADJUNK = re.compile(r"^\W")
_ZEROPAD = re.compile(r"\b0\d")


def fmt_record(n, a):
    """Raw-format fingerprints. True copies carry more formatting noise (case changes, junk prefixes, domain
    style, scripts) than distractors, which are clean text with a semantic edit (FORENSICS-01)."""
    fn = (int(n.isupper()) | int(n.islower()) << 1 | int(bool(_LEADJUNK.match(n))) << 2 | int("(" in n or "[" in n) << 3
          | int("  " in n) << 4 | int(bool(_DOMAIN.search(n))) << 5 | int(not n.isascii()) << 6
          | int(any(ch.isdigit() for ch in n)) << 7 | int(bool(_ALIAS.search(n))) << 8 | int(bool(_IDTAG.search(n))) << 9)
    fa = (int(not a.strip()) | int(a.isupper()) << 1 | int("#" in a) << 2 | int(bool(_NULL.search(a))) << 3
          | int("  " in a) << 4 | int(bool(_ZEROPAD.search(a))) << 5 | int(bool(_LANDMARK.search(a))) << 6)
    return fn, fa, zlib.crc32(" ".join(n.lower().split()).encode()), zlib.crc32(" ".join(a.lower().split()).encode())


def _norm_chunk(args):
    names, addrs = args
    out = []
    for n, a in zip(names, addrs):
        nc, _, legal = norm_name(n)
        ac, _, nums, st = norm_addr(a)
        out.append((nc, " ".join(legal), ac, " ".join(nums), st,
                    " ".join(skeleton(t) for t in nc.split()),          # name skeleton tokens
                    " ".join(skeleton(t) if t.isalpha() else t for t in ac.split()),  # addr skeleton tokens
                    " ".join(raw_tokens(n)), " ".join(raw_tokens(a)),   # raw tokens (noise model)
                    " ".join(raw_numbers(a)))                           # raw numbers incl. '#' masks
                   + fmt_record(n or "", a or ""))
    return out


def normalize_df(df, n_jobs=None, chunk=50_000):
    n_jobs = n_jobs or os.cpu_count()
    names, addrs = df.business_name.tolist(), df.business_address.tolist()
    parts = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(df), chunk)]
    if n_jobs > 1 and len(parts) > 1:
        with mp.get_context("fork").Pool(n_jobs) as pool:
            res = pool.map(_norm_chunk, parts)
    else:
        res = [_norm_chunk(p) for p in parts]
    flat = [r for part in res for r in part]
    out = pd.DataFrame(flat, columns=NORM_COLS + FMT_COLS, index=df.index)
    out[FMT_COLS[:2]] = out[FMT_COLS[:2]].astype(np.int16)
    out[FMT_COLS[2:]] = out[FMT_COLS[2:]].astype(np.int64)
    # raw text is no longer needed downstream; dropping it roughly halves RAM at full scale
    return pd.concat([df.drop(columns=["business_name", "business_address"]), out], axis=1)


def _compact(df):
    """Arrow-backed strings: ~3x less RAM than Python str objects at 10M rows."""
    for c in NORM_COLS:
        if c in df:
            df[c] = df[c].astype("string[pyarrow]")
    return df


def _read_arrow(path):
    """Read parquet straight into Arrow-backed string columns (never materializes Python str objects)."""
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    df = t.to_pandas(types_mapper=lambda typ: pd.StringDtype("pyarrow") if (str(typ) in ("string", "large_string")) else None,
                     self_destruct=True)
    del t
    return df


def load_split(data_dir, split, cache_dir=None, n_jobs=None):
    """Returns (s1, pool). pool = S2 ++ S3 with a `src` column (2/3)."""
    cache = os.path.join(cache_dir, f"{split}_norm") if cache_dir else None
    if cache and os.path.exists(cache + "_s1.parquet"):
        return _read_arrow(cache + "_s1.parquet"), _read_arrow(cache + "_pool.parquet")
    d = os.path.join(data_dir, split)
    s1 = read_tsv(os.path.join(d, f"{split}_source1.tsv"))
    s2 = read_tsv(os.path.join(d, f"{split}_source2.tsv"))
    s3 = read_tsv(os.path.join(d, f"{split}_source3.tsv"))
    s2["src"], s3["src"] = np.int8(2), np.int8(3)
    pool = pd.concat([s2, s3], ignore_index=True)
    s1["src"] = np.int8(1)
    s1 = normalize_df(s1, n_jobs)
    pool = normalize_df(pool, n_jobs)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        s1.to_parquet(cache + "_s1.parquet")
        pool.to_parquet(cache + "_pool.parquet")
    return _compact(s1), _compact(pool)


def load_gt(path):
    gt = read_tsv(path)
    rows = [(s, m) for s, ms in zip(gt.source1_entity_id, gt.matched_entity_ids) for m in ms.split(",") if m]
    return gt, pd.DataFrame(rows, columns=["s1_id", "pool_id"])

# ============================================================
# CELL 10 [code]
# ============================================================
%%writefile src/prerank.py
"""Learned candidate pruning (stage 0): cheap blocking-level features only (S1-side context),
so it can run inside every blocking chunk."""
import numpy as np
import pandas as pd

from .blocking import _group_rank

PR_COLS = ["bs_name", "bs_addr", "bs_cross", "bs_nkeys", "bs_nnk", "bs_nak", "bs_nxk", "bs_total",
           "r_total", "r_name", "r_addr", "r_cross", "q_total", "q_name", "q_addr", "q_cross", "n_s1"]


def prerank_features(b):
    g = b.s1_idx.values
    X = pd.DataFrame({c: b[c].values.astype(np.float32) for c in
                      ["bs_name", "bs_addr", "bs_cross", "bs_nkeys", "bs_nnk", "bs_nak", "bs_nxk"]})
    X["bs_total"] = X.bs_name + X.bs_addr + X.bs_cross
    for k, c in [("total", "bs_total"), ("name", "bs_name"), ("addr", "bs_addr"), ("cross", "bs_cross")]:
        v = X[c].values
        X["r_" + k] = _group_rank(g, v).astype(np.float32)
        mx = pd.Series(v).groupby(g).transform("max").values
        X["q_" + k] = (v / (mx + 1e-6)).astype(np.float32)
    X["n_s1"] = pd.Series(g).groupby(g).transform("size").values.astype(np.float32)
    cols = list(PR_COLS)
    if "bs_cos" in b:                  # cosine-normalized blocking scores (--cos)
        for c in ("bs_cos", "bs_ncos", "bs_acos"):
            v = b[c].values.astype(np.float32)
            X[c] = v
            X["r_" + c] = _group_rank(g, v).astype(np.float32)
            X["q_" + c] = (v / (pd.Series(v).groupby(g).transform("max").values + 1e-6)).astype(np.float32)
            cols += [c, "r_" + c, "q_" + c]
    if "bs_prior" in b:                # reserved-slot records (--k-prior): code + rank within their code
        pc = b.bs_prior.values.astype(np.float32)
        sc = np.where(pc == 1, X.bs_name.values, np.where(pc == 2, X.bs_addr.values, -1.0)).astype(np.float32)
        X["bs_prior"] = pc
        X["r_prior"] = np.where(pc > 0, _group_rank(g, sc), 99).astype(np.float32)
        cols += ["bs_prior", "r_prior"]
    return X[cols]


def make_pruner(model, k_keep, p_keep=None):
    """Keep top-k_keep per S1 by pre-rank score (and anything with score >= p_keep)."""
    def prune(b):
        if len(b) == 0:
            return b
        s = model.predict(prerank_features(b), num_threads=0).astype(np.float32)
        r = _group_rank(b.s1_idx.values, s)
        keep = r < k_keep
        if p_keep is not None:
            keep |= s >= p_keep
        out = b[keep].copy()
        out["pr_score"] = s[keep]
        out["pr_rank"] = r[keep].astype(np.float32)
        return out
    return prune

# ============================================================
# CELL 11 [code]
# ============================================================
%%writefile src/run.py
"""End-to-end pipeline v2.

data -> normalize -> key index (cap 1200 single / 600 composite)
     -> blocking pass on a small S1 slice (wide K) -> train pre-ranker
     -> blocking over ALL S1 with learned pruning (top-K per S1)          [train and test]
     -> noise-model dictionaries learned on S1 group D (labels, out-of-model-sample)
     -> features on S1 group M -> LightGBM (holdout = part of M) -> threshold on holdout macro F0.5
     -> test features in chunks -> predict -> one-to-one selection -> TSVs

Train S1 roles by hash h in [0, 10000):
  h <  PR_END            : pre-ranker training slice (blocking-level labels only)
  h in [M_START, M_END)  : model training (last `holdout` share = holdout)
  h >= D_START           : noise-dictionary learning
Every stage caches to --work; delete a file there to redo it.
"""
import argparse
import gc
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

from .prep import load_split, load_gt
from .blocking import KeyIndex, generate_candidates
from .prerank import prerank_features, make_pruner, PR_COLS
from .features import build_idf, cheap_features, string_features, string_feature_cols, name_oov
from .noise import LogOddsCounter, EditCounter, DeltaCounter, frequent_tokens
from .matching import train_lgb, feat_cols, select, f05_macro
from .select import expected_f_select

T0 = time.time()


def rss_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return -1.0


def trim():
    """Return freed heap to the OS (glibc): after deleting the 88M-row candidate frame the RSS otherwise stays up."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def log(*a):
    print(f"[{time.time() - T0:7.0f}s | {rss_gb():5.1f}GB]", *a, flush=True)


def s1_hash(ids):
    return (pd.util.hash_array(np.asarray(ids, dtype=object)) % 10_000).astype(np.int32)


def labels(c, s1, pool, gtp):
    """Integer-key join (no 77M-row string MultiIndex): pair key = s1_idx * 2^24 + pool_idx."""
    si = pd.Index(s1.entity_id.values).get_indexer(gtp.s1_id.values)
    pj = pd.Index(pool.entity_id.values).get_indexer(gtp.pool_id.values)
    ok = (si >= 0) & (pj >= 0)
    gk = np.unique((si[ok].astype(np.int64) << 24) | pj[ok].astype(np.int64))
    ck = (c.s1_idx.values.astype(np.int64) << 24) | c.pool_idx.values.astype(np.int64)
    return np.isin(ck, gk, assume_unique=False).astype(np.int8)


def write_lists(path, s1_ids, s1_idx, pool_ids_col, pool_idx, col, chunk=200_000):
    """Stream one row per S1 (empty list allowed). Works on integer indices; strings are only
    materialized per chunk of S1 rows. Returns a bool array: S1 row has >= 1 id."""
    s1_ids = np.asarray(s1_ids, dtype=object)
    key = (np.asarray(s1_idx, np.int64) << 24) | np.asarray(pool_idx, np.int64)
    key = np.unique(key)                       # sorted by s1, dedup
    ks, kp = (key >> 24).astype(np.int64), (key & ((1 << 24) - 1)).astype(np.int64)
    bounds = np.searchsorted(ks, np.arange(len(s1_ids) + 1))
    has = np.diff(bounds) > 0
    with open(path, "w") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for r0 in range(0, len(s1_ids), chunk):
            r1 = min(r0 + chunk, len(s1_ids))
            b0, b1 = bounds[r0], bounds[r1]
            pid = np.asarray(pool_ids_col.take(kp[b0:b1]), dtype=object)
            lines = []
            for r in range(r0, r1):
                lo, hi = bounds[r] - b0, bounds[r + 1] - b0
                lines.append(f"{s1_ids[r]}\t{','.join(pid[lo:hi])}\n")
            f.write("".join(lines))
    return has


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="output")
    ap.add_argument("--cap", type=int, default=1200, help="pool df cap for single-token keys")
    ap.add_argument("--cap-composite", type=int, default=600, help="pool df cap for composite keys")
    ap.add_argument("--df-scale", type=float, default=1.0)
    ap.add_argument("--wide", default="150,40,40,40", help="k_total,k_name,k_addr,k_cross before pruning")
    ap.add_argument("--k-keep", type=int, default=35, help="candidates kept per S1 after pre-ranking")
    ap.add_argument("--pr-frac", type=float, default=0.05)
    ap.add_argument("--m-frac", type=float, default=0.30, help="share of train S1 used for the model")
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--d-frac", type=float, default=0.30, help="share of train S1 used for noise dictionaries")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--predict-chunk", type=int, default=3_000_000)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--fmt", action="store_true", help="add raw-format fingerprint features (FORENSICS-01)")
    ap.add_argument("--extra-keys", action="store_true", help="joined-name suffix + name-pair x number keys")
    ap.add_argument("--dump-xy", action="store_true", help="save the model feature matrix (for learner experiments)")
    ap.add_argument("--numdiff", action="store_true", help="house-number difference structure features (ERR-01)")
    ap.add_argument("--skelnoise", action="store_true", help="noise model on phonetic-skeleton tokens (cross-script)")
    ap.add_argument("--edits", action="store_true", help="one-letter edit class log-odds: typo vs distractor edit")
    ap.add_argument("--phon", action="store_true", help="sound-key name keys + features (native-script India names)")
    ap.add_argument("--numsign", action="store_true", help="signed house-number delta + log-odds of the exact delta")
    ap.add_argument("--morph", action="store_true", help="with --edits: 2-3 letter suffix/prefix appends as edit classes")
    ap.add_argument("--k-prior", type=int, default=0, help="reserved wide slots per S1 for empty-address (by name) "
                    "and invented-name (by address) pool records")
    ap.add_argument("--edits-ctx", action="store_true", help="--edits statistics split by lone vs accompanied edit")
    ap.add_argument("--oov", action="store_true", help="pool-name tokens unseen in the country's S1 names (invented names)")
    ap.add_argument("--cos", type=int, default=0, help="cosine-normalized blocking: keep top-N by cosine in the wide "
                                                        "set and feed cosine scores to the pre-ranker (0 = off)")
    ap.add_argument("--d-max-pairs", type=int, default=15_000_000, help="max candidate pairs for noise dictionaries")
    ap.add_argument("--stage", default="all", choices=["all", "train", "test-cands", "predict"],
                    help="train: stop after model.pkl | test-cands: pre-ranker + test blocking only | "
                         "predict: needs model.pkl, prerank.txt, test_cands.parquet in --work")
    ap.add_argument("--extra-work", nargs="*", default=[],
                    help="dirs whose files are copied into --work first (outputs of other jobs)")
    ap.add_argument("--countries", default="", help="predict only these S1 countries (comma list); other rows are "
                    "left empty - combine the per-country outputs with scripts/combine_by_country.py")
    ap.add_argument("--dict-countries", default="", help="learn noise dictionaries only from these S1 countries "
                    "(simulates an unseen country; experiments only)")
    ap.add_argument("--fr-bundle", action="store_true", help="unseen-country normalization bundle (src/france.py) on "
                    "the test rows of --fix-countries: learned admin components, extra legal forms, own-country token")
    ap.add_argument("--fix-countries", default="France")
    ap.add_argument("--retrain", action="store_true",
                    help="with --extra-work: reuse imported caches and candidates but NOT an imported model "
                         "(retrain the matcher with the current code; valid only if normalization/blocking are unchanged)")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    if a.extra_keys:
        import src.blocking as _blk
        _blk.MORE_KEYS = True
    if a.phon:
        import src.blocking as _blk
        _blk.PHON_KEYS = True
    if a.edits_ctx:
        import src.noise as _nz
        _nz.EDIT_CTX = a.edits = True
    if a.morph:
        import src.noise as _nz
        _nz.MORPH = a.edits = True
    import shutil
    stale = {"model.pkl", "holdout_preds.parquet", "train_report.json"} if a.retrain else set()
    stale |= {"test_scores.parquet"}             # a job's own output: never imported (would be written through a link)
    if a.fr_bundle:
        stale |= {"test_cands.parquet"}          # candidates change with the re-normalized rows
    for d in a.extra_work:
        for f in os.listdir(d):
            if f in stale:
                continue
            src, dst = os.path.abspath(os.path.join(d, f)), os.path.join(a.work, f)
            if os.path.isfile(src) and not os.path.lexists(dst):   # first job listed wins (T and S share train files)
                os.symlink(src, dst) if f.endswith(".parquet") else shutil.copy(src, dst)
                log("imported", f, "from", d)
    W = lambda f: os.path.join(a.work, f)
    kt, kn, ka, kx = map(int, a.wide.split(","))
    wide = dict(k_total=kt, k_name=kn, k_addr=ka, k_cross=kx, k_cos=a.cos, k_prior=a.k_prior)

    def with_prior(s1, pool):
        """Pool prior codes for --k-prior (1 empty address, 2 invented name); also caches the --oov columns."""
        if (a.k_prior or a.oov) and "oov_f" not in pool:
            pool["oov_f"], pool["oov_n"] = name_oov(s1, pool)
        if not a.k_prior:
            return None
        empty = np.asarray(pool.a_clean.str.len().fillna(0).values) == 0
        inv = (np.asarray(pool.oov_f.values) >= 1.0) & (np.asarray(pool.oov_n.values) > 0)
        code = np.where(empty, 1, np.where(inv, 2, 0)).astype(np.int8)
        log(f"pool prior: empty address {(code == 1).mean():.3f}, invented name {(code == 2).mean():.3f}")
        return code
    PR_END = int(a.pr_frac * 10_000)
    M_START, M_END = PR_END, PR_END + int(a.m_frac * 10_000)
    D_START = 10_000 - int(a.d_frac * 10_000)
    assert D_START >= M_END, "dictionary S1 group must not overlap the model group"

    def ensure_prerank(s1, pool, gtp, h, ix):
        """Deterministic pre-ranker (same data + params -> same model in every job)."""
        if os.path.exists(W("prerank.txt")):
            return lgb.Booster(model_file=W("prerank.txt"))
        cw = generate_candidates(s1, pool, index=ix, s1_subset=h < PR_END, verbose=False, prior=with_prior(s1, pool), **wide)
        yw = labels(cw, s1, pool, gtp)
        ntrue = gtp.s1_id.isin(set(s1.entity_id.values[h < PR_END])).sum()
        log(f"pre-ranker slice: {len(cw):,} wide pairs, wide recall {yw.sum() / ntrue:.4f}")
        Xw = prerank_features(cw)
        prm = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, verbose=-1,
                   num_threads=4, deterministic=True, force_row_wise=True, seed=7)
        from .blocking import _group_rank
        half = (cw.s1_idx.values % 2) == 0
        m0 = lgb.train(prm, lgb.Dataset(Xw[half], yw[half]), 200)
        r = _group_rank(cw.s1_idx.values[~half], m0.predict(Xw[~half]))
        nt_half = gtp.s1_id.isin(set(s1.entity_id.values[np.unique(cw.s1_idx.values[~half])])).sum()
        for K in (20, 30, 35, 40, 60):
            log(f"  pre-ranker recall@{K} {yw[~half][r < K].sum() / nt_half:.4f} (held-out half of slice)")
        pr = lgb.train(prm, lgb.Dataset(Xw, yw), 200)
        pr.save_model(W("prerank.txt"))
        return pr

    if a.stage == "test-cands" and not os.path.exists(W("prerank.txt")):
        s1, pool = load_split(a.data, "train", a.work, a.jobs)
        _, gtp = load_gt(os.path.join(a.data, "train", "train_ground_truth.tsv"))
        ix = KeyIndex(s1, pool, a.cap, a.df_scale, a.jobs, cap_composite=a.cap_composite)
        ensure_prerank(s1, pool, gtp, s1_hash(s1.entity_id.values), ix)
        del s1, pool, ix
        gc.collect()

    # ================================================================ TRAIN
    if a.stage in ("all", "train") and not os.path.exists(W("model.pkl")):
        s1, pool = load_split(a.data, "train", a.work, a.jobs)
        log("train loaded", len(s1), len(pool))
        _, gtp = load_gt(os.path.join(a.data, "train", "train_ground_truth.tsv"))
        h = s1_hash(s1.entity_id.values)

        if not os.path.exists(W("train_cands.parquet")):
            ix = KeyIndex(s1, pool, a.cap, a.df_scale, a.jobs, cap_composite=a.cap_composite)
            pr = ensure_prerank(s1, pool, gtp, h, ix)
            c = generate_candidates(s1, pool, index=ix, prerank=make_pruner(pr, a.k_keep), prior=with_prior(s1, pool), **wide)
            del ix
            gc.collect()
            c.to_parquet(W("train_cands.parquet"))
        c = pd.read_parquet(W("train_cands.parquet"))
        y_all = labels(c, s1, pool, gtp)
        log(f"train cands {len(c):,}  per S1 {len(c) / len(s1):.1f}  blocking recall {y_all.sum() / len(gtp):.4f}")

        # ---- noise-model dictionaries from group D (not used for model fitting)
        hc = h[c.s1_idx.values]
        d = hc >= D_START
        if a.dict_countries:              # simulation of an unseen country: dictionaries from these countries only
            dc = {x.strip().lower() for x in a.dict_countries.split(",")}
            d &= np.isin(np.char.lower(np.asarray(s1.country.values, dtype=str))[c.s1_idx.values], list(dc))
        dsel = np.flatnonzero(d)
        if len(dsel) > a.d_max_pairs:   # plenty for token statistics; bounds time and memory
            dsel = np.sort(np.random.default_rng(0).choice(dsel, a.d_max_pairs, replace=False))
        cnt_n, cnt_a, cnt_s = LogOddsCounter(), LogOddsCounter(), LogOddsCounter()
        cnt_en, cnt_ea = EditCounter(), EditCounter()
        cnt_ns = DeltaCounter()
        di_all, dj_all, dy_all = c.s1_idx.values[dsel], c.pool_idx.values[dsel], y_all[dsel]
        for k in range(0, len(dsel), 2_000_000):     # stream: never more than 2M pairs of strings alive
            di, dj, dy = di_all[k:k + 2_000_000], dj_all[k:k + 2_000_000], dy_all[k:k + 2_000_000]
            ct = s1.country.array.take(di)
            cnt_n.update(s1.rn.array.take(di), pool.rn.array.take(dj), dy, ct)
            cnt_a.update(s1.ra.array.take(di), pool.ra.array.take(dj), dy, ct)
            if a.skelnoise:
                cnt_s.update(s1.nsk.array.take(di), pool.nsk.array.take(dj), dy, ct)
            if a.edits:
                cnt_en.update(s1.rn.array.take(di), pool.rn.array.take(dj), dy)
                cnt_ea.update(s1.ra.array.take(di), pool.ra.array.take(dj), dy)
            if a.numsign:
                cnt_ns.update(s1.nums.array.take(di), pool.nums.array.take(dj), dy, ct)
        nadd, ndrop = cnt_n.result()
        aadd, adrop = cnt_a.result()
        sadd, sdrop = cnt_s.result() if a.skelnoise else ({}, {})
        edits_t = (cnt_en.result(), cnt_ea.result()) if a.edits else None
        numsign_t = cnt_ns.result() if a.numsign else None
        if a.numsign:
            log("signed number delta log-odds:", {k: round(numsign_t[k], 2) for k in ("us|1", "us|2", "us|-1", "us|-2", "us|7",
                                                                                       "india|1", "india|-1") if k in numsign_t})
        if a.edits:
            show = {k: round(v, 2) for k, v in edits_t[0].items() if k.split("|")[0][-3:] in ("ins", "sub") and k.count("|") == 1}
            log(f"edit classes: name {len(edits_t[0])} addr {len(edits_t[1])}; name (op|position) log-odds {show}")
        del cnt_n, cnt_a, cnt_s, cnt_en, cnt_ea, cnt_ns, di_all, dj_all, dy_all, dsel, d
        gc.collect()
        log(f"noise dicts from {min(int((hc >= D_START).sum()), a.d_max_pairs):,} D pairs: name add {len(nadd)} drop {len(ndrop)} | addr add {len(aadd)} drop {len(adrop)}")
        idf_n = build_idf([s1.nsk, pool.nsk])
        idf_a = build_idf([s1.ask, pool.ask])
        fq = (frequent_tokens([s1.rn, pool.rn]), frequent_tokens([s1.ra, pool.ra]))
        tables = (*idf_n, *idf_a, nadd, ndrop, aadd, adrop, *fq)
        extras = tuple(e for e, on in (("numdiff", a.numdiff), ("skel", a.skelnoise), ("edits", a.edits),
                                        ("numsign", a.numsign), ("phon", a.phon)) if on)
        skel = (sadd, sdrop, frequent_tokens([s1.nsk, pool.nsk])) if a.skelnoise else None

        # ---- features for group M (context computed over ALL candidates, materialized for M only)
        m = (hc >= M_START) & (hc < M_END)
        va_all = hc >= M_START + (1 - a.holdout) * (M_END - M_START)
        # order M rows as [train..., holdout...] so both are contiguous views (no copies)
        idx = np.flatnonzero(m)
        idx = idx[np.argsort(va_all[idx], kind="stable")]
        if a.oov:
            with_prior(s1, pool)
        cheap = cheap_features(c, s1, pool, rows=idx, fmt=a.fmt, oov=a.oov)
        cm, y = c.iloc[idx].reset_index(drop=True), y_all[idx]
        ntr = int((~va_all[idx]).sum())
        del y_all, c, hc, va_all, m
        trim()
        log(f"features on {len(cm):,} model pairs")
        # one preallocated training matrix: the cheap block is copied in and freed, the string features are
        # written straight into their columns (2x model rows x 120+ features would otherwise peak ~8 GB higher)
        scols = string_feature_cols(extras)
        F = scols + list(cheap.columns)
        X = np.empty((len(cm), len(F)), np.float32)
        X[:, len(scols):] = cheap.values
        del cheap
        trim()
        log(f"training matrix allocated {X.shape[0]:,} x {X.shape[1]} ({X.nbytes / 1e9:.1f} GB)")
        string_features(cm, s1, pool, tables, a.jobs, chunk=250_000, extras=extras, skel=skel, edits=edits_t,
                        numsign=numsign_t, out=X[:, :len(scols)])
        gc.collect()
        log(f"feature matrix {X.shape[0]:,} x {X.shape[1]} ({X.nbytes / 1e9:.1f} GB)")
        if a.dump_xy:
            np.save(W("xy_X.npy"), X)
            np.save(W("xy_y.npy"), y)
            pd.DataFrame({"s1_idx": cm.s1_idx.values, "pool_idx": cm.pool_idx.values}).to_parquet(W("xy_keys.parquet"))
            json.dump(dict(F=F, ntr=int(ntr)), open(W("xy_meta.json"), "w"))
        ids, pids = s1.entity_id.values, pool.entity_id.values
        del s1, pool
        trim()
        log(f"fit lgb: train {ntr:,}  holdout {len(cm) - ntr:,}  pos rate {y.mean():.3f}  feats {len(F)}")
        model = train_lgb(X[:ntr], y[:ntr], rounds=a.rounds, valid=(X[ntr:], y[ntr:]),
                          params=dict(learning_rate=a.lr), feature_name=F)
        va = np.zeros(len(cm), bool)
        va[ntr:] = True
        p = model.predict(X[ntr:], num_threads=0)
        hcut = M_START + (1 - a.holdout) * (M_END - M_START)
        hs1 = np.flatnonzero((h >= hcut) & (h < M_END))          # ALL holdout S1 (also those w/o candidates)
        # integer-keyed truth for fast scoring (string merges made this loop take ~16 min in v2)
        gi = pd.Index(ids).get_indexer(gtp.s1_id.values)
        gj = pd.Index(pids).get_indexer(gtp.pool_id.values)
        hmask = np.zeros(len(ids), bool)
        hmask[hs1] = True
        keep_t = (gi >= 0) & (gj >= 0) & hmask[np.maximum(gi, 0)]
        true = pd.DataFrame({"s1": gi[keep_t], "pool": gj[keep_t]})
        pd.DataFrame({"s1_idx": cm.s1_idx.values[va], "pool_idx": cm.pool_idx.values[va], "y": y[va], "p": p}) \
            .to_parquet(W("holdout_preds.parquet"))
        hi, hj = cm.s1_idx.values[va], cm.pool_idx.values[va]

        def score(sel):
            return f05_macro(sel[["s1", "pool"]], true, hs1)

        best = (-1, 0.5)
        for thr in np.arange(0.50, 0.931, 0.025):
            f, info = score(select(hi, hj, p, thr))
            log(f"  thr {thr:.3f}  F0.5 {f:.4f}  {info}")
            best = max(best, (f, thr))
        best_ef = (-1, None)
        for mp_ in (0.0, 0.05, 0.1):
            for pw in (1.0, 1.25, 1.5, 2.0):
                f, _ = score(expected_f_select(hi, hj, p, miss_prior=mp_, power=pw))
                log(f"  expected-F  miss_prior {mp_}  power {pw}  F0.5 {f:.4f}")
                best_ef = max(best_ef, (f, (mp_, pw)), key=lambda x: x[0])
        method = ("expected_f", best_ef[1]) if best_ef[0] > best[0] + 1e-4 else ("threshold", best[1])
        log(f"SELECTION: threshold best {best[0]:.4f} @ {best[1]:.3f} | expected-F best {best_ef[0]:.4f} @ {best_ef[1]}"
            f" -> using {method}")
        log(f"HOLDOUT F0.5 {best[0]:.4f} at threshold {best[1]:.3f}")
        imp = pd.Series(model.feature_importance("gain"), F).sort_values(ascending=False)
        log("top features", imp.head(20).round(0).to_dict())
        with open(W("model.pkl"), "wb") as f:
            pickle.dump(dict(model=model, feats=F, thr=float(best[1]), holdout_f05=max(best[0], best_ef[0]),
                             method=method, noise=(nadd, ndrop, aadd, adrop), fmt=a.fmt, extras=extras,
                             skel_dicts=(sadd, sdrop), edit_tables=edits_t, edit_ctx=a.edits_ctx, morph=a.morph, oov=a.oov,
                             numsign_table=numsign_t), f)
        json.dump(dict(holdout_f05=max(best[0], best_ef[0]), holdout_f05_threshold=best[0], threshold=float(best[1]),
                       holdout_f05_expected_f=best_ef[0], expected_f_params=best_ef[1], method=method,
                       best_iter=model.best_iteration,
                       features=F, args=vars(a)), open(W("train_report.json"), "w"), indent=1)
        del X, cm
        gc.collect()

    if a.stage == "train":
        log("stage train done")
        return

    # ================================================================ TEST
    s1, pool = load_split(a.data, "test", a.work, a.jobs)
    log("test loaded", len(s1), len(pool), s1.country.value_counts().to_dict())
    if a.fr_bundle:
        from .france import apply_bundle
        apply_bundle(a.data, "test", s1, pool, [x.strip() for x in a.fix_countries.split(",") if x.strip()], a.jobs, log)
    only_c = {x.strip().lower() for x in a.countries.split(",") if x.strip()}
    if not os.path.exists(W("test_cands.parquet")):
        pr = lgb.Booster(model_file=W("prerank.txt"))
        ix = KeyIndex(s1, pool, a.cap, a.df_scale, a.jobs, cap_composite=a.cap_composite)
        sub = np.isin(np.char.lower(np.asarray(s1.country.values, dtype=str)), list(only_c)) if only_c else None
        c = generate_candidates(s1, pool, index=ix, prerank=make_pruner(pr, a.k_keep), prior=with_prior(s1, pool),
                                s1_subset=sub, **wide)
        del ix
        gc.collect()
        c.to_parquet(W("test_cands.parquet"))
    c = pd.read_parquet(W("test_cands.parquet"))
    log(f"test cands {len(c):,}  per S1 {len(c) / len(s1):.1f}")
    if a.stage == "test-cands":
        log("stage test-cands done")
        return
    with open(W("model.pkl"), "rb") as f:
        M = pickle.load(f)
    thr = a.threshold if a.threshold is not None else M["thr"]
    if M.get("edit_ctx"):
        import src.noise as _nz
        _nz.EDIT_CTX = True
    if M.get("morph"):
        import src.noise as _nz
        _nz.MORPH = True
    log(f"using threshold {thr:.3f}")
    write_lists(os.path.join(a.out, "candidate_pairs.tsv"), s1.entity_id.values, c.s1_idx.values,
                pool.entity_id.array, c.pool_idx.values, "candidate_entity_ids")
    log("wrote candidate_pairs.tsv")
    if M.get("oov", False) and "oov_f" not in pool:
        pool["oov_f"], pool["oov_n"] = name_oov(s1, pool)
    idf_n = build_idf([s1.nsk, pool.nsk])
    idf_a = build_idf([s1.ask, pool.ask])
    fq = (frequent_tokens([s1.rn, pool.rn]), frequent_tokens([s1.ra, pool.ra]))
    log(f"frequent tokens (fallback gate): name {len(fq[0]):,} addr {len(fq[1]):,}")
    tables = (*idf_n, *idf_a, *M["noise"], *fq)
    extras = M.get("extras", ())
    skel = (*M["skel_dicts"], frequent_tokens([s1.nsk, pool.nsk])) if "skel" in extras else None
    p = np.zeros(len(c), np.float32)
    # Keys are country-scoped, so no candidate crosses countries and the competition context of a pair only
    # involves its own country: building the cheap block one country at a time gives identical features with
    # about half the peak memory (all ~69M test pairs x 50+ columns at once is ~14 GB plus temporaries).
    codes, names = pd.factorize(np.asarray(s1.country.values))
    ctry = codes.astype(np.int16)[c.s1_idx.values]
    done = 0
    only = {x.strip().lower() for x in a.countries.split(",") if x.strip()}
    if only:
        log(f"predicting only {sorted(only)} (other countries left empty)")
    for ci, ct in enumerate(names):
        rows = np.flatnonzero(ctry == ci)
        if not len(rows) or (only and str(ct).lower() not in only):
            continue
        cc = c.iloc[rows].reset_index(drop=True)
        cheap = cheap_features(cc, s1, pool, fmt=M.get("fmt", False), oov=M.get("oov", False))
        ch = cheap.values
        cheap_cols = list(cheap.columns)
        del cheap
        for k in range(0, len(cc), a.predict_chunk):
            sl = slice(k, k + a.predict_chunk)
            S = string_features(cc.iloc[sl], s1, pool, tables, a.jobs, extras=extras, skel=skel, edits=M.get("edit_tables"),
                                numsign=M.get("numsign_table"))
            X = np.hstack([S.values, ch[sl]])
            assert list(S.columns) + cheap_cols == M["feats"], "feature order mismatch"
            p[rows[sl]] = M["model"].predict(X, num_threads=0)
            done += len(X)
            del S, X
            log(f"  predicted {done:,}/{len(c):,} ({ct})")
        del ch, cc, rows
        gc.collect()
    if os.path.islink(W("test_scores.parquet")):
        os.unlink(W("test_scores.parquet"))
    pd.DataFrame({"s1_idx": c.s1_idx.values, "pool_idx": c.pool_idx.values, "p": p}).to_parquet(W("test_scores.parquet"))
    method = M.get("method", ("threshold", thr))
    if a.threshold is not None:
        method = ("threshold", a.threshold)
    if method[0] == "expected_f":
        sel = expected_f_select(c.s1_idx.values, c.pool_idx.values, p, miss_prior=method[1][0], power=method[1][1])
    else:
        sel = select(c.s1_idx.values, c.pool_idx.values, p, method[1])
    log(f"selection method {method}")
    n = write_lists(os.path.join(a.out, "matching_results.tsv"), s1.entity_id.values, sel.s1.values,
                    pool.entity_id.array, sel.pool.values, "matched_entity_ids")
    log(f"DONE: {len(s1):,} S1 rows, {n.mean():.3f} with >=1 match, {len(sel):,} matched pairs "
        f"({len(sel) / len(s1):.2f} per S1)")
    log("by country (share with match):", pd.Series(n).groupby(np.asarray(s1.country)).mean().round(3).to_dict())


if __name__ == "__main__":
    main()

# ============================================================
# CELL 12 [code]
# ============================================================
%%writefile src/select.py
"""Per-S1 set selection maximizing expected F0.5.

F0.5 for one S1 with k predictions, TP hits and T true matches: 1.25*TP / (k + 0.25*T)
(1 if k = T = 0). With calibrated candidate probabilities q (after the one-to-one rule),
E[F] for predicting the top-k is approximated by 1.25*E[TP_k] / (k + 0.25*E[T]),
and the empty set scores P(T = 0) = prod(1 - q). We pick the best k per S1.
"""
import numpy as np
import pandas as pd


def one_to_one(s1_idx, pool_idx, p):
    df = pd.DataFrame({"s1": s1_idx, "pool": pool_idx, "p": p})
    best = df.groupby("pool").p.transform("max")
    return df[df.p >= best].drop_duplicates("pool")


def expected_f_select(s1_idx, pool_idx, p, miss_prior=0.0, min_p=0.0, power=1.0):
    """miss_prior: expected number of true matches not among candidates (blocking misses) per S1."""
    df = one_to_one(s1_idx, pool_idx, p)
    q_all = df.p.values ** power
    df = df.assign(q=q_all).sort_values(["s1", "q"], ascending=[True, False])
    g = df.s1.values
    q = df.q.values
    first = np.r_[True, g[1:] != g[:-1]]
    starts = np.flatnonzero(first)
    grp = np.cumsum(first) - 1
    k = np.arange(len(g)) - starts[grp] + 1
    csum = np.cumsum(q)
    etp = csum - np.r_[0, csum][starts][grp]
    et = np.add.reduceat(q, starts)[grp] + miss_prior
    ef = 1.25 * etp / (k + 0.25 * et)
    logq0 = np.add.reduceat(np.log1p(-np.clip(q, 0, 1 - 1e-7)), starts)
    p_empty = np.exp(logq0) * np.exp(-miss_prior)
    # best k per group
    best_ef = pd.Series(ef).groupby(grp).transform("max").values
    kbest = pd.Series(np.where(ef >= best_ef, k, 10 ** 9)).groupby(grp).transform("min").values
    choose = (k <= kbest) & (best_ef[...] > p_empty[grp]) & (q >= min_p)
    return df[choose][["s1", "pool", "p"]]

# ============================================================
# CELL 13 [code]
# ============================================================
%%writefile src/translit.py
"""Romanization to ASCII without GPL dependencies.

Indic scripts: the nine Brahmic blocks in Unicode (Devanagari .. Malayalam) share the
ISCII layout, so one offset -> Latin table covers all of them. Handles the inherent
vowel, virama, anusvara (n / m before labials), nukta, Malayalam chillus.
Latin with diacritics: NFKD + drop combining marks. Anything else: anyascii (ISC).
"""
import unicodedata

from anyascii import anyascii

_BLOCKS = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)
_TAMIL, _MALAYALAM = 0x0B80, 0x0D00

# offset within block -> latin; consonants carry an inherent 'a'
_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
    0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
    0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
    0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_VOWEL = {  # independent vowels
    0x04: "a", 0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
    0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o",
    0x14: "au", 0x60: "ri", 0x61: "li", 0x72: "i", 0x73: "u",
}
_SIGN = {  # dependent vowel signs (replace the inherent 'a')
    0x3A: "e", 0x3B: "e", 0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri",
    0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
    0x4C: "au", 0x4F: "aw", 0x62: "li", 0x63: "li", 0x56: "ai", 0x57: "au",
}
_NASAL = {0x01, 0x02, 0x70}          # candrabindu, anusvara, gurmukhi tippi
_MARKS = {0x3C, 0x4D, 0x71, 0x51, 0x52, 0x53, 0x54, 0x55, 0x3D}   # nukta, virama, addak, stress, avagraha
_VIRAMA, _NUKTA = 0x4D, 0x3C
_NUKTA_OF = {"ph": "f", "j": "z", "k": "q", "kh": "kh", "g": "g", "d": "r", "dh": "rh"}
_EXTRA = {  # script-specific code points (absolute)
    0x09F0: "r", 0x09F1: "w", 0x0B71: "w", 0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l",
    0x0D7E: "l", 0x0D7F: "k", 0x0D54: "m", 0x0D55: "y", 0x0D56: "l", 0x0D4E: "r", 0x0C58: "ts",
    0x0C59: "dz", 0x0B83: "",
}
_LATIN = {"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss", "đ": "d",
          "Đ": "D", "ł": "l", "Ł": "L", "ı": "i", "þ": "th", "ð": "d", "°": " ", "’": "'", "‘": "'",
          "“": '"', "”": '"', "–": "-", "—": "-", " ": " "}
_ZW = {"‌", "‍", "​", "﻿"}
_LABIAL = {"p", "b", "m", "f"}


def _block(cp):
    if 0x0900 <= cp < 0x0D80:
        base = cp & ~0x7F
        return base, cp - base
    return None, None


def _indic_run(chars, i):
    """Transliterate a maximal run of Indic code points starting at i. Returns (text, next_i)."""
    out = []           # list of syllable pieces
    pending = None     # consonant waiting for its vowel
    n = len(chars)
    while i < n:
        ch = chars[i]
        if ch in _ZW:
            i += 1
            continue
        cp = ord(ch)
        base, off = _block(cp)
        if base is None:
            break
        if cp in _EXTRA:
            if pending:
                out.append(pending + "a")
                pending = None
            out.append(_EXTRA[cp])
        elif off in _CONS:
            if pending:
                out.append(pending + "a")
            pending = _CONS[off]
            # Malayalam റ്റ (rra virama rra) is "tt"
            if base == _MALAYALAM and off == 0x31 and i + 2 < n and ord(chars[i + 1]) == 0x0D4D \
                    and ord(chars[i + 2]) == 0x0D31:
                pending = "tt"
                i += 2
        elif off in _SIGN:
            out.append((pending or "") + _SIGN[off])
            pending = None
        elif off == _VIRAMA:
            if pending:
                out.append(pending)
                pending = None
        elif off == _NUKTA:
            if pending:
                pending = _NUKTA_OF.get(pending, pending)
        elif off in _NASAL:
            if pending:
                out.append(pending + "a")
                pending = None
            out.append("\x00" if off == 0x02 else "\x01")   # anusvara / other nasal, resolved below
        elif off == 0x03:             # visarga
            if pending:
                out.append(pending + "a")
                pending = None
            out.append("h")
        elif off in _VOWEL:
            if pending:
                out.append(pending + "a")
                pending = None
            out.append(_VOWEL[off])
        elif 0x66 <= off <= 0x6F:     # digits
            if pending:
                out.append(pending + "a")
                pending = None
            out.append(str(off - 0x66))
        elif off in (0x64, 0x65):     # danda
            if pending:
                out.append(pending + "a")
                pending = None
            out.append(" ")
        elif off in _MARKS:
            pass
        else:
            if pending:
                out.append(pending + "a")
                pending = None
        i += 1
    if pending:
        out.append(pending)           # word-final schwa deletion
    s = "".join(out)
    if "\x00" in s or "\x01" in s:
        s = "".join(_nasal(s, j) if c in "\x00\x01" else c for j, c in enumerate(s))
    return s, i


def _nasal(s, j):
    nxt = s[j + 1:j + 3]
    if not nxt:
        return "ng" if s[j] == "\x00" else "n"   # word-final anusvara: 'marketing', 'building'
    return "m" if (nxt[0] in _LABIAL and nxt != "ph") else "n"


def to_ascii(s: str) -> str:
    if not s or s.isascii():
        return s
    chars, out, i = list(s), [], 0
    while i < len(chars):
        ch = chars[i]
        cp = ord(ch)
        if cp < 128:
            out.append(ch)
            i += 1
        elif 0x0900 <= cp < 0x0D80:
            t, i = _indic_run(chars, i)
            out.append(t)
        elif ch in _ZW:
            i += 1
        else:
            if ch in _LATIN:
                out.append(_LATIN[ch])
            else:
                d = unicodedata.normalize("NFKD", ch)
                base = "".join(c for c in d if not unicodedata.combining(c))
                out.append(base if base.isascii() else anyascii(base))
            i += 1
    return "".join(out)

# ============================================================
# CELL 14 [code]
# ============================================================
%%writefile scripts/ce_kaggle.py
"""Cross-encoder re-scoring on top of the GBDT (GPU job).

1. Train a multilingual pair classifier on train candidate pairs (T job caches), excluding the GBDT holdout S1.
   Positives, hardest negatives by pre-ranker score, random negatives. Raw 'name | address' texts.
2. Re-score the GBDT holdout pairs whose GBDT probability lies in the uncertain band; pick the blend weight and
   threshold that maximize holdout macro F0.5 (all holdout S1 by hash, blocking misses included).
3. Re-score the uncertain test pairs (F job test_scores.parquet), blend, select (best S1 per pool record + threshold),
   write matching_results.tsv; candidate_pairs.tsv is copied from the predict output.

  python scripts/ce_kaggle.py --data <dataset dir> --train-work <T work> --pred-work <predict work: model.pkl,
      holdout_preds.parquet, test_scores.parquet> --test-work <S work: test_norm_*> --cands <predict output dir>
      --out <dir> [--model intfloat/multilingual-e5-small] [--n-pos 400000 --n-hard 600000 --n-rand 200000]"""
import argparse
import json
import math
import os
import pickle
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.getcwd())
from src.prep import read_tsv, load_gt, _read_arrow
from src.run import s1_hash, write_lists
from src.matching import select

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def texts(path_tsvs, ids):
    need = set(ids)
    out = {}
    for p in path_tsvs:
        for ch in pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3, chunksize=1_000_000,
                              usecols=["entity_id", "business_name", "business_address"]):
            ch = ch[ch.entity_id.isin(need)]
            out.update(zip(ch.entity_id, ch.business_name + " | " + ch.business_address))
    return out


def resolve_model(name):
    """HF id if downloadable (Internet on), else a local copy attached under /kaggle/input."""
    import glob
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(name, allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"])
    except Exception as e:                                     # offline
        short = name.split("/")[-1]
        hits = [os.path.dirname(p) for p in glob.glob(f"/kaggle/input/**/{short}*/**/config.json", recursive=True)]
        hits += [os.path.dirname(p) for p in glob.glob(f"/kaggle/input/**/config.json", recursive=True) if short in p]
        if not hits:
            raise RuntimeError(f"cannot download {name} ({e}) and no local copy under /kaggle/input") from e
        return hits[0]


class CE:
    def __init__(self, name, device):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).to(device)
        self.dev = device
        emb = self.m.get_input_embeddings()
        for p in emb.parameters():                 # the multilingual vocabulary table dominates the parameters
            p.requires_grad = False

    def enc(self, a, b, max_len=80):
        e = self.tok(a, b, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return {k: v.to(self.dev) for k, v in e.items()}

    def fit(self, A, B, y, bs=128, lr=5e-5, epochs=1, minutes=None):
        from transformers import get_linear_schedule_with_warmup
        opt = torch.optim.AdamW([p for p in self.m.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
        steps = epochs * math.ceil(len(y) / bs)
        sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        scaler = torch.amp.GradScaler("cuda", enabled=self.dev == "cuda")
        lossf = torch.nn.BCEWithLogitsLoss()
        self.m.train()
        self.t_fit = time.time()
        k = 0
        for ep in range(epochs):
            order = np.random.default_rng(ep + 1000 * getattr(self, 'seed', 0)).permutation(len(y))
            for s in range(0, len(y), bs):
                r = order[s:s + bs]
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.dev == "cuda"):
                    out = self.m(**self.enc([A[i] for i in r], [B[i] for i in r])).logits.squeeze(-1)
                    loss = lossf(out.float(), torch.tensor(y[r], dtype=torch.float32, device=self.dev))
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad()
                if k % 1000 == 0:
                    log(f"  step {k}/{steps} loss {loss.item():.4f}")
                k += 1
                if minutes is not None and time.time() - self.t_fit > minutes * 60:
                    log(f"  training time budget reached at step {k}/{steps}")
                    self.m.eval()
                    return
        self.m.eval()

    def replicas(self):
        """The model on every visible GPU (inference only)."""
        if not hasattr(self, "_reps"):
            import copy
            self._reps = [(self.m, self.dev)]
            if self.dev == "cuda" and torch.cuda.device_count() > 1:
                for g in range(1, torch.cuda.device_count()):
                    self._reps.append((copy.deepcopy(self.m).to(f"cuda:{g}").eval(), f"cuda:{g}"))
                log(f"  inference on {len(self._reps)} GPUs")
        return self._reps

    @torch.no_grad()
    def predict(self, A, B, bs=512):
        out = np.empty(len(A), np.float32)
        reps = self.replicas()
        starts = list(range(0, len(A), bs))
        for i in range(0, len(starts), len(reps)):
            res = []
            for (m, d), s in zip(reps, starts[i:i + len(reps)]):
                e = self.tok(A[s:s + bs], B[s:s + bs], truncation=True, max_length=80, padding=True, return_tensors="pt")
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=d.startswith("cuda")):
                    res.append((s, m(**{k: v.to(d) for k, v in e.items()}).logits.squeeze(-1)))
            for s, z in res:
                out[s:s + bs] = torch.sigmoid(z.float()).cpu().numpy()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--train-work", required=True)
    ap.add_argument("--pred-work", required=True)
    ap.add_argument("--test-work", required=True)
    ap.add_argument("--cands", default=None, help="dir with the predict job's candidate_pairs.tsv; if missing, it is "
                    "rebuilt from test_scores.parquet (the predict job writes both from the same candidate table)")
    ap.add_argument("--scores", nargs="*", default=None, help="dirs with test_scores.parquet (per-country predict jobs); "
                    "merged by max. Default: --pred-work")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--n-pos", type=int, default=400_000)
    ap.add_argument("--n-hard", type=int, default=600_000)
    ap.add_argument("--n-rand", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=128, help="training batch (e5-large on a T4: 32)")
    ap.add_argument("--seed", type=int, default=0, help="sampling / training-order seed (a second run with another seed -> ensemble)")
    ap.add_argument("--train-minutes", type=float, default=None, help="stop training after this many minutes")
    ap.add_argument("--band", type=float, nargs=2, default=[0.01, 0.99])
    ap.add_argument("--force-w", type=float, default=None, help="use this GBDT weight instead of the holdout choice (testing)")
    ap.add_argument("--load", default=None, help="saved ce_model dir: skip training (rerun after a later-stage failure)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = os.environ.get("CE_DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    rep = json.load(open(os.path.join(a.pred_work, "train_report.json")))["args"]
    M = pickle.load(open(os.path.join(a.pred_work, "model.pkl"), "rb"))
    method = M.get("method", ("threshold", M["thr"]))
    log(f"device {dev}; GBDT selection {method}; model {a.model}")

    # ---------------- train pairs
    s1 = _read_arrow(os.path.join(a.train_work, "train_norm_s1.parquet"))[["entity_id"]]
    pool = _read_arrow(os.path.join(a.train_work, "train_norm_pool.parquet"))[["entity_id"]]
    ids, pids = np.asarray(s1.entity_id.values, dtype=object), np.asarray(pool.entity_id.values, dtype=object)
    c = pd.read_parquet(os.path.join(a.train_work, "train_cands.parquet"), columns=["s1_idx", "pool_idx", "pr_score"])
    _, gtp = load_gt(os.path.join(a.data, "train", "train_ground_truth.tsv"))
    gi = pd.Index(ids).get_indexer(gtp.s1_id.values); gj = pd.Index(pids).get_indexer(gtp.pool_id.values)
    ok = (gi >= 0) & (gj >= 0)
    tk = np.unique((gi[ok].astype(np.int64) << 24) | gj[ok])
    y = np.isin((c.s1_idx.values.astype(np.int64) << 24) | c.pool_idx.values, tk).astype(np.int8)
    h = s1_hash(ids)
    pr_end = int(rep["pr_frac"] * 10_000); m_end = pr_end + int(rep["m_frac"] * 10_000)
    hcut = pr_end + (1 - rep["holdout"]) * (m_end - pr_end)
    hold = (h >= hcut) & (h < m_end)
    usable = ~hold[c.s1_idx.values]
    rng = np.random.default_rng(a.seed)
    pos = np.flatnonzero(usable & (y == 1)); neg = np.flatnonzero(usable & (y == 0))
    pos = rng.choice(pos, min(a.n_pos, len(pos)), replace=False)
    ord_ = neg[np.argsort(-c.pr_score.values[neg])]
    hard = ord_[:a.n_hard]
    rnd = rng.choice(ord_[a.n_hard:], min(a.n_rand, len(ord_) - a.n_hard), replace=False)
    rows = np.concatenate([pos, hard, rnd])
    log(f"train pairs {len(rows):,} (pos {len(pos):,} hard {len(hard):,} random {len(rnd):,})")
    tr_dir = os.path.join(a.data, "train")
    if not a.load:
        T1 = texts([f"{tr_dir}/train_source1.tsv"], ids[c.s1_idx.values[rows]])
        T2 = texts([f"{tr_dir}/train_source2.tsv", f"{tr_dir}/train_source3.tsv"], pids[c.pool_idx.values[rows]])
    if a.load:
        ce = CE(a.load, dev)
        log(f"loaded trained model from {a.load}")
    else:
        A = [T1[ids[i]] for i in c.s1_idx.values[rows]]; B = [T2[pids[j]] for j in c.pool_idx.values[rows]]
        ce = CE(resolve_model(a.model), dev)
        ce.seed = a.seed
        torch.manual_seed(a.seed)
        ce.fit(A, B, y[rows].astype(np.float32), bs=a.bs, epochs=a.epochs, minutes=a.train_minutes)
        del A, B
        ce.m.save_pretrained(os.path.join(a.out, "ce_model")); ce.tok.save_pretrained(os.path.join(a.out, "ce_model"))
        log("saved ce_model")
    del c

    # ---------------- holdout: blend weight + threshold
    hp = pd.read_parquet(os.path.join(a.pred_work, "holdout_preds.parquet"))
    lo, hi = a.band
    band = np.flatnonzero((hp.p.values >= lo) & (hp.p.values <= hi))
    H1 = texts([f"{tr_dir}/train_source1.tsv"], ids[hp.s1_idx.values[band]])
    H2 = texts([f"{tr_dir}/train_source2.tsv", f"{tr_dir}/train_source3.tsv"], pids[hp.pool_idx.values[band]])
    pc = ce.predict([H1[ids[i]] for i in hp.s1_idx.values[band]], [H2[pids[j]] for j in hp.pool_idx.values[band]])
    log(f"holdout band pairs re-scored: {len(band):,} of {len(hp):,}")
    pd.DataFrame({"row": band, "pc": pc}).to_parquet(os.path.join(a.out, "ce_holdout_band.parquet"))   # for CE ensembles
    hs1 = np.flatnonzero(hold)
    kk = pd.Series(gtp.s1_id.value_counts()).reindex(ids[hs1]).fillna(0).values

    def f05(p, thr):
        d = pd.DataFrame({"s1": hp.s1_idx.values, "pool": hp.pool_idx.values, "p": p, "y": hp.y.values})
        d = d[d.p >= d.groupby("pool").p.transform("max")].drop_duplicates("pool")
        d = d[d.p >= thr]
        tp = d.groupby("s1").y.sum().reindex(hs1).fillna(0).values
        npred = d.groupby("s1").size().reindex(hs1).fillna(0).values
        return np.where((npred == 0) & (kk == 0), 1.0, 1.25 * tp / np.maximum(npred + 0.25 * kk, 1e-9)).mean()

    thrs = np.round(np.arange(0.5, 0.951, 0.025), 3)
    pg = hp.p.values.astype(np.float32)
    best = max(((f05(pg, t), 1.0, t) for t in thrs))
    log(f"holdout GBDT alone: F0.5 {best[0]:.4f} @ {best[2]}")
    for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):     # 0 = cross-encoder alone inside the band
        pb = pg.copy(); pb[band] = w * pg[band] + (1 - w) * pc
        cand = max(((f05(pb, t), w, t) for t in thrs))
        log(f"holdout blend w={w}: F0.5 {cand[0]:.4f} @ {cand[2]}")
        best = max(best, cand)
    if a.force_w is not None:
        pb = pg.copy(); pb[band] = a.force_w * pg[band] + (1 - a.force_w) * pc
        best = max(((f05(pb, t), a.force_w, t) for t in thrs))
    log(f"chosen: GBDT weight {best[1]}, threshold {best[2]}, holdout F0.5 {best[0]:.4f}")
    json.dump(dict(f05=best[0], w=best[1], thr=best[2]), open(os.path.join(a.out, "ce_blend.json"), "w"))

    # ---------------- test
    ts = None
    for d in (a.scores or [a.pred_work]):
        x = pd.read_parquet(os.path.join(d, "test_scores.parquet"))
        if ts is None:
            ts = x
        else:
            assert len(x) == len(ts) and (x.s1_idx.values == ts.s1_idx.values).all(), "different candidate sets"
            ts["p"] = np.maximum(ts.p.values, x.p.values)
    log(f"test scores from {len(a.scores or [a.pred_work])} dir(s): {len(ts):,} pairs")
    t1 = _read_arrow(os.path.join(a.test_work, "test_norm_s1.parquet"))[["entity_id"]]
    tp_ = _read_arrow(os.path.join(a.test_work, "test_norm_pool.parquet"))[["entity_id"]]
    tids, tpids = np.asarray(t1.entity_id.values, dtype=object), np.asarray(tp_.entity_id.values, dtype=object)
    p = ts.p.values.astype(np.float32)
    band = np.flatnonzero((p >= lo) & (p <= hi))
    log(f"test band pairs to re-score: {len(band):,} of {len(p):,}")
    if best[1] < 1.0:
        te = os.path.join(a.data, "test")
        Q1 = texts([f"{te}/test_source1.tsv"], tids[ts.s1_idx.values[band]])
        Q2 = texts([f"{te}/test_source2.tsv", f"{te}/test_source3.tsv"], tpids[ts.pool_idx.values[band]])
        pct = ce.predict([Q1[tids[i]] for i in ts.s1_idx.values[band]], [Q2[tpids[j]] for j in ts.pool_idx.values[band]])
        p[band] = best[1] * p[band] + (1 - best[1]) * pct
        pd.DataFrame({"row": band, "pc": pct}).to_parquet(os.path.join(a.out, "ce_test_band.parquet"))
    pd.DataFrame({"s1_idx": ts.s1_idx.values, "pool_idx": ts.pool_idx.values, "p": p}).to_parquet(os.path.join(a.out, "test_scores_blend.parquet"))
    sel = select(ts.s1_idx.values, ts.pool_idx.values, p, best[2])
    n = write_lists(os.path.join(a.out, "matching_results.tsv"), tids, sel.s1.values, pd.array(tpids, dtype="string"),
                    sel.pool.values, "matched_entity_ids")
    cp = os.path.join(a.cands or "", "candidate_pairs.tsv")
    if a.cands and os.path.isfile(cp):
        shutil.copy(cp, os.path.join(a.out, "candidate_pairs.tsv"))
    else:
        write_lists(os.path.join(a.out, "candidate_pairs.tsv"), tids, ts.s1_idx.values, pd.array(tpids, dtype="string"),
                    ts.pool_idx.values, "candidate_entity_ids")
        log("rebuilt candidate_pairs.tsv from the test scores")
    log(f"DONE: {len(sel):,} matched pairs ({len(sel) / len(tids):.2f} per S1), {n.mean():.3f} of S1 with a match")


if __name__ == "__main__":
    main()

# ============================================================
# CELL 15 [code]
# ============================================================
I = "/kaggle/input"
real = lambda p: os.path.isfile(p) and not os.path.islink(p)
DATA = os.path.dirname(os.path.dirname(glob.glob(f"{I}/**/train/train_ground_truth.tsv", recursive=True)[0]))
TRAIN = [os.path.dirname(p) for p in glob.glob(f"{I}/**/train_cands.parquet", recursive=True) if real(p)
         and real(os.path.join(os.path.dirname(p), "train_norm_s1.parquet")) and real(os.path.join(os.path.dirname(p), "model.pkl"))]
SCORES = sorted({os.path.dirname(p) for p in glob.glob(f"{I}/**/test_scores.parquet", recursive=True) if real(p)})
TEST = [os.path.dirname(p) for p in glob.glob(f"{I}/**/test_norm_s1.parquet", recursive=True) if real(p)]
print("train (T5, model + holdout):", TRAIN, "\nscores (F5):", SCORES, "\ntest (S5):", TEST)
assert len(TRAIN) == 1 and len(SCORES) == 2 and len(TEST) >= 1, "attach ber-v5-T, ber-v5-S and both F5 outputs"
# the predict job's candidate_pairs.tsv (output/ next to work/, or the same folder); none -> rebuilt from test_scores
CANDS = next((d for d in (os.path.join(os.path.dirname(SCORES[0]), "output"), SCORES[0])
              if os.path.isfile(os.path.join(d, "candidate_pairs.tsv"))), "")
print("cands:", CANDS or "none attached -> rebuilt from test_scores.parquet")
TRAIN, PRED, TEST = TRAIN[0], TRAIN[0], TEST[0]
SC = " ".join(f'"{d}"' for d in SCORES)

# ============================================================
# CELL 16 [code]
# ============================================================
!python scripts/ce_kaggle.py --data "$DATA" --train-work "$TRAIN" --pred-work "$PRED" --test-work "$TEST" --cands "$CANDS" --scores {SC} --out /kaggle/working/ce_out --model intfloat/multilingual-e5-base --n-pos 500000 --n-hard 800000 --n-rand 300000 --seed 1 --train-minutes 50 2>&1 | grep --line-buffered -v -i "warning"

# ============================================================
# CELL 17 [code]
# ============================================================
V = (glob.glob(os.path.dirname(DATA) + "/utils/validate_submission.py") or glob.glob(f"{I}/**/validate_submission.py", recursive=True) or [None])[0]
if V:
    r = subprocess.run([sys.executable, V, "--matching", "/kaggle/working/ce_out/matching_results.tsv",
                        "--candidate", "/kaggle/working/ce_out/candidate_pairs.tsv", "--test-dir", f"{DATA}/test"], capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:])
!rm -rf /kaggle/working/ber/src/__pycache__; du -sh /kaggle/working/ce_out/*
