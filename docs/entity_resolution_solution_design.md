# Business Entity Resolution — End-to-End Solution Design
### Amazon ML Challenge 2026

This document is a stage-by-stage engineering design for the challenge, built directly from the uploaded problem statement (treated as the sole authoritative spec) plus external entity-resolution (ER) research used to justify each design choice. Every external claim is attributed to its source; every challenge-specific fact (file formats, scoring formula, rules) is cited to "the spec" and not treated as something external research can override.

**How to read this doc:** Sections 1–9 are the technical design. Sections 10–14 turn that design into an architecture, a build plan, an experiment plan, a risk register, and a compliance sign-off. The very end has the condensed deliverables (stack, diagram, phases, experiment matrix, top innovations, anti-patterns, definition-of-done, submission checklist) for quick reference while building.

No dataset files were provided to this session — only the PDF spec. Every claim about *this* dataset's actual singleton rate, match cardinality distribution, row counts, or noise severity is therefore explicitly flagged **"[EDA-REQUIRED]"**: a number your team must measure, not one this document invents.

---

## 0. Executive Summary — the five decisions that matter most

1. **Blocking is the scoring axis you can most differentiate on.** The spec now scores `candidate_pairs.tsv` and its generating code directly, on top of the leaderboard F₀.₅. Treat blocking as a first-class deliverable, not a throwaway pre-step. Primary method: **top-k TF‑IDF/BM25 retrieval over character n-grams**, which VLDB'23 research (Paulsen, Govind & Doan, "Sparkly") found beat eight state-of-the-art blockers, including deep-learning blockers, while remaining simple, CPU-only, and fully explainable.
2. **Matching model: engineered features + gradient-boosted trees (CatBoost or LightGBM), not a transformer.** Independent large-scale tabular benchmarks (McElfresh et al., NeurIPS 2023, 176 datasets; Shmuel et al., 111 datasets; TabArena 2025) consistently show GBDTs — with CatBoost usually a hair ahead on heterogeneous/categorical data — beating deep tabular nets, and Ditto-style transformers add real latency, GPU dependency, and audit/reproducibility risk for a challenge whose final package is manually reviewed. A GBDT on strong features captures most of what a transformer would learn implicitly, at far lower risk.
3. **The decision rule is not "prob > 0.5."** Because the metric is F₀.₅ *per Source‑1 entity, macro-averaged, with a variable and unknown number of true matches (0..many)*, this is exactly the "per-instance / macro multi-label F-measure" problem studied by Dembczynski et al. (NeurIPS 2011; ICML 2013) and Lipton, Elkan & Naryanaswamy (2014). Their result: naive fixed-threshold decisions are provably sub-optimal for this metric family; the Bayes-optimal rule sorts candidates by calibrated probability and picks the prefix length that maximizes *expected* F_β — which for β=0.5 is a strongly precision-biased rule that this doc implements explicitly (Section 6).
4. **Country is an open set — architect for it now, not after France breaks something.** The spec explicitly says France appears only in test. Every step (normalization, blocking, features, encoding) must be validated by *simulating* an unseen country during development (leave-one-country-out), because there is no other way to rehearse for France before the real test set.
5. **Do not build a Spark cluster or fine-tune an LLM.** The spec's own "billions of records" framing describes Amazon's real system, not necessarily this challenge's train/test file sizes [EDA-REQUIRED: confirm actual row counts]. A single-machine, vectorized, CPU-first pipeline (sparse TF-IDF matrices + GBDT) is almost certainly sufficient and is dramatically easier to reproduce, document, and defend under the manual code-audit the spec describes.

---

## 1. Problem & Data Analysis

### 1.1 Structural facts from the spec (not assumptions)
- Three sources: **S1** (deduplicated reference), **S2**, **S3** (noisy). Task: for every S1 test entity, return its S2/S3 matches.
- Cardinality is **0..many**: singletons are valid and explicitly rewarded (F₀.₅ = 1.0 for a correctly-predicted-empty singleton, 0.0 for any false merge on it).
- No direct S2↔S3 ground truth is given or required — you only ever need to resolve "does this S2/S3 record match this S1 entity," which lets you treat the problem as **|S1| independent multi-label retrieval problems**, not a general pairwise clustering problem. This is a significant simplification versus classic ER (no need for transitive closure/clustering logic across S2 and S3 directly).
- Country is US + India in train; **France added only in test**, explicitly flagged as an open string set the pipeline must not hard-code against.
- Ground truth file gives comma-separated `matched_entity_ids` per S1 id — this is your only labeled signal; treat every (S1, listed-id) as a positive pair and, once blocking is applied, every (S1, retrieved-but-not-listed) as a negative pair.

### 1.2 What must be measured empirically before any modeling decision is finalized [EDA-REQUIRED]
| Quantity | Why it matters | How to measure |
|---|---|---|
| Row counts per file (S1/S2/S3, train/test) | Determines whether CPU-only sparse retrieval is sufficient or you need approximate NN | `wc -l`, `pd.read_csv(..., sep="\t").shape` |
| Singleton rate (fraction of S1 entities with empty `matched_entity_ids`) | Directly sets the *base rate* your threshold must beat; if singletons are >50%, a model that over-predicts matches is catastrophic under F₀.₅ | `groupby` + count of empty match lists |
| Match-count distribution (0,1,2,3+ matches per S1) | Determines whether a fixed top-1 rule is adequate or you truly need a variable-length decision rule | histogram of `len(matched_entity_ids.split(','))` |
| Per-source contribution (fraction of matches from S2 vs S3) | May justify separate models/thresholds per source-pair | split ground truth by prefix of matched ids |
| Per-country match rate and noise severity (edit distance of matched name/address pairs) | Tells you how aggressive normalization/fuzzy matching needs to be per country | compute string distance only on **known positive** pairs (train) |
| Missingness rate of address sub-fields (PIN/ZIP/postal, state) | Determines whether a "both fields present" gating feature is needed | regex extraction + null-rate per country |
| Duplicate/near-duplicate business names within S1 itself | Confirms S1 truly is deduplicated as claimed, or reveals residual duplicates you should not "fix" (out of scope) | self-similarity check within S1 |

### 1.3 Class imbalance and why it dictates the whole architecture
The true positive-pair rate over the full S1 × (S2 ∪ S3) Cartesian product is astronomically small (this is standard in ER: with n₁ S1 records, n₂₃ S2+S3 records, the match rate is `O(matches / (n1 * n23))`, typically far below 0.01%). This is *the* reason ER always factors into **blocking (recall-oriented, cheap, coarse) → matching (precision-oriented, expensive, fine)** rather than one classifier over all pairs — confirmed as the standard framework by the Papadakis et al. ACM Computing Surveys blocking/filtering survey (2020) and its predecessor VLDB tutorials. Skipping straight to a pairwise classifier over the full cross product is both computationally infeasible at scale and statistically terrible (a classifier trained on 1:1,000,000+ imbalance without blocking will not learn useful decision boundaries from a modestly sized labeled set).

### 1.4 Validation design (get this wrong and every later number is fiction)
- **Split at the S1-entity level**, not the pair level. Use `GroupKFold`/`GroupShuffleSplit` keyed on `source1_entity_id`. Splitting at pair level lets the same S1 entity's positive and negative candidates leak across train/val, inflating validation F₀.₅ in a way that will not reproduce on the real test set.
- **Stratify by singleton vs. non-singleton and by country** so validation folds mirror the true macro-average composition — a fold that happens to be singleton-heavy or India-heavy will give a biased F₀.₅ estimate otherwise.
- **Simulate the unseen-country problem**: hold out one full country (e.g., train on India-only + a subset of US, validate on held-out US) as a *rehearsal* for how the pipeline behaves on a country it has never seen — the closest available proxy for France, since France ground truth does not exist anywhere for you to check against.
- **Never let test-set French records touch anything label-derived** (IDF weights fit on label-based subsets, target encodings, hard-negative mining thresholds tuned only on train countries) without re-deriving those statistics from the actual corpus being indexed at inference time (see §7 and §8 on where corpus statistics are legitimately allowed to be recomputed vs. where they'd leak).

### 1.5 Leakage risks catalogue
| Risk | Mechanism | Mitigation |
|---|---|---|
| Pair-level split leakage | Same S1 entity's pairs split across train/val | Group split by `source1_entity_id` |
| Threshold overfit to public LB | Repeatedly tuning threshold against public leaderboard subset | Trust internal CV F₀.₅ as primary signal; treat public LB as a secondary sanity check only |
| Blocking-recall computed on the wrong set | Measuring recall@K using test data you don't have labels for | Recall@K is only measurable on train/val (with ground truth); test recall is inferred, not verified — budget a safety margin in K |
| IDF/statistics computed with label knowledge | Weighting tokens using match/no-match info | Fit IDF purely from record content (S2+S3 text), never from label distribution |
| Metric miscomputation | Implementing per-pair F₀.₅ instead of per-entity macro-averaged F₀.₅ | Unit-test your scorer against the spec's own worked example (S1-00001 → F₀.₅ = 0.714) before trusting any validation number |

---

## 2. Normalization

**Golden rule:** normalization is *additive*, never destructive. Always keep the raw field alongside a normalized field (or several: `name_norm`, `name_tokens`, `name_ngrams`), because downstream feature engineering needs both — the exact challenge document's own noise list ("abbreviations, DBA/trade names, punctuation, word-order, typos" for names; "abbreviations, transliteration, missing components, landmark references, municipal numbering, reordering" for addresses) is a checklist of what normalization must handle, but several of those (DBA names, landmarks) must be *preserved as separate signal*, not deleted.

### 2.1 Name normalization
| Step | Action | Rationale / risk if skipped or overdone |
|---|---|---|
| Unicode | NFKC normalize, casefold (not just `.lower()`, to correctly fold ß, etc.) | Standard first step in production entity-matching pipelines (confirmed pattern in multiple company-name-normalization writeups) |
| Legal suffix handling | Detect and **split off**, don't just delete: `Pvt Ltd / Private Limited`, `LLP`, `Inc / Incorporated`, `Corp / Corporation`, `Co`, `Ltd / Limited`, `LLC`, `SARL`, `SAS`, `SA`, `EURL` → map to a small canonical vocabulary; store as `legal_form` categorical feature | Deleting entirely loses a legitimate (if weak) match/mismatch signal; keeping it inline in the name string just adds noise to string-similarity scores |
| Punctuation/ampersand | `&` ↔ `and`, strip commas/periods, collapse whitespace | Explicit noise pattern in the spec |
| Token-order invariance | Compute similarity on **token sets/multisets**, not raw string order, in addition to literal string similarity | Spec explicitly lists "word-order transpositions" |
| DBA / trade names | If a second name-like field or clear alternate form appears (rare in this schema but check EDA), keep as an alternate string to match against, never merge into a single lossy token bag | Business identity is often carried in the DBA, and blocking or matching only on the legal name misses it |
| Numeric tokens inside names | Extract and keep separately (e.g., "7-Eleven", "3M", "24 Seven Logistics") | Aggressive "strip all digits" normalization silently breaks brand identity for numeric brand names |
| Acronym/initialism detection | Flag when one record's short all-caps token equals the initials of the other record's expanded tokens (e.g., "IBM" vs. "International Business Machines") | Common, high-value, and easy to miss with plain string similarity (low edit-distance score despite being a true match) |
| Stopword handling | Be conservative — do **not** strip generic business words (`the`, `of`, `and`, `group`, `services`, `enterprises`) globally, since two *distinct* businesses often differ only by such a word (e.g. "ABC Foods" vs. "ABC Exports") | This is the single most common way over-normalization *hurts precision*, which is what the metric penalizes most |
| Transliteration variants (Indian names) | Character n-gram similarity is more robust here than English-tuned phonetic algorithms (Soundex/Double Metaphone are tuned to English phonotactics and degrade on Indic transliteration, e.g. Lakshmi/Laxmi, Shree/Sri) | Use n-gram Jaccard/cosine as primary signal for these; treat phonetic score as a *secondary*, not primary, feature for Indian names |
| French diacritics | NFKD decompose + strip combining marks (é→e, ç→c, œ→oe) | Standard, low-risk, high-value for French test records |

### 2.2 Address normalization
| Step | Action | Rationale |
|---|---|---|
| Abbreviation expansion | `Rd↔Road`, `St↔Street`, `Ave↔Avenue`, `Apt/Flat`, `Nr/Near` | Spec-listed noise pattern |
| Structured extraction | Regex-extract, per country: US ZIP (`\d{5}(-\d{4})?`), Indian PIN (`\d{6}`), French postal code (`\d{5}`), plus a generic fallback (`\d{4,6}` as last resort for unseen-country records) | Numeric postal codes should be compared **exactly or with tiny edit-distance tolerance**, never with generic fuzzy string similarity — a 1-digit PIN difference is a *different location*, not a near-match |
| Missing-component flags | Explicit boolean features: `has_postal_code`, `has_state` | Comparing "present vs. absent" fields with the same similarity function as "present vs. present" produces misleading mid-range scores; the model needs to know a field was simply missing |
| Landmark phrases (e.g., "Near SBI ATM") | Extract into a separate `landmark` field; fuzzy-match landmarks against each other, but do not let a landmark's absence lower the *formal* address score | Explicit spec noise pattern; landmarks are a genuine recall aid in India-style addresses but a low-precision signal on their own |
| Component reordering | Token-set / token-sort based similarity, not literal left-to-right string comparison | Explicit spec noise pattern |
| Self-built gazetteer (compliant) | Build a city/state/token frequency table **from the training corpus itself** (no external geocoding) to detect and canonicalize common city/state spellings that recur across records | Improves matching without violating the external-lookup ban — the "gazetteer" is derived entirely from the provided data, not from any outside source |

### 2.3 Country handling
The spec is explicit: **do not hard-code or one-hot to `{US, India}`.** Practical implementation:
- Treat `country` as a free-text categorical field encoded with **frequency/count encoding** (or CatBoost's native categorical handling, see §5), never a fixed-width one-hot vector — a fixed one-hot vector silently has *no column* for France and either crashes or (worse) silently encodes it as all-zeros, which most model families interpret as some other category rather than "unknown."
- Use `country` primarily as a **blocking partition key** with a graceful fallback: if country is missing/inconsistent, fall back to a global (unpartitioned) block for that record rather than dropping it.
- Choose the address-parsing regex **conditionally on `country`**, but always keep a generic numeric-extraction fallback path so an unrecognized country string (like `France` appearing for the first time) still gets *some* postal-code-like signal extracted instead of nothing.

### 2.4 What normalization should explicitly NOT do
- Do not fully strip legal suffixes without retaining them as a separate feature — they carry weak-but-real signal.
- Do not strip digits from names.
- Do not remove common business words (stopword removal is often used in generic NLP; here it destroys precision).
- Do not treat "missing field" the same as "field present but different" — always carry a presence flag.
- Do not throw away the raw string — always keep it for exact-match features and audit/debugging.

---

## 3. Blocking / Candidate Generation

This is the section the spec now scores directly (`candidate_pairs.tsv` + code, weighted in final ranking beyond leaderboard F₀.₅), so it gets the deepest comparative treatment.

### 3.1 Method comparison

| Method | How it works | Recall | Candidate-set size | Compute cost | Verdict for this challenge |
|---|---|---|---|---|---|
| Exact/structured blocking (exact normalized name or exact postal code) | Hash-join on a canonicalized key | Low–moderate; misses any typo/transliteration | Very small | Trivial | Too brittle alone; fine as one *channel* among several |
| Standard token/q-gram blocking | Block key = sorted token set or fixed-length q-grams | Moderate | Can be large if generic tokens dominate | Low | Useful building block, not sufficient alone |
| **Top-k TF‑IDF / BM25 retrieval** (character n-grams + word tokens, inverted index, top-K per query) | Rank S2/S3 candidates for each S1 query by TF-IDF/BM25 score, keep top-K | **High** — Paulsen, Govind & Doan ("Sparkly," VLDB 2023) show this class of method outperforming 8 SOTA blockers including deep-learning blockers, on recall, output size, *and* runtime, in a large controlled study | Small if tuned via recall/size curve | Low (sparse matrix ops; no GPU) | **Primary channel** |
| Phonetic blocking (Soundex/Double Metaphone) | Encode names to phonetic codes, block on equality | Low alone; catches a distinct error class (spelling-by-sound) | Small | Trivial | Cheap **supplementary channel**, mainly for US/English-pattern names |
| ANN/embedding retrieval (fastText/SBERT + FAISS or brute-force cosine) | Encode records to dense vectors, retrieve nearest neighbors | Can catch pure paraphrase/semantic variants token overlap misses | Depends on K | Higher (encoding pass, index build) | Zeakis et al. (VLDB 2023) ran a controlled 12-embedding × 17-benchmark study and found embedding-based blocking *is* competitive, but with real vectorization overhead and no consistent, large win over strong lexical baselines. **Recommendation: secondary recall-topper**, not the primary channel, given Sparkly's evidence that a simpler method already covers most of the space |
| Multi-pass / adaptive blocking with meta-blocking pruning | Union several blocking-channel outputs into a graph, weight edges by combined evidence, prune low-weight edges | Preserves the best recall of all unioned channels | **This is the lever that directly optimizes the spec's new candidate-set-size criterion** — Papadakis et al.'s meta-blocking work (and its supervised/generalized variants, Gagliardelli et al., VLDB 2022) show pruning can remove the bulk of redundant/low-value comparisons at near-zero recall cost | Low incremental cost over the channels already computed | **Final fusion + pruning stage** |

### 3.2 Recommended architecture: four-stage blocking

```
Stage A — Coarse partition            Stage B — Primary retrieval
  by country (open-set safe,            top-K TF-IDF/BM25 over
  generic fallback bucket for              name n-grams + tokens,
  unrecognized/missing country)            within each partition

Stage C — Recall top-up channels      Stage D — Meta-blocking fusion
  (union, not intersection)             combine per-channel scores
  • exact postal-code match              into one blocking score,
  • phonetic-code match (US/English)     re-rank, cut to top-N —
  • n-gram Jaccard on address            THIS final cut is what
                                          candidate_pairs.tsv holds
```

- **Stage A** partitions the search space by country to shrink the universe before any retrieval cost is spent — but always includes a fallback partition so a record with a missing/garbled/unseen country label is never silently dropped.
- **Stage B** is the workhorse: build a `TfidfVectorizer`-style sparse matrix over character n-grams (n=2,3) + word tokens of `name_norm` (optionally concatenated with a lighter-weighted address field), then retrieve top-K nearest S2/S3 candidates per S1 query via sparse matrix multiplication or `sklearn.neighbors.NearestNeighbors(metric='cosine')`. This directly mirrors the Sparkly design (TF-IDF + top-k), minus the Spark/Lucene distribution layer that paper needed only because it targeted 100M+-record production settings.
- **Stage C** unions in candidates the primary channel would miss by construction: exact postal-code matches (catches heavily-abbreviated/garbled names with a stable address) and phonetic-code matches (catches phonetic misspellings with low n-gram overlap).
- **Stage D** (meta-blocking) is what actually produces `candidate_pairs.tsv`: combine each channel's evidence into one blocking score per (S1, candidate) edge, re-rank, and cut at the smallest N that preserves your validation recall ceiling (see §3.3). This is the step that lets you claim, with evidence, that your candidate set is *both* high-recall *and* deliberately minimized — exactly what the spec's new scoring criterion rewards.

### 3.3 How to size K and N (never guess)
1. Compute **recall@K** on the labeled train/val split: fraction of true positive pairs whose true match is present in the union of blocking channels at cutoff K.
2. Plot recall@K vs. average candidate-set size as K grows from small to large.
3. Pick the smallest K (and post-meta-blocking N) where the recall curve visibly plateaus (typically recall gains beyond this point are marginal while candidate count keeps growing linearly) — this is your **reduction-ratio-optimal operating point**, and it's the number you report in the methodology doc.
4. **Never tune K/N against the leaderboard.** Tune only against your own recall ceiling measurement, because the private leaderboard is where consistency will actually be judged, and a K chosen to just barely pass a specific public subset is a form of overfitting.

### 3.4 Why not lead with a deep blocker or transformer retriever
- Evidence (Sparkly, VLDB 2023) shows a well-built TF-IDF blocker beating deep blockers, so the expected marginal gain from starting with something heavier is not well supported.
- Deep blockers add embedding fine-tuning burden, need more labeled data than a modestly sized challenge train set may offer, and complicate the "runnable pipeline anyone can reproduce" requirement in the spec's Final Submission Package review.
- This does not forbid using an embedding similarity **as one input to Stage D's fused score**, or as a recall-topper for known-hard cases — that is exactly where it belongs (see §9, Innovation #5).

---

## 4. Pair Feature Engineering

Organize features by what they cost to compute and what error class they catch. All of these should be computed **vectorized** over the whole candidate set at once (see §8), never in a per-pair Python loop.

### 4.1 Name features
- Exact match (post-normalization) — boolean, highest-precision single feature you have.
- Token-set Jaccard, token multiset (bag) overlap.
- Character n-gram (n=2,3) Jaccard and cosine (TF-IDF weighted).
- Levenshtein distance (raw + length-normalized ratio).
- Jaro-Winkler similarity — particularly strong on short names with prefix agreement and transpositions.
- Token-sort ratio (reorders tokens alphabetically before comparing) — directly targets the spec's "word-order transposition" noise pattern.
- Longest common subsequence ratio.
- TF-IDF cosine similarity using the same vectorizer/index built for blocking (free reuse of Stage B's artifacts — see §8 caching).
- Legal-form match flag (from §2.1's canonical `legal_form` field).
- Acronym/initialism match flag.
- Shared numeric-token flag (e.g., both contain "7", "3M").
- Length difference (character count, token count).
- Substring containment flag (does the shorter name appear inside the longer one — catches DBA/abbreviated forms).
- *(Optional, secondary)* cosine similarity of a pretrained sentence/word embedding (fastText or a generic open-license SBERT) — one extra numeric column, not a separate model.

### 4.2 Address features
- Token-set Jaccard, character n-gram cosine.
- Postal-code exact-match flag; postal-code edit distance (small, discrete — treat as near-categorical, not continuous fuzzy).
- Street-number exact-match flag (extracted leading numeric token).
- City/state match flag, using the self-built gazetteer from §2.2 to canonicalize spelling variants before comparing.
- Landmark similarity score (fuzzy match on extracted landmark phrases) as a separate, lower-weighted feature.
- Missing-component flags on both sides (`has_postal_code_s1`, `has_postal_code_cand`) — lets the model learn "we can't confirm via postal code" rather than treating absence as a mismatch.
- Address length difference.

### 4.3 Country / cross-field features
- Country exact-match flag — expect this to be a strong feature; **validate on training data whether any true positive pairs ever cross countries** [EDA-REQUIRED] before treating mismatch as a hard reject.
- Source-pair type (`S1-S2` vs. `S1-S3`) as a categorical feature — sources may carry systematically different noise, and a model that knows which pair type it's scoring can calibrate accordingly.

### 4.4 Positional / retrieval-derived ("competition") features
These are the features that most differentiate a well-designed feature set from a naive one, because they let the tabular model reason about *relative* competition among candidates for the same S1 entity without needing a full listwise-ranking model:
- Blocking-stage rank of this candidate among all candidates retrieved for this S1 entity.
- Raw blocking score (TF-IDF/BM25 score) and its normalized rank/percentile within this entity's candidate set.
- Score gap to the next-best candidate for the same S1 entity (large gap → this candidate is a confident standout; small gap → an ambiguous tie the model/threshold logic should treat cautiously).
- Count of candidates for this S1 entity above a moderate similarity threshold (a proxy for "how contested is this entity's candidate pool").
- Total candidate-set size for this S1 entity (very small sets are often either a clean singleton or a clean single match; very large sets need more evidence to commit to any one candidate).

### 4.5 Numeric features
- Set of extracted numeric tokens from name and address; exact-overlap count and Jaccard.
- Difference between extracted postal codes as integers (only meaningful when both present).

### 4.6 Feature validation
Run permutation importance and/or SHAP on the trained GBDT (cheap, since GBDTs give this natively) after the first full feature set is built, to (a) confirm the "competition" features from §4.4 actually help — they are the novel part and deserve direct ablation — and (b) catch any feature that is accidentally a leakage proxy (e.g., if `source1_entity_id` or `candidate_entity_id` string patterns correlate with row order in a way that leaks train/val membership).

---

## 5. Matching Models

### 5.1 Candidates compared

| Model family | Strength here | Weakness here | Verdict |
|---|---|---|---|
| Logistic Regression | Fast, fully interpretable, trivially calibrated | Needs manual interaction terms to capture nonlinear feature combos (e.g., "high name similarity AND matching postal code" vs. either alone); ceiling well below GBDTs on this kind of heterogeneous tabular feature mix | Baseline / sanity check only |
| XGBoost | Mature, fast, excellent regularization knobs, huge tooling ecosystem | Slightly behind CatBoost on categorical-heavy, heterogeneous data in recent large-scale benchmarks | Strong, viable option |
| LightGBM | Leaf-wise growth, very fast to train, competitive accuracy | Can overfit small/noisy datasets faster than depth-wise growth; needs care on a modestly sized labeled set | Strong, viable option, fastest to iterate with |
| **CatBoost** | Native categorical handling (ideal for `country`, `legal_form`, `source_pair_type` without manual encoding), ordered boosting reduces target leakage/overfitting on smaller labeled sets | Slightly slower to train than LightGBM | **Recommended primary** — supported by Shmuel et al. (111 datasets: CatBoost 19 best-score wins vs. LightGBM 15, XGBoost 5; best avg rank), McElfresh et al. (NeurIPS 2023, 176 datasets: CatBoost best mean rank among GBDTs), and TabArena (2025: CatBoost top GBDT under conventional tuning, particularly strong on high-cardinality categoricals and mixed-type/noisy features — exactly this problem's shape) |
| Transformer / pretrained-LM sequence-pair classifier (Ditto-style) | Ditto (Li et al., 2020) reports up to ~29% F1 gain over prior SOTA on EM benchmarks, and is especially strong on **dirty data with limited labels** — a real match to this challenge's noise profile | Needs GPU for practical fine-tuning/inference at scale, adds nondeterminism and a heavier reproduction burden for the spec's manual code audit, and the marginal gain over a *feature-rich* GBDT is smaller than the paper's gain over *un-engineered* baselines (Ditto's comparison points did not have this document's hand-built feature set) | Optional differentiator (§9), not the backbone |
| Sentence-embedding cosine similarity alone | Simple, catches pure semantic/transliteration paraphrase | On its own, weaker precision than a fused feature-based classifier; embeddings help more as one input feature than as a sole score (consistent with Zeakis et al.'s finding that no single embedding dominates lexical baselines across benchmarks) | Feature input only, not a standalone matcher |

### 5.2 Recommendation
**Primary matcher: CatBoost (or LightGBM as a closely competitive alternative) trained on the full engineered feature set from §4, with native categorical handling for `country`, `legal_form`, and `source_pair_type`.** This satisfies the spec's model constraint trivially (open-source, no "parameters" in the LLM sense, well under any size ceiling) and gives calibrated-enough probability outputs for the thresholding logic in §6 (with a Platt/isotonic calibration pass layered on top, since raw GBDT leaf-averaged probabilities are usually reasonably but not perfectly calibrated).

**Optional ensemble:** average or lightly stack CatBoost + LightGBM (two models, not more) for stability — this is cheap insurance against a single model's idiosyncratic errors, without the complexity/audit cost of a larger ensemble or a neural component. Do not go beyond two base learners; the spec's own "efficiency" framing and "avoid unnecessary complexity" tip explicitly caution against this, and diminishing returns on a challenge of this shape are real.

**Why not lead with a transformer:** the case *for* Ditto-style modeling is genuinely evidence-backed (dirty data, small-label regime — see §5.1), which is why it is kept as a listed innovation (§9) rather than dismissed — but leading with it trades a well-evidenced, reproducible, fast baseline for a harder-to-audit approach whose marginal benefit over strong features + GBDT is the open empirical question your team should test in the experiment roadmap (§12), not assume.

---

## 6. Thresholding & Decision Logic

### 6.1 Why "probability > 0.5" is the wrong mental model here
The scored metric is **F_β (β=0.5), computed per S1 entity, then macro-averaged, including singletons** (spec's own formula and example). This is precisely the "macro/per-instance multi-label F-measure" setting studied by:
- Dembczynski, Waegeman, Cheng & Hüllermeier, *An exact algorithm for F-measure maximization* (NeurIPS 2011) and Dembczynski et al., *Optimizing the F-measure in multi-label classification: plug-in rule vs. structured loss* (ICML 2013): the Bayes-optimal prediction for per-instance F-measure is **not** "threshold each label independently at a fixed cutoff" — it requires an algorithm that, given the sorted probability estimates for one instance's candidate labels, computes the expected F_β for every possible predicted-set size and picks the maximizing one.
- Lipton, Elkan & Naryanaswamy, *F1-optimal thresholding in the multi-label setting* (2014): shows the optimal threshold is **not generically 0.5** and can even depend on the score distribution of the whole batch, and that naive fixed-threshold macro-F1 can conceal degenerate behavior.

### 6.2 Recommended decision rule (the "F-measure maximizer" / FMM approach)
For each S1 test entity with candidates sorted by descending calibrated match probability `p_1 ≥ p_2 ≥ ... ≥ p_m`:
1. Under the (standard, and reasonably robust in practice per the multi-label literature above) assumption that candidate match probabilities are conditionally independent given the entity, compute the **expected F_β** for predicting the top-`k` candidates as matches, for every `k = 0, 1, ..., m`.
2. Predict the prefix length `k*` that maximizes expected F_β for that entity.
3. `k* = 0` naturally reproduces "predict singleton" — no special-casing needed; it falls out of the same optimization.

```python
def optimal_prefix_fbeta(sorted_probs, beta=0.5):
    """
    sorted_probs: candidate match probabilities for ONE S1 entity,
                   sorted descending. Returns k* (number of top candidates
                   to predict as matches) that maximizes expected F_beta,
                   using the standard multi-label F-measure-maximizer
                   plug-in-rule recursion (Dembczynski et al. 2011/2013).
    """
    # Practical implementation: dynamic-programming expected-F_beta table
    # over possible (true-positive-count, predicted-count) pairs, using
    # sorted_probs as the per-candidate match-probability estimates.
    # See Dembczynski et al. (2011) for the exact O(m^2) recursion;
    # a simpler, close-to-optimal approximation many practitioners use
    # is to compute expected F_beta directly under independence:
    #   E[F_beta | predict top-k] ≈ (1+beta^2) * sum(p_1..k) /
    #                                 (beta^2 * k + sum(p_1..k))   [approx.]
    # and grid-search k in [0, m]. Validate the approximation against
    # the exact DP on a held-out set before trusting it at scale.
    ...
```
- **Fallback / simpler baseline** (use this first, in V1, before implementing the full FMM): a single global probability threshold `T`, chosen by sweeping `T` on the validation set and picking the value that maximizes the *actual* macro per-entity F₀.₅ (not accuracy, not pair-level F1). Given F₀.₅ weights precision 2× recall, expect the F₀.₅-optimal `T` to sit **higher** than an F1-optimal `T` would — this must be swept, not assumed.
- Promote to the full per-entity FMM rule in V3 (§12) once the simpler global threshold is working and measured, and treat the *difference* between the two as the concrete, measurable value of implementing the more sophisticated rule.

### 6.3 False-merge guardrails (precision-side extra safety, beyond the raw probability)
- **Country-mismatch hard rule**: if EDA (§1.2) confirms true positives essentially never cross countries in training data, treat a country mismatch as a strong prior toward exclusion regardless of string similarity — a near-miss on country given real spec noise is worth double-checking, but a full mismatch should require very strong compensating evidence to survive.
- **Confidence-gap requirement for multi-match predictions**: when predicting more than one match for a single S1 entity, additionally require some minimum probability gap between the last included and first excluded candidate, to avoid weak flip-of-a-coin inclusions of a second/third candidate that are individually plausible but jointly evidence of an ambiguous, low-precision block.
- **Segment-specific calibration**: if EDA shows S1-S2 and S1-S3 pairs, or different countries, have systematically different score distributions, calibrate probabilities (and potentially the FMM computation) per segment rather than globally — validate this need empirically via segment-wise reliability diagrams before adding the complexity.

### 6.4 How to measure whether the decision-logic choice actually helped
Compare, on the same validation fold and same trained model: (a) fixed global threshold, (b) per-segment threshold, (c) FMM per-entity rule. Report macro F₀.₅ for each, plus average candidate-set size implied by each rule's predictions (a rule that quietly predicts more matches to chase recall will look fine on F₀.₅ only if precision holds — watch both numbers together).

---

## 7. Validation & Experimentation

### 7.1 Split design
- `GroupKFold` (or `GroupShuffleSplit` for a single held-out set, faster to iterate) grouped on `source1_entity_id`, stratified by singleton-flag and country where feasible.
- Maintain a **leave-one-country-out** variant purely for robustness testing (train excluding one country's S1 entities entirely, validate only on that country) — this is your rehearsal for France, not a fold you'd use to pick your main hyperparameters.

### 7.2 Hard-negative mining
1. Train an initial model on blocking-derived negatives (candidates retrieved but not in ground truth).
2. Score all training candidates with this model; identify **high-scoring false positives** — pairs the model is confidently wrong about.
3. Explicitly upweight/oversample these hard negatives in the next training round (and add any additional true near-duplicates you can construct as synthetic hard negatives, e.g., two genuinely different S1 entities that share a common token).
4. Also ensure **negatives are drawn from within blocks**, not randomly from the whole corpus — random far-away negatives are trivially separable and teach the model nothing about the precision boundary that actually matters for F₀.₅ (a boundary that only gets tested by near-miss, in-block negatives).

### 7.3 Ablation plan
| Ablation | What it isolates |
|---|---|
| Remove blocking channel X (phonetic / postal-exact / TF-IDF) one at a time | Marginal recall contribution of each channel — needed to justify keeping or cutting a channel on cost/benefit grounds |
| Remove meta-blocking fusion/pruning (use raw union instead) | Value of Stage D specifically, since that's what the new scoring criterion rewards |
| Remove feature group (name-only / address-only / competition-features) | Value of each feature family — the "competition" features (§4.4) are the most novel and most worth isolating |
| Global threshold vs. per-segment vs. FMM | Value of the decision-logic sophistication (§6.4) |
| With/without hard-negative mining round | Precision improvement attributable to the mining loop specifically |
| CatBoost vs. LightGBM vs. XGBoost vs. LR | Confirms the model choice empirically on *this* dataset rather than on published benchmarks alone |

### 7.4 Metrics to track every experiment
- **Blocking**: recall@K, average candidate-set size per S1 entity, reduction ratio (`1 - candidates_compared / (|S1| * |S2∪S3|)`).
- **Matching**: macro F₀.₅ (official formula, singleton-inclusive), precision, recall, at both the fixed-threshold and FMM decision rules.
- **Calibration**: reliability diagram / Brier score of the matching model's probability outputs, since the FMM decision logic's validity depends on reasonably calibrated probabilities.

### 7.5 F₀.₅ scorer — validate it against the spec's own example before trusting anything
```python
def f_beta_per_entity(true_ids: set, pred_ids: set, beta: float = 0.5) -> float:
    if not true_ids and not pred_ids:
        return 1.0          # correct singleton
    if not true_ids and pred_ids:
        return 0.0          # false merge on a true singleton
    tp = len(true_ids & pred_ids)
    precision = tp / len(pred_ids) if pred_ids else 0.0
    recall = tp / len(true_ids) if true_ids else 0.0
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)

# Sanity check against the spec's worked example:
# true = {S2-00047, S3-00812}, pred = {S2-00047, S2-00193, S3-00812}
assert abs(f_beta_per_entity({"S2-00047","S3-00812"},
                              {"S2-00047","S2-00193","S3-00812"}) - 0.714) < 0.001
```

---

## 8. Efficiency

- **CPU-first by design.** Sparse TF-IDF matrices (`scipy.sparse`) and vectorized pandas/numpy feature computation should make this pipeline entirely CPU-tractable without needing a Spark cluster or GPU, given a single competition team's realistic per-file scale [EDA-REQUIRED: confirm]. Reserve GPU only for an *optional* embedding-similarity feature or an optional §9 transformer experiment.
- **Reuse the blocking index for features.** The TF-IDF vectorizer/index built in Stage B (§3.2) directly produces the TF-IDF cosine feature used in §4.1 — do not refit a second, inconsistent vectorizer for features.
- **Vectorize every pair feature.** Compute string-similarity features over the whole candidate-pair DataFrame using array operations (or a small, well-tested batch of `apply`-free vectorized string-metric libraries), not per-row Python loops — this is the single biggest practical runtime lever at this scale.
- **Cache immutable artifacts**: normalized text, tokenizations, IDF weights, and the built retrieval index should be persisted to disk between the train/val/test runs of the *same* corpus, and rebuilt fresh (not reused) whenever the underlying record set changes (train run's IDF must come from train S2+S3; test run's IDF must come from test S2+S3 — these are legitimately different fits, not leakage, since IDF describes the corpus being searched, not any label).
- **Batch inference.** Score the entire candidate-pair feature matrix through the trained GBDT in one call, not entity-by-entity.
- **Avoid unnecessary complexity** (the spec's own tip): no distributed infrastructure, no unnecessary ensemble depth, no fine-tuning pipeline unless the §9/§12 experiments show it clears a meaningful bar over the simpler baseline.

---

## 9. Stand-Out Innovations (ranked)

1. **Per-entity F-measure-maximizer decision rule (§6.2)** instead of a single global threshold — a principled, literature-backed technique (Dembczynski et al.) that directly targets the exact metric structure the spec scores (per-entity, macro-averaged, variable-length, precision-weighted), rather than a proxy.
2. **Four-stage blocking with explicit meta-blocking fusion/pruning (§3.2)** — directly optimizes the spec's newly added, explicitly-weighted "smaller candidate set at equal recall" criterion, not just the leaderboard F₀.₅.
3. **Hard-negative-mining feedback loop (§7.2)** — measurably improves precision (the 2×-weighted half of F₀.₅) without needing any additional labeled data, by making the model's own errors the next round's training signal.
4. **"Competition" features (§4.4)**: score-rank, gap-to-runner-up, count-above-threshold, and candidate-set size. These let a simple, auditable tabular GBDT reason about *relative* competition among an entity's candidates — the thing that normally requires a full listwise learning-to-rank model — without adding a second model family.
5. **Country-open-set robustness by construction (§2.3, §1.4)**: never one-hot to a fixed vocabulary, and rehearse against an unseen country via leave-one-country-out validation *before* the real France-containing test set is scored — a discipline most competitors, focused only on public-leaderboard F₀.₅, are likely to skip.
6. *(Optional 6th, lower priority, higher risk/reward)*: a single pretrained sentence/word-embedding similarity column (fastText or a generic open-license SBERT run locally) folded into the existing tabular feature set as one more numeric feature — captures pure semantic/transliteration matches that lexical methods miss, at the cost of one extra, fully local, license-checked model, without promoting embeddings to a separate primary matcher.

---

## 10. Final Architecture

```mermaid
flowchart TD
    A[Raw S1 / S2 / S3 TSVs] --> B[Normalization<br/>names, addresses, country<br/>additive: raw + normalized fields kept]
    B --> C1[Stage A: Country partition<br/>+ generic fallback bucket]
    C1 --> C2[Stage B: TF-IDF/BM25<br/>top-K retrieval per S1 entity]
    C1 --> C3[Stage C: recall top-up<br/>postal-exact + phonetic channels]
    C2 --> D[Stage D: Meta-blocking fusion + pruning<br/>-> candidate_pairs.tsv]
    C3 --> D
    D --> E[Vectorized pair feature engineering<br/>name / address / country / competition features]
    E --> F[CatBoost (+/- LightGBM ensemble)<br/>calibrated match probability]
    F --> G[Per-entity decision logic<br/>F-measure-maximizer / FMM]
    G --> H[matching_results.tsv]
    D --> H
    subgraph Validation harness
      V1[Group-split CV by S1 entity]
      V2[Leave-one-country-out rehearsal]
      V3[Recall@K + reduction ratio]
      V4[Macro F0.5 scorer, spec-verified]
    end
```

**Why each component was chosen** (cross-references to the section that argues it):
- Normalization additive, not destructive → §2.
- Country partition with fallback, never fixed one-hot → §2.3, §1.4 (open-set requirement).
- TF-IDF/BM25 as primary blocking channel → §3.1 (Sparkly evidence).
- Postal-exact + phonetic as supplementary channels → §3.2 (distinct error classes, near-zero cost).
- Meta-blocking fusion/pruning as the final candidate-set-producing stage → §3.2–3.3 (directly targets the spec's new scoring axis).
- Vectorized feature engineering reusing the blocking index → §4, §8 (efficiency + consistency).
- CatBoost (or LightGBM) as the sole primary model family → §5 (benchmark evidence + reproducibility/audit considerations).
- FMM per-entity decision logic → §6 (matches the exact metric structure).
- Validation harness with group-split + leave-one-country-out → §1.4, §7 (leakage control + France rehearsal).

---

## 11. Implementation Plan (phased order)

| Phase | Deliverable | Gate to move on |
|---|---|---|
| 0 — Setup | Repo skeleton (`src/`, `README.md`, `requirements.txt`), pinned environment | Environment reproduces on a clean machine |
| 1 — EDA | All [EDA-REQUIRED] numbers from §1.2 measured and written up | Singleton rate, match-count distribution, and per-country stats are known and documented |
| 2 — Normalization | Name/address/country normalization functions + unit tests on synthetic noisy examples covering every noise pattern the spec lists | All listed spec noise patterns have a passing test case |
| 3 — Blocking | Four-stage pipeline (§3.2) implemented; recall@K and candidate-set-size curves plotted | Recall@K plateau identified; chosen K/N documented with the curve as evidence |
| 4 — Features | Full vectorized feature matrix for candidate pairs | Feature computation runs on the full candidate set within an acceptable runtime budget (measure it) |
| 5 — Model | CatBoost/LightGBM baseline trained, calibrated | Beats the Baseline experiment (§12) on held-out macro F₀.₅ |
| 6 — Decision logic | Global-threshold rule first, then FMM per-entity rule | FMM rule measurably beats global-threshold rule on the same validation fold |
| 7 — Validation harness | Group CV + leave-one-country-out + spec-verified F₀.₅ scorer | Scorer passes the worked-example unit test (§7.5) |
| 8 — Hard-negative mining | One mining round completed, model retrained | Precision improves on validation without a material recall drop |
| 9 — Full inference | `candidate_pairs.tsv` and `matching_results.tsv` generated for the real test set | `utils/validate_submission.py` returns PASS |
| 10 — Packaging | `code/business_entity_resolution/` self-contained and reproducible from scratch, `README.md`, `requirements.txt`, `Documentation_template.md` filled in | A teammate who was not involved can reproduce both output files following only the README |

---

## 12. Experiment Roadmap: Baseline → V1 → V2 → V3 → Final

| Experiment | What changes | Metric gate to proceed |
|---|---|---|
| **Baseline** | Exact-normalized-name blocking only; 3–4 obvious features (name exact match, name Jaccard, address Jaccard, country match); Logistic Regression; single fixed threshold | Establishes the F₀.₅ floor and the blocking recall ceiling reference point — nothing to "beat" yet, just a documented starting line |
| **V1** | Full TF-IDF/BM25 top-K blocking (Stage B only, no fusion yet) + full engineered feature set (§4) + CatBoost or LightGBM + threshold swept on validation for max F₀.₅ | Must clear recall@K ≥ ~0.95 [tune to your EDA] at a defensible K, and beat Baseline's macro F₀.₅ by a clear margin |
| **V2** | Add Stage C recall-top-up channels + Stage D meta-blocking fusion/pruning; add one hard-negative-mining round | Must **reduce** average candidate-set size at equal or better recall@K versus V1, **and** improve precision/F₀.₅ from the mining round |
| **V3** | Replace fixed threshold with the per-entity FMM decision rule (§6.2); add competition features (§4.4) if not already in V1's feature set; add probability calibration | Must improve macro F₀.₅ over V2 at equal or smaller candidate set — this isolates the value of the decision-logic sophistication specifically |
| **Final** | Leave-one-country-out robustness pass; optional embedding-similarity feature (§9 innovation #6); optional 2-model ensemble (CatBoost + LightGBM); finalize docs/package | No regression in blocking recall or macro F₀.₅ versus V3 on the standard validation fold; leave-one-country-out fold shows the pipeline behaves sanely (doesn't collapse) on an unseen country, as a France rehearsal; `validate_submission.py` PASS |

---

## 13. Failure Modes — ways this could look strong locally and fail on the hidden/private leaderboard

| Failure mode | Mechanism | Guard against it |
|---|---|---|
| Country overfit | Pipeline implicitly tuned to US+India string patterns collapses on France | Leave-one-country-out validation (§1.4, §7.1) *before* submission, not after a bad private-LB surprise |
| Public-LB overfitting | Repeatedly tuning threshold/K against the public subset | Treat internal CV F₀.₅ as primary; public LB as secondary confirmation only |
| Wrong metric optimized | Threshold/model tuned against pair-level accuracy or F1 instead of the spec's exact per-entity macro F₀.₅ | Always score with the spec-verified scorer (§7.5); never trust `sklearn`'s default multi-label averaging without checking it matches the spec's definition |
| Distribution shift in blocking | Candidate generation tuned to US/India name-and-address conventions silently loses recall on French naming/addressing conventions (e.g., `SARL`, different postal formats) | Keep a generic, country-agnostic fallback channel (n-gram/TF-IDF works regardless of language) and explicitly test recall@K on the held-out-country fold |
| Pair-level split leakage | Same S1 entity split across train/val inflates validation F₀.₅ unrealistically | Group-split by `source1_entity_id` (§1.4) — verify no group appears in more than one fold |
| Easy-negative overconfidence | Random, far-outside-block negatives make the model overconfident and imprecise on real, in-block hard negatives at inference | In-block negative sampling + hard-negative mining (§7.2) |
| Over-aggressive candidate-set pruning | Chasing a smaller `candidate_pairs.tsv` past the point recall starts dropping | F₀.₅ is still the primary scored metric — never trade meaningful recall for a smaller candidate set beyond the plateau point identified in §3.3 |
| Non-reproducibility | Unpinned dependency versions, unseeded randomness, notebook-execution-order dependence | Pin `requirements.txt` exactly; seed everything; structure `src/` as scripts/modules runnable end-to-end, not a notebook-only flow |
| Runtime blowup at full test scale | Feature/blocking code tested only on a small dev sample, not load-tested at full test-set size before the deadline | Explicitly time the full pipeline on the full test set well before the deadline, not the night of |

---

## 14. Compliance Audit

| Technique proposed in this doc | Uses only provided data? | Notes |
|---|---|---|
| TF-IDF/BM25 blocking | Yes | Built entirely from the provided `business_name`/`business_address` text |
| Phonetic blocking (Soundex/Double Metaphone) | Yes | Pure string algorithm, no external data |
| Self-built gazetteer for city/state canonicalization | Yes | Derived only from the training corpus's own address fields, not any external directory |
| Postal-code regex extraction | Yes | Pattern-matching only, no geocoding API call — the spec explicitly bans geocoding APIs |
| Optional pretrained fastText / open-license SBERT embedding feature | **Conditionally yes** | Must be a generic language/sentence encoder trained on unrelated public corpora (e.g., common-crawl word vectors, generic paraphrase-mining sentence encoders) — **not** any business-registry-aware or entity-linking model, and its license must be checked against the spec's MIT/Apache-2.0, ≤8B-parameter constraint before use |
| GBDT (CatBoost/XGBoost/LightGBM) | Yes | Open-source, Apache-2.0-family licensed, no "external lookup" involved, and no meaningful "parameter count" in the LLM sense — trivially inside the ≤8B constraint |
| No calls to commercial ER APIs, company registries (GST portal, Companies House, EIN registry, etc.), or geocoding services | Confirmed absent from every stage of this design | This is the spec's brightest line — explicitly re-check every third-party library added during implementation for any network call before shipping |
| Documentation | — | Every pretrained model actually used must be named + licensed in the methodology document, per the spec's audit expectation |

**Sign-off checklist before submission:** grep the final `code/` tree for any HTTP client usage, API keys, or geocoding library imports; confirm every model artifact's license file is present and satisfies MIT/Apache-2.0 + ≤8B params; confirm no step in `README.md`'s reproduction instructions requires network access beyond installing pinned packages.

---

## Final Deliverables (condensed reference)

### Recommended final stack
- **Language/runtime:** Python, CPU-only, pinned `requirements.txt` (pandas, numpy, scipy, scikit-learn, catboost, lightgbm, jellyfish or similar for phonetic/Jaro-Winkler, regex).
- **Blocking:** `scikit-learn` `TfidfVectorizer` (char n-grams + word tokens) + sparse cosine top-K retrieval, or a BM25 implementation (`rank_bm25` or a hand-rolled inverted index) — no Spark/Lucene cluster needed at this scale.
- **Matching model:** CatBoost primary, LightGBM as a cross-check/ensemble partner.
- **Decision logic:** global-threshold rule first (V1–V2), per-entity F-measure-maximizer (Dembczynski-style) by V3/Final.
- **Validation:** `GroupKFold`/`GroupShuffleSplit` by `source1_entity_id`, plus a dedicated leave-one-country-out fold.

### Architecture diagram
See §10.

### Exact implementation phases
See §11 table (Phase 0 → 10).

### Experiment matrix
See §12 table (Baseline → V1 → V2 → V3 → Final).

### Top 5 highest-impact innovations
1. Per-entity F-measure-maximizer decision rule.
2. Four-stage blocking with meta-blocking fusion/pruning targeting the new candidate-set-size criterion.
3. Hard-negative-mining feedback loop.
4. Competition/rank/gap features letting a simple GBDT reason relatively across an entity's candidates.
5. Country-open-set robustness verified via leave-one-country-out rehearsal.

### What NOT to build
- A distributed Spark/Lucene cluster for a single-team, competition-scale dataset.
- A fine-tuned transformer as the *primary* matcher before a feature-rich GBDT baseline has been measured and beaten by it in a controlled experiment.
- A full geocoding/external-gazetteer system — explicitly banned, and unnecessary given a self-built, in-corpus gazetteer covers the same need compliantly.
- Direct S2↔S3 matching or entity clustering across all three sources — out of scope; the spec only asks for S1-anchored matches.
- An ensemble deeper than two base learners.
- Any decision rule based on a single global "prob > 0.5" cutoff without first sweeping/validating it against the *actual* macro per-entity F₀.₅ definition.
- Skipping normalization on the theory that "the model will learn it" — the metric's precision weighting punishes exactly the kind of noisy near-miss the model would otherwise have to learn from scratch, with a labeled set that's likely too small to teach it reliably.

### Definition of "done" per stage
See the "Gate to move on" column in §11's phase table — each phase has an explicit, measurable exit criterion, not just "code runs."

### Concrete checklist for producing the final submission
- [ ] `output/matching_results.tsv` — one row per test S1 entity, correct columns, no self-matches, no duplicate IDs, empty list for singletons.
- [ ] `output/candidate_pairs.tsv` — the *actual* final-stage candidate set fed to the matching model at inference (post meta-blocking pruning), superset of everything in `matching_results.tsv`.
- [ ] `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` returns **PASS**.
- [ ] `code/business_entity_resolution/src/` contains the full, runnable pipeline; `README.md` gives exact, from-scratch reproduction steps; `requirements.txt` is pinned.
- [ ] `Documentation_template.md` filled in with: methodology, blocking/candidate-generation strategy (with your recall@K and reduction-ratio numbers as evidence), model architecture and feature engineering, and any other relevant detail — no page limit, per the spec, so err toward the depth this document models.
- [ ] Every external/pretrained component used (if any) is named with its license, and the compliance sign-off checklist in §14 has been run against the final code tree.
- [ ] Full pipeline has been timed end-to-end on the full test set at least once before the deadline, not only on a dev sample.
