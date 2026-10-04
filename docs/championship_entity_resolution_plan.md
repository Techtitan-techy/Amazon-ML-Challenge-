# Amazon ML Challenge 2026 — Championship Entity Resolution Plan

**Purpose:** Build a competition-grade Business Entity Resolution system that is stronger than the publicly documented participant solutions we reviewed, while preserving precision, recall, entity-level Macro F0.5, and candidate-reduction performance.

**Important:** No plan can guarantee a 100% leaderboard win. The objective of this document is to establish the strongest empirically testable architecture, with strict promotion gates so that every optimization must improve the actual competition metric without creating hidden regressions.

---

## 0. Evidence Base

This plan combines:

1. Our internal `reports/` experiments and audit.
2. The supplied five-experiment protocol.
3. Public participant repositories from Amazon ML Challenge 2026.

### Public participant evidence reviewed

| Repository | Publicly reported approach/result | What we should learn |
|---|---|---|
| Akash Bardia | LightGBM/HGB, 28–32 features, threshold tuning, global consistency, source-specific thresholds, LGB+HGB ensemble; reports Macro F0.5 up to **0.976653** | Entity-level decision logic and model diversity matter, not just the classifier |
| Ayan Ahmed Khan | 5 TF-IDF blocking views, learned native→Latin dictionary, abbreviation maps, reverse retrieval, static multilingual embeddings, stage-1 LGBM, collective/sibling features, stage-2 LGBM, threshold + exclusivity | Stronger retrieval representations + learned collective features are a major next frontier |
| Madhav064 | 5 sparse TF-IDF channels, Indic transliteration, legal/DBA normalization, phonetic skeleton, rarity, house-number and pool-competition features, exclusivity and expected-F0.5 decisions | Representation diversity and entity-level competition features |
| Team aads | Compact uint32 inverted indexes, frequent-key pruning, batching, candidate cap, vectorized RapidFuzz, XGBoost | Engineering/scalability techniques allow broader retrieval |
| PrateekTechie | High-recall blocking, pairwise features, XGBoost, F0.5-focused decisions, SQLite/cache architecture | Reproducibility and scalable pipeline design |
| Sid-techweb | Complete pipeline with reported validation Macro F0.5 around **0.978**, including threshold tuning and exact competition metric | Confirms that ~0.978 is a credible public benchmark to beat |
| Other public 2026 repos | TF-IDF + LightGBM/XGBoost, embedding KNN, chunked execution, strict validator integration | Common baseline; should not be our final differentiator |

Sources:
- https://github.com/Akash-bardia/amazon-ml-challenge-2026
- https://github.com/AyanAhmedKhan/amazon-ml-challenge
- https://github.com/Madhav064/Amazon-ML-Challenge-2026
- https://github.com/swaindhruti/Amazon-ML-Challenge-2026
- https://github.com/PrateekTechie/business-entity-resolution-amazon-ml
- https://github.com/Sid-techweb/AmazonML-New

### Evidence caveat

Public repository scores are self-reported validation results and are not proof of final leaderboard rank. We use them as engineering benchmarks, not as guaranteed comparable measurements.

---

# 1. Competition Objective

The optimization target is:

## Primary metric

**Entity-level Macro F0.5**

Precision is weighted more heavily than recall.

Therefore:

> A high pairwise F0.5 is insufficient.

The pipeline must optimize the complete entity decision:

`S1 → zero / one / many target matches`

including zero-match singletons.

## Secondary objectives

1. Candidate recall
2. Candidate reduction
3. Pair precision
4. Pair recall
5. Singleton precision
6. One-match recall
7. Multi-match recall
8. Cross-source consistency
9. Country robustness, including France
10. Runtime and memory
11. Submission-format correctness

---

# 2. Current Internal Baseline

Our current evidence establishes approximately:

- Pair F0.5: ~0.99
- Entity Macro F0.5: ~0.966
- XGBoost is already a strong pair classifier.
- Hybrid Word/Character TF-IDF retrieval reaches approximately **99.31% candidate recall** at approximately **47.77 candidates/S1**.
- Fixed K=30 retrieval in another candidate experiment gives approximately **98.35% recall**.
- Aggressive adaptive K can reduce candidate volume substantially but loses recall.
- Missing-address and cross-script examples dominate important residual errors.
- Higher decision thresholds around 0.80–0.90 performed better than low thresholds in the existing sweep, but this must be revalidated on an untouched grouped split.

## Main conclusion

**The model is not the primary bottleneck.**

The largest remaining opportunities are:

1. retrieval coverage,
2. candidate ranking,
3. hard-negative precision,
4. entity-level decision logic,
5. collective/cross-source evidence,
6. controlled model diversity.

---

# 3. Championship Architecture

```text
RAW S1 / S2 / S3
        |
        v
[1] Multi-view normalization
        |
        +-- Unicode / punctuation
        +-- legal forms
        +-- abbreviations
        +-- DBA/domain parsing
        +-- address parsing
        +-- transliteration
        +-- phonetic representation
        +-- numeric / postal representation
        |
        v
[2] Multi-channel retrieval
        |
        +-- word TF-IDF
        +-- name TF-IDF
        +-- address TF-IDF
        +-- character TF-IDF
        +-- exact anchors
        +-- rare-token retrieval
        +-- numeric retrieval
        +-- transliteration retrieval
        +-- optional static multilingual embedding retrieval
        |
        v
[3] High-recall candidate union
        |
        | target: >=99.5% candidate recall
        |
        v
[4] Learned candidate reranker
        |
        | target: 25–35 candidates/S1
        | subject to recall constraint
        |
        v
[5] Pair feature engine
        |
        +-- lexical similarity
        +-- address similarity
        +-- numeric evidence
        +-- rarity
        +-- retrieval evidence
        +-- source/country
        +-- missingness
        +-- competition features
        +-- cross-script evidence
        |
        v
[6] Diverse pair models
        |
        +-- XGBoost
        +-- LightGBM
        +-- HistGradientBoosting
        +-- optional calibrated embedding model
        |
        v
[7] OOF calibration + hard-negative correction
        |
        v
[8] Collective / entity features
        |
        +-- candidate rank
        +-- score margin
        +-- sibling support
        +-- target competition
        +-- cross-source support
        +-- collision evidence
        |
        v
[9] Entity-level decision engine
        |
        +-- source-specific thresholds
        +-- expected-F0.5 decision
        +-- verified exclusivity
        +-- multi-match handling
        +-- singleton defense
        |
        v
[10] Final graph consistency
        |
        v
matching_results.tsv
candidate_pairs.tsv
        |
        v
official validator
```

---

# 4. Stage 1 — Multi-View Normalization

## Goal

Create representations that expose equivalent entities without destroying original evidence.

### 4.1 Name representations

Generate:

- Unicode normalized
- lowercase
- punctuation-stripped
- whitespace-normalized
- legal-form normalized
- abbreviation-expanded view
- DBA/domain-cleaned view
- tokenized view
- character n-gram view
- transliterated view
- phonetic skeleton
- numeric-token view

### 4.2 Address representations

Generate:

- normalized address
- tokenized address
- house number
- postal code
- city/locality
- state/region
- street tokens
- numeric tokens
- character n-grams
- abbreviation-normalized address

### 4.3 Cross-script

Train any transliteration/alias dictionaries **only from the competition data**.

Examples of useful learned mappings:

- native script token ↔ Latin token
- repeated abbreviation ↔ expanded form
- recurring legal suffix ↔ normalized form

Do not use external business databases.

### Promotion gate

Normalization is promoted only if it improves at least one of:

- retrieval recall,
- FN recovery,
- Macro F0.5,

without materially degrading precision.

---

# 5. Stage 2 — High-Recall Retrieval

This is the highest-priority stage.

## Required retrieval channels

### R1 — Full-record word TF-IDF

Primary backbone.

### R2 — Name-only word TF-IDF

Protects name-dominant cases.

### R3 — Address-only word TF-IDF

Protects address-dominant cases.

### R4 — Name character TF-IDF

Recovers:

- typos
- transpositions
- OCR-like corruption
- spelling variations

### R5 — Address character TF-IDF

Recovers:

- address formatting changes
- spelling variations
- abbreviations

### R6 — Exact normalized anchors

Very high precision.

### R7 — Rare-token retrieval

Use IDF-weighted distinctive tokens.

### R8 — Numeric retrieval

Use:

- house number
- postal code
- meaningful numeric tokens

### R9 — Transliteration retrieval

Explicitly recover cross-script pairs.

### R10 — Domain/DBA retrieval

Handle:

`foo.com`, `foo-bar.com`, joined business names, DBA names.

### R11 — Optional embedding retrieval

Only if a licensed model within challenge constraints demonstrates incremental recall on missed-GT pairs.

---

# 6. Retrieval Union

The candidate pool should be:

```text
UNION(
    exact,
    word,
    name,
    address,
    char-name,
    char-address,
    transliteration,
    rare-token,
    numeric,
    domain
)
```

The union should be measured against ground truth.

## Target

**Candidate recall >=99.5%**

Stretch target:

**>=99.8%**

Candidate count is secondary to recall at this stage.

---

# 7. Stage 3 — Learned Candidate Reranking

Current hybrid retrieval is approximately:

`99.31% recall / 47.77 candidates per S1`

The objective is:

> compress the candidate pool without dropping true matches.

## Candidate-ranker features

Use:

- word TF-IDF cosine
- name TF-IDF cosine
- address TF-IDF cosine
- character cosine
- transliteration similarity
- exact-name flag
- exact-address flag
- token Jaccard
- rare-token overlap
- numeric overlap
- postal agreement
- house-number agreement
- retrieval rank
- best retrieval rank
- number of independent channels retrieving the candidate
- channel agreement score
- candidate frequency/density

## Ranking objective

Train specifically for:

`Recall@K`

not ordinary classification accuracy.

Sweep:

```text
K = 20
K = 25
K = 28
K = 30
K = 32
K = 35
K = 40
```

## Promotion gate

Preferred:

```text
K <= 30
Recall >= 99.0%
```

Championship target:

```text
K <= 35
Recall >= 99.5%
```

Never choose K solely because it reduces candidate count.

---

# 8. Stage 4 — Pair Feature Engine

Build a comprehensive but controlled feature matrix.

## Name features

- Jaro-Winkler
- Levenshtein
- normalized edit similarity
- Jaccard
- token overlap
- containment
- token count
- length ratio
- character cosine
- transliteration similarity
- rare-token overlap
- prefix/suffix agreement

## Address features

- Jaro-Winkler
- Levenshtein
- Jaccard
- token overlap
- address cosine
- house-number equality
- postal equality
- city equality
- state equality
- numeric overlap
- street overlap
- locality overlap

## Cross-field features

Examples:

```text
name strong + address weak
name weak + address strong
name moderate + house number exact
postal exact + locality strong
rare token exact + address moderate
```

## Retrieval features

- rank in each channel
- best rank
- best score
- number of retrieval channels
- retrieval consensus

## Competition features

Borrow the strongest idea from the public Ayan/Madhav-style architectures:

- target competition
- candidate density
- rank margin
- number of candidates above probability bands
- sibling/copy support
- collective score support

---

# 9. Stage 5 — Pair Model Ensemble

Benchmark:

1. XGBoost
2. LightGBM
3. HistGradientBoosting

Use identical candidate sets and identical folds.

## Required diagnostics

Measure:

- PR-AUC
- pair precision
- pair recall
- pair F0.5
- entity Macro F0.5
- singleton precision
- OOF probability correlation
- error overlap

### Ensemble only if it adds real diversity

Do not ensemble models whose predictions/errors are effectively identical.

Test:

```text
XGB
LGB
HGB
40/40/20
50/30/20
33/33/34
rank-average
calibrated probability average
```

## Promotion gate

Keep the ensemble only if:

- entity Macro F0.5 improves,
- precision does not materially regress,
- singleton performance does not regress,
- the improvement survives an untouched grouped split.

---

# 10. Stage 6 — Hard-Negative Mining

This should be iterative.

## Mine:

### H1 — High-score singleton false positives

Most important.

### H2 — Same/common business name

### H3 — Same building/address

### H4 — Same postal/city

### H5 — Transliteration collision

### H6 — Domain/DBA collision

### H7 — High TF-IDF but wrong entity

### H8 — Candidate collisions between multiple S1 entities

The training set should become increasingly difficult.

## Loop

```text
model
  ↓
score validation candidates
  ↓
find high-confidence errors
  ↓
classify error type
  ↓
add hard negatives
  ↓
retrain
  ↓
re-evaluate
```

Stop when Macro F0.5 stops improving.

---

# 11. Stage 7 — Probability Calibration

Candidate probabilities must be meaningful enough for entity-level decisions.

Test:

- Platt scaling
- isotonic regression
- beta calibration if useful

Measure:

- Brier score
- calibration curve
- Macro F0.5
- singleton precision
- precision at high probability

Do not select calibration based on Brier alone.

The final objective remains entity Macro F0.5.

---

# 12. Stage 8 — Collective / Sibling Features

This is the major idea to borrow from the strongest-looking public architecture.

For each S1, compute:

- candidate rank
- p1
- p2
- p3
- p1-p2 margin
- candidate score density
- number of strong candidates
- number of retrieval channels supporting candidate
- target-side competition
- sibling support from related candidates
- cross-source support

## Why

A pair's probability should depend partly on the **context in which it competes**.

Example:

```text
Candidate A: 0.96
Candidate B: 0.95
Candidate C: 0.20
```

is fundamentally different from:

```text
Candidate A: 0.96
Candidate B: 0.42
Candidate C: 0.15
```

The first is ambiguous.

The second is decisive.

---

# 13. Stage 9 — Verified Global Consistency

Do not blindly assume target uniqueness.

First measure the training ground truth:

```text
target ID
→ number of distinct S1 entities
```

If the training ground truth establishes strict target uniqueness, use it as a constraint.

If not, use a learned penalty rather than a hard constraint.

## Collision policy

For each target appearing under multiple S1 predictions:

```text
compare:
    calibrated probability
    margin
    retrieval consensus
    feature evidence
    entity context
```

Then resolve collisions only when confidence is sufficiently asymmetric.

Do not automatically suppress near-ties.

---

# 14. Stage 10 — Source-Specific Decision Policy

Do not assume S2 and S3 have identical probability calibration.

Search:

```text
tau_S2 ∈ {0.70,0.75,0.80,0.85,0.90}
tau_S3 ∈ {0.70,0.75,0.80,0.85,0.90}
```

Then optionally test country/source interactions if sample size supports them.

Optimize:

**entity Macro F0.5**

not pair F0.5.

---

# 15. Stage 11 — Expected-F0.5 Decision Engine

Instead of a simple threshold:

```text
p >= tau
```

evaluate candidate sets directly.

For each S1:

```text
candidate set C
candidate probabilities
candidate ranks
entity type signals
```

Estimate which accepted subset maximizes expected entity F0.5.

Candidate actions:

- empty set
- top-1
- top-2
- top-k
- thresholded subset

This is particularly valuable because the official metric is calculated per S1 entity.

---

# 16. Stage 12 — Singleton Defense

Singletons deserve a dedicated subsystem because a false positive can turn an otherwise perfect entity score into zero.

## Inputs

- top probability
- second probability
- margin
- name similarity
- address similarity
- rare-token agreement
- numeric agreement
- retrieval consensus
- target competition
- hard-negative indicators

## Decision tiers

### Tier A — very strong isolated evidence

Accept.

### Tier B — strong but ambiguous

Require additional independent evidence.

### Tier C — weak/highly ambiguous

Prefer empty.

Never claim that a fixed 0.40 floor provides perfect singleton protection.

---

# 17. Stage 13 — Cross-Source S2/S3 Evidence

For an S1 with both S2 and S3 candidates:

```text
S1
├── S2 candidate
└── S3 candidate
```

calculate:

- name similarity between S2/S3
- address similarity
- postal compatibility
- house-number compatibility
- city compatibility
- cross-source retrieval agreement

Use these as learned features.

## Important

Do not automatically reject two candidates merely because their addresses differ.

Businesses can have:

- branches,
- registered/operating addresses,
- stale records,
- crawler differences.

Address conflict should initially be a feature/penalty.

---

# 18. Stage 14 — Cross-Script Specialist

Build explicit recovery channels for:

- Devanagari ↔ Latin
- Gujarati ↔ Latin
- other observed Indic scripts
- accented Latin variants
- transliteration variants

Measure:

```text
cross-script candidate recall
cross-script pair recall
cross-script Macro F0.5
```

The goal is not to transliterate everything into one lossy representation.

Keep:

```text
native
normalized native
Latin transliteration
```

simultaneously.

---

# 19. Stage 15 — Missing-Address Specialist

When address is missing:

Do not force the normal feature regime.

Use:

- name retrieval
- character name retrieval
- transliteration retrieval
- rare token evidence
- legal-form-normalized name
- domain/DBA evidence
- numeric/name evidence

When address exists:

Use full address evidence.

The model should explicitly know:

```text
both_address_present
s1_address_present
target_address_present
```

and interpret similarity conditional on missingness.

---

# 20. Stage 16 — Country Robustness

Training includes US/India while test introduces France.

Do not hard-code country membership.

Use country as a feature and representation selector.

For France test records, support:

- accent normalization
- French address abbreviations
- postal extraction
- commune/locality representation
- legal-form normalization

But keep the system open-set.

## Promotion gate

France improvements must not materially degrade:

- US
- India
- S2
- S3

performance.

---

# 21. Stage 17 — Segment-Specific Scoreboard

Every experiment must produce:

| Segment | Metrics |
|---|---|
| Overall | Macro F0.5, precision, recall |
| US | Macro F0.5 |
| India | Macro F0.5 |
| France | Macro F0.5 |
| S1→S2 | P/R/F0.5 |
| S1→S3 | P/R/F0.5 |
| 0-match | precision/F0.5 |
| 1-match | recall/F0.5 |
| 2+ matches | recall/F0.5 |
| Missing address | recall |
| Cross-script | recall |
| Common names | precision |
| Domain/DBA | precision |
| High-collision targets | precision |

A global gain with a serious segment regression is not automatically accepted.

---

# 22. Oracle Diagnostics

Before changing the matcher, calculate:

## Oracle candidate F0.5

If every true candidate inside the pool were perfectly classified, what Macro F0.5 is theoretically achievable?

This separates:

```text
retrieval problem
```

from:

```text
matcher/decision problem
```

Run this for:

```text
K=10
20
30
35
40
50
raw hybrid
```

This should drive the candidate-budget decision.

---

# 23. Validation Design

Never tune everything on one validation split.

Use:

## Development split

For rapid experimentation.

## Grouped OOF validation

S1-grouped folds.

No S1 leakage across folds.

## Untouched holdout

Used only for final promotion.

## Final test

Only after model freeze.

---

# 24. Experiment Promotion Rule

Every experiment must have:

### A hypothesis

Example:

> Learned candidate ranking will preserve retrieval recall while reducing candidates.

### A primary metric

Example:

`Recall@30`

### A competition metric

`Macro F0.5`

### Regression checks

- precision
- recall
- singleton precision
- country slices
- source slices

### Promotion

An experiment is promoted only if:

```text
PRIMARY METRIC improves
AND
Macro F0.5 improves or remains statistically/operationally stable
AND
no critical segment collapses
```

---

# 25. Experiment Order

## P0 — Freeze baseline

Record:

- exact code version
- candidate files
- probabilities
- validation metrics
- random seeds

## P1 — Ground-truth structure audit

Measure target collisions and S2/S3 relationships.

## P2 — Source-specific thresholds

Very cheap.

## P3 — XGB/LGB/HGB diversity benchmark

Cheap and informative.

## P4 — Hard-negative mining

Improve precision.

## P5 — Multi-view retrieval expansion

Improve recall.

## P6 — Learned candidate reranker

Compress high-recall pool.

## P7 — Collective/sibling features

Use entity context.

## P8 — Stage-2 model

Train on stage-1 OOF probabilities + collective features.

## P9 — Expected-F0.5 decision engine

Optimize the actual entity metric.

## P10 — Cross-source graph features

Add relational evidence.

## P11 — Optional multilingual embedding retrieval

Only if measured gain exists.

## P12 — Final threshold and policy search

Freeze only after untouched validation.

---

# 26. Stronger Two-Stage Learning Architecture

The final preferred model is:

```text
STAGE 1
candidate pair evidence
        ↓
LightGBM / XGBoost
        ↓
OOF probability
        ↓
collective features

STAGE 2
pair probability
+ rank
+ margin
+ sibling support
+ target competition
+ cross-source evidence
+ retrieval consensus
+ country/source
        ↓
LightGBM / XGBoost
        ↓
calibrated entity-aware probability
        ↓
expected-F0.5 decision
```

This is more powerful than simply adding more pairwise similarity features to one XGBoost.

---

# 27. Optional Embedding Layer

Only after lexical retrieval is strong.

Use a small, permissively licensed multilingual model if allowed by the challenge rules.

Generate:

- name embedding cosine
- address embedding cosine
- joint-record embedding cosine

Use embeddings as:

**retrieval channel + features**

rather than replacing the tree matcher.

The public Ayan repository demonstrates this pattern with a static multilingual embedding model and then tree-based stages.

---

# 28. Efficiency Architecture

The test scale is large.

Use:

- integer record IDs
- uint32 posting lists
- sparse matrices
- batch S1 processing
- cached normalization
- cached retrieval
- cached features
- streaming output
- frequent-token pruning
- vectorized RapidFuzz
- Parquet intermediates
- deterministic seeds

Avoid:

- Python object-heavy indexes
- DataFrame merge per candidate
- repeated normalization
- repeated model inference
- storing unnecessary strings in candidate tables

---

# 29. Candidate Budget Policy

Do not use one fixed K blindly.

Preferred:

```text
easy / high-confidence retrieval:
    small K

ambiguous retrieval:
    larger K

missing address:
    larger name-oriented K

cross-script:
    specialist K

high-density/common-name:
    larger K
```

But adaptive K must be learned and validated.

The previous steep-dropoff experiment showed that aggressive truncation can lose candidate recall.

Therefore:

> candidate reduction is subordinate to candidate recall.

---

# 30. Target Performance Gates

These are engineering targets, not guarantees.

| Layer | Minimum | Championship target |
|---|---:|---:|
| Candidate recall | 99.0% | **99.5–99.8%** |
| Candidates/S1 | ≤40 | **25–35** |
| Pair precision | ≥98.8% | **≥99.2%** |
| Pair recall | ≥99.0% | **≥99.5%** |
| Pair F0.5 | ≥0.99 | **≥0.995** |
| Entity Macro F0.5 | ≥0.970 | **≥0.980+** |
| Singleton precision | ≥99% | **≥99.5%** |
| Critical segment regression | none | **none** |
| Validator | PASS | **PASS** |

A target is not considered achieved until it survives untouched validation.

---

# 31. Competitive Benchmark

Known public validation claims currently give us approximately:

```text
Public benchmark A:
Macro F0.5 ≈ 0.9767

Public benchmark B:
Macro F0.5 ≈ 0.978
```

Our internal baseline:

```text
Macro F0.5 ≈ 0.966
```

Therefore our immediate objective is:

```text
0.966
  ↓
0.970+
  ↓
0.975+
  ↓
0.980+
```

The purpose of the architecture is to create measurable headroom above the strongest public validation numbers we have found.

Do not claim that a higher validation score guarantees a higher leaderboard position.

---

# 32. What We Should NOT Do

## Do not

- blindly increase model size,
- blindly increase K,
- blindly reduce K,
- optimize pair F0.5 only,
- use a single global threshold without testing,
- hard-code France handling,
- hard-code target exclusivity before validating it,
- hard-reject address conflicts,
- use external business databases,
- use geocoding,
- use external entity lookup,
- add a Transformer merely because it is sophisticated,
- ensemble models without measuring error diversity.

---

# 33. Final Championship Pipeline

```text
                 ┌─────────────────────┐
                 │       DATA          │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ MULTI-VIEW TEXT     │
                 │ + ADDRESS NORMALIZE │
                 └──────────┬──────────┘
                            ↓
        ┌────────────────────────────────────────┐
        │          HIGH-RECALL RETRIEVAL         │
        │                                        │
        │ word | char | name | addr | exact     │
        │ translit | numeric | rare | domain    │
        └────────────────────┬───────────────────┘
                             ↓
                     ≥99.5% RECALL
                             ↓
                ┌─────────────────────┐
                │ CANDIDATE RERANKER   │
                └──────────┬──────────┘
                           ↓
                     25–35 / S1
                           ↓
        ┌──────────────────────────────────────┐
        │        PAIR FEATURE ENGINE           │
        │ lexical + address + numeric + IDF   │
        │ retrieval + rarity + competition    │
        └──────────────────┬───────────────────┘
                           ↓
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
        XGB              LGB              HGB
          └────────────────┼────────────────┘
                           ↓
                    OOF CALIBRATION
                           ↓
                  HARD-NEGATIVE LOOP
                           ↓
              COLLECTIVE / SIBLING FEATURES
                           ↓
                     STAGE-2 MODEL
                           ↓
               ENTITY DECISION ENGINE
                           ↓
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
       threshold       margin          expected F0.5
                           ↓
                 VERIFIED CONSISTENCY
                           ↓
                CROSS-SOURCE EVIDENCE
                           ↓
                  FINAL MATCH SET
                           ↓
              OFFICIAL VALIDATOR
```

---

# 34. Final Definition of “Best”

We do not define “best” as:

> highest pair F0.5.

We define it as:

> **highest validated entity-level Macro F0.5 while preserving candidate recall, singleton precision, multi-match recall, country robustness, and submission correctness under the challenge constraints.**

The pipeline should therefore be considered championship-ready only when:

1. Candidate recall is near its maximum.
2. Candidate volume is compressed efficiently.
3. Pair precision remains extremely high.
4. Pair recall remains extremely high.
5. Singleton false positives are aggressively controlled.
6. Multi-match entities are handled correctly.
7. S2/S3 relational evidence is exploited.
8. Cross-script cases are recovered.
9. Missing-address cases are recovered.
10. France is handled without hard-coded assumptions.
11. Model diversity produces measurable gain.
12. Entity-level decisions optimize Macro F0.5.
13. Every improvement survives untouched validation.
14. Official output validation passes.

---

# 35. Immediate Execution Checklist

### First

- [ ] Freeze current baseline.
- [ ] Save OOF probabilities.
- [ ] Save current candidate sets.
- [ ] Save segment metrics.

### Then

- [ ] Audit target collisions.
- [ ] Run 25-combination S2/S3 threshold search.
- [ ] Benchmark XGB/LGB/HGB.
- [ ] Measure prediction correlations.
- [ ] Mine singleton hard negatives.
- [ ] Build multi-view retrieval union.
- [ ] Train Recall@K candidate reranker.
- [ ] Test K=20/25/28/30/32/35/40.
- [ ] Build collective features.
- [ ] Train stage-2 model.
- [ ] Test expected-F0.5 decisions.
- [ ] Add cross-source evidence.
- [ ] Test multilingual embedding retrieval.
- [ ] Run untouched validation.
- [ ] Freeze champion.
- [ ] Generate exact final candidate TSV.
- [ ] Generate exact final matching TSV.
- [ ] Run official validator.
- [ ] Package only after PASS.

---

# 36. Decision Rule

At every step:

```text
                    EXPERIMENT
                         |
                         v
             Does candidate recall improve?
                    /          \
                  no            yes
                  |              |
             reject/park         v
                         Does Macro F0.5 improve?
                              /        \
                            no          yes
                            |            |
                         reject          v
                               Any segment regression?
                                  /           \
                                yes            no
                                |               |
                              reject           KEEP
```

This prevents leaderboard-oriented overfitting and keeps the system focused on the actual scoring function.

---

## Bottom Line

The public solutions establish a strong baseline around the **0.976–0.978 validation range**. The most valuable ideas we found are:

- multi-view retrieval,
- learned cross-script/abbreviation representations,
- collective/sibling features,
- two-stage boosting,
- source-specific thresholds,
- verified exclusivity,
- hard-negative mining,
- candidate ranking,
- expected-F0.5 decisions.

Our proposed system goes one level further by combining those ideas with the strongest parts of our existing experiments:

**~99.5% retrieval → learned 25–35 candidate compression → rich pair evidence → diverse OOF models → collective stage-2 features → hard-negative precision loop → entity-level expected-F0.5 decisions → verified graph consistency.**

That is the architecture to implement if the goal is to push beyond the public participant benchmarks rather than merely reproduce them.
