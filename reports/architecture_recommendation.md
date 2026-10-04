# Hardened Architecture Specification & Empirical Evidence
### Amazon ML Challenge 2026 — Business Entity Resolution

This document formalizes the production architecture derived from the diagnostic pipeline, addressing all 15 diagnostic phases and the end-to-end K-sweep benchmarks on the 22.77-trillion comparison search space.

---
## 1. Executive Summary: Evidence-Driven Architecture

```text
                         S1 (2.21M)
                             │
                             ▼
                   Multi-view Normalization
             (Unicode NFKC + AnyAscii + Legal Canonical)
                             │
             ┌───────────────┼────────────────┐
             │               │                │
             ▼               ▼                ▼
         Name Views    Address Views    Structured Components
             │               │          (Postal / House Num)
             └───────────────┼────────────────┘
                             ▼
                    PRIMARY RETRIEVAL
                 Word Token TF-IDF (98.81%)
                             │
                    Top-K per Channel
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
           Character TF-IDF       Exact Normalized
           (Recall Booster)       (Recall Booster)
                    │                 │
                    └────────┬────────┘
                             ▼
                       UNION + DEDUP
                     (99.31% Recall)
                             │
                             ▼
                     Candidate Budget
                     K = 30 (Efficiency) / 50 (High-Recall)
                             │
                             ▼
                    Pair Feature Engine
             (Address Jaccard, Name JW, Missingness Flag)
                             │
                             ▼
                          XGBoost
                   (PR-AUC = 0.995 - 0.996)
                             │
                             ▼
                    Entity Decision Layer
                 (Optimal Threshold = 0.70)
                             │
                             ▼
                        0 / 1 / MANY
                 (Out-of-Fold Macro F0.5 = 0.965)
```

---
## 2. Quantitative Retrieval Hierarchy

On the evaluated 20,000-S1 cohort (tested against 317,277 target records including 250,000 distractors), retrieval channels exhibited the following performance:

| Channel | Nature of Blocker | Recall on Cohort | Avg Candidates / S1 | Role in Architecture |
| :--- | :--- | :---: | :---: | :--- |
| **Word TF-IDF** | Token bag over full record | **98.81%** | 20.00 | **Primary Engine** (captures almost all true matches) |
| **Addr Char TF-IDF** | Character 3–4 grams | **91.34%** | 20.00 | **Recall Booster** (recovers address typos & reorderings) |
| **Name Char TF-IDF** | Character 3–4 grams | **80.52%** | 20.00 | **Recall Booster** (recovers name transliteration & typos) |
| **Exact Name** | Normalized equality | **25.42%** | 1.21 | **Precision Anchor** (instant high-confidence pairs) |
| **Exact Addr** | Normalized equality | **8.40%** | 0.29 | **Precision Anchor** (instant high-confidence pairs) |
| **HYBRID UNION** | Multi-channel union | **99.31%** | 47.77 | **Final Candidate Generator** (Reduction ratio: 0.99984) |

> [!IMPORTANT]
> Word-level TF-IDF is the primary backbone of the retrieval system (98.81% standalone recall). Character n-grams and exact relational keys serve as surgical recall boosters to close the final 0.50% gap.

---
## 3. Decisive End-to-End K-Sweep: Candidate Volume vs Final Macro F0.5

To determine whether higher candidate budgets translate into final leaderboard gains or introduce precision dilution, we evaluated $K \in [10, 20, 30, 40, 50]$ using 3-fold `GroupKFold` cross-validation keyed on `s1_id`:

| Candidate Budget ($K$) | Candidate Recall | Candidate Precision | Total Candidates | Optimal Threshold | Out-of-Fold Macro $F_{0.5}$ | Pair Precision | Pair Recall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 10$** | $92.75\%$ | $31.98\%$ | $200,000$ | $0.65$ | **$0.9466$** | $0.9796$ | $0.9728$ |
| **$K = 20$** | $95.37\%$ | $16.44\%$ | $400,000$ | $0.70$ | **$0.9541$** | $0.9826$ | $0.9579$ |
| **$K = 30$** | **$98.35\%$** | $11.30\%$ | $600,000$ | $0.70$ | **$0.9639$** | $0.9818$ | $0.9521$ |
| **$K = 40$** | **$98.95\%$** | $8.54\%$ | $799,048$ | $0.70$ | **$0.9646$** | $0.9816$ | $0.9494$ |
| **$K = 50$** | **$99.29\%$** | $7.29\%$ | $939,664$ | $0.70$ | **$0.9649$** | $0.9820$ | $0.9465$ |

### Key Takeaways from the End-to-End Sweep:
1. **$K=30$ is the premier efficiency configuration**: It recovers **98.35% candidate recall** and achieves **0.9639 Macro F0.5**, requiring 36% fewer candidate pairs than $K=50$.
2. **$K=50$ is the high-recall ceiling configuration**: It reaches **99.29% candidate recall** and achieves **0.9649 Macro F0.5**. The marginal gain of $+0.0010$ Macro F0.5 comes at the cost of $+339,664$ candidate pairs.
3. **Threshold Stability**: Across all budgets $K \ge 20$, the optimal decision threshold is **identically 0.70**, demonstrating strong calibration and stability.

---
## 4. Root-Cause Analysis on the 477 Residual Missed Pairs

On the 20,000-S1 evaluation cohort, **477 ground-truth pairs were not recovered** by the evaluated hybrid retrieval configuration ($0.69\%$ residual gap). Full classification reveals:

| Primary Root Cause | Missed Count | % of Missed Pairs | % of Total Evaluation GT | Description & Diagnosis |
| :--- | :---: | :---: | :---: | :--- |
| **Indic Phonetic Distance** | $169$ | $35.43\%$ | $0.245\%$ | Cross-script phonetic shift (`laiph` vs `life`, `sarvises` vs `services`, `praivet` vs `pvt`). Ranked at positions 25–60 in character TF-IDF. |
| **Missing Target Address** | $87$ | $18.24\%$ | $0.126\%$ | Target address in S2/S3 is literally `None` / `null`, coupled with slight name typo, preventing multi-field grounding. |
| **State Acronym / Truncated Address** | $79$ | $16.56\%$ | $0.115\%$ | Address compressed to state abbreviation (`WB`, `MH`, `MP`, `KA`) + house number alone. |
| **Other Residual** | $68$ | $14.26\%$ | $0.099\%$ | Displaced sub-locality or landmark reordering. |
| **Severe Name Typo / Corruption** | $48$ | $10.06\%$ | $0.070\%$ | OCR corruption or multiple character transpositions. |
| **Address Token Displacement** | $14$ | $2.94\%$ | $0.020\%$ | Street name absent in one record, building name absent in the other. |
| **Name Acronym / Extreme Truncation** | $12$ | $2.52\%$ | $0.017\%$ | Record shortened to initials or legal shell restructure. |

---
## 5. Architectural Inclusions & Explicit Exclusions

### Frozen Inclusions:
- **Multi-view Normalization**: Unicode NFKC + `anyascii` transliteration (handles Hindi, Tamil, Kannada, French accents) + legal suffix canonicalization.
- **Primary Retrieval**: Word token TF-IDF on concatenated name + address, partitioned by country label.
- **Secondary Retrieval Boosters**: Character 3–4 gram TF-IDF on name and address + exact normalized keys.
- **Candidate Budget**: $K=30$ for fast iteration and validation; $K=50$ for final submission candidate generation.
- **Matcher**: XGBoost GBDT on 16 engineered string, address, missingness, and retrieval score features.
- **Entity Decision Policy**: Flat score threshold at $0.70$. Singletons with $\max(p) < 0.40$ mapped to empty match.

### Explicit Exclusions (Empirically Justified):
- ❌ **Score-Margin Constraint**: Rejected. Degraded Macro $F_{0.5}$ from $0.9704 \to 0.9678$ because multi-match entities have valid matches with tied high scores (`best ≈ second_best`).
- ❌ **Neural Bi-Encoders / Embedding Retrieval (ANN)**: Deferred. With classical hybrid retrieval already recovering **99.31% of true matches**, neural models cannot meaningfully improve recall, while introducing memory and latency bottlenecks across 10.32M records.
- ❌ **Fixed Top-1 Matching**: Rejected. S1 entities match an average of $3.67$ records across S2 and S3; variable-cardinality multi-label decisions are mandatory.
- ❌ **External APIs / Geocoding**: Strictly prohibited by competition rules.
