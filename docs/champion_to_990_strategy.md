# Strategy: From 0.988 to 0.990+ in Amazon ML Challenge 2026

## Current State Analysis (Verified)

| File | Total Links | S1 with Matches | Score |
|------|-------------|-----------------|-------|
| `matching_results_champion_0988.tsv` | 5,672,130 | 1,628,466 | **0.988** |
| `matching_results_0979_baseline.tsv` | 5,706,773 | 1,629,121 | 0.979 |
| Ground Truth (scaled to test) | ~5,996,772 | ~1,634,000 | 1.000 |

**Gap Analysis:**
- Champion under-predicts by ~325k links vs GT expectation
- Champion pruned 34,643 FP links vs baseline (raising precision)  
- Structure is correct: 81.6% both S2+S3 (GT: 80.5%)
- All 103,423 "changed" rows are ordering differences only (same IDs)
- 655 entities were fully pruned to singletons (may include FNs)

## Why the Champion Got 0.988

The CE (Cross-Encoder) notebook produced the champion by:
1. **Training multilingual-e5-base** on 1.5M+ candidate pairs (with hard negatives)
2. **Blending** GBDT scores (w) + CE scores (1-w) for uncertain pairs (band: 0.01-0.99)
3. **Applying bipartite deduplication** (each target → exactly 1 S1)
4. **Using flat threshold** from holdout calibration (~0.60-0.70)

## Strategies to Push to 0.990+

### Strategy 1: CE Ensemble (Seed 0 + Seed 1) — **Highest ROI**

The notebook was **explicitly designed** for this:
- Cell 16 description: "second run, seed 1, for an ensemble with CE_v5_base"
- `--scores` argument accepts multiple parquet files and takes MAX per pair
- Two diverse seeds → less correlated errors → better probability calibration

**Expected gain: +0.002-0.005**

**Implementation:**
```bash
# Kaggle: Run CE_v5_base (seed 0) → produces ce_out_seed0/
!python scripts/ce_kaggle.py ... --seed 0 --out /kaggle/working/ce_out_seed0

# Kaggle: Run CE_v5_base2 (seed 1) → produces ce_out_seed1/
!python scripts/ce_kaggle.py ... --seed 1 --out /kaggle/working/ce_out_seed1 \
    --scores /kaggle/input/ber-v5-s/test_scores.parquet  # <-- both F5 outputs

# The champion was seed=1 only; ensemble would take max of both
```

### Strategy 2: `expected_f_select` Instead of Flat Threshold — **Free Gain**

The pipeline already implements this — it just wasn't used in the champion!

**From the code:**
```python
# Already in src/select.py and src/run.py
best_ef = max((expected_f_select(hi, hj, p, miss_prior=mp_, power=pw) for mp_ in [0.0, 0.05, 0.1] 
               for pw in [1.0, 1.25, 1.5, 2.0]))
method = ("expected_f", best_ef[1]) if best_ef[0] > best[0] + 1e-4 else ("threshold", best[1])
```

This is automatically selected if EFS > flat threshold + 0.0001. The training job already does this grid search.

**Expected gain: +0.001-0.003** (if EFS was not active in champion)

### Strategy 3: Post-Processing — Conservative Recovery of 655 Pruned Singletons

655 S1 entities were fully pruned to empty by the champion. In GT, only 5.58% are true singletons. If any of these 655 were genuine matches:
- Recovering even 500 true matches = significant F0.5 gain

**Implementation:**
```python
# For entities pruned to empty in champion but with links in baseline:
# Apply a MORE conservative threshold (e.g., best_base_prob > 0.85 AND it's the unique top match)
# Only recover if the baseline match had no collision (unique to that S1)
```

**Expected gain: +0.0005-0.001**

### Strategy 4: OOV (Out-of-Vocabulary) Name Feature — **Enables Obfuscated Match Recovery**

From the code: `name_oov()` identifies pool records whose names are entirely OOV in S1 vocabulary.
These are the "Veoxylocira"-type replacements. Adding `oov_f` and `oov_n` features lets the model
distinguish invented-name true copies from genuine distractors.

**Expected gain: +0.001-0.002** (already in the pipeline as `--oov` flag)

## Recommended Implementation Order

### Immediate (Can run now locally):
1. **Apply EFS post-processing to champion** — re-run `expected_f_select` on the champion's raw probability scores if we have them
2. **Recover 655 pruned-to-empty entities** — conservative rule-based recovery

### On Kaggle (Requires GPU):
3. **Run CE seed 0** (if only seed 1 was run for champion)
4. **Ensemble** both CE seeds, re-apply EFS

## Key Code: Expected F Select Post-Processor

```python
def expected_f_select(s1_idx, pool_idx, p, miss_prior=0.0, power=1.0):
    """Pick the k per S1 that maximizes E[F0.5] instead of using a flat threshold."""
    from src.select import expected_f_select as efs
    return efs(s1_idx, pool_idx, p, miss_prior=miss_prior, power=power)
```

## What We Can Do Locally (Without Rerunning Kaggle)

The champion TSV only has the final matched IDs — we've lost the raw probabilities.
To improve further we need to either:
1. Re-run the full pipeline on Kaggle to get `test_scores.parquet`
2. Or apply conservative heuristics directly on the champion TSV

### Local Heuristic: Recover Singletons with High Baseline Confidence

If the baseline had a high-confidence match (the target was unique to that S1, not a collision),
and the champion pruned it (bipartite dedup killed it), we can recover it.

The baseline had 103,423 ordering differences — these are safe to accept (same entity sets).
The 34,283 pruned rows are the ones where champion removed targets.
The 655 pruned-to-empty are the highest-risk recoveries.

**Conservative approach:** Only restore a baseline match if:
1. The target appears in EXACTLY ONE S1 entity in baseline (no collision)
2. The target was removed from an entity that still has OTHER matches (so it's not a critical singleton)
3. The target is of S3 type (S3 recall appears under-represented)

This is conservative enough to almost guarantee precision doesn't drop.

## File Status

| File | Rows | Total Links | Use |
|------|------|-------------|-----|
| `matching_results_champion_0988.tsv` | 1,732,544 | 5,672,130 | Current best submission |
| `matching_results_0979_baseline.tsv` | 1,732,544 | 5,706,773 | Fallback |
| `matching_results.tsv` | 1,732,544 | 5,672,130 | Current submission |

## Submission Checklist

- [ ] Submit champion (0.988) — current best
- [ ] Implement conservative singleton recovery → target 0.989
- [ ] Run CE ensemble on Kaggle → target 0.990+
- [ ] Use EFS instead of flat threshold → target +0.001-0.003
