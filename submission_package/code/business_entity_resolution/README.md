# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

## 1. Overview
This package implements a scalable, high-precision Business Entity Resolution system designed to resolve noisy, multi-source records (Source 2 and Source 3) against a deduplicated reference source (Source 1).

The architecture addresses all core challenges:
- Multi-source noise: Transliteration (`anyascii`), abbreviation expansion, and legal entity canonicalization.
- Search space reduction: Country-partitioned inverted index retrieval reducing ~22.77 trillion Cartesian comparisons to 30 candidates per entity.
- Decision policy: C++ SIMD string similarity extraction paired with an XGBoost classifier tuned for precision (Macro F_0.5 = 0.9639 at K=30, 0.9649 at K=50) with singleton floor pruning.

---

## 2. Package Architecture
```
code/business_entity_resolution/
├── src/
│   ├── normalize.py      # Multi-view text normalizer & feature extractor
│   ├── retrieval.py      # Country-partitioned sparse TF-IDF inverted index
│   ├── features.py       # C++ SIMD RapidFuzz pairwise feature engineering
│   ├── model.py          # XGBoost candidate scoring & decision policy
│   └── pipeline.py       # End-to-end command-line runner
├── requirements.txt      # Pinned dependency specifications
└── README.md             # End-to-end execution guide
```

---

## 3. Environment Setup

### Prerequisites
- Python 3.9+ (Python 3.10 to 3.13 recommended)
- 8GB+ RAM

### Installation
```bash
pip install -r requirements.txt
```

---

## 4. End-to-End Reproduction Instructions

### Step 1: Run Inference on Test Dataset
To generate both `matching_results.tsv` and `candidate_pairs.tsv` from the test data:

```bash
python src/pipeline.py \
    --data-dir ../../6ab10eb3b23ba_student_resource/student_resource/dataset/test \
    --train-dir ../../6ab10eb3b23ba_student_resource/student_resource/dataset/train \
    --output-dir ../../output \
    --k 30
```

### Options:
- `--k 30`: Efficiency configuration (Macro F0.5 = 0.9639, 30 candidates/S1).
- `--k 50`: High-recall configuration (Macro F0.5 = 0.9649, 47 candidates/S1).
- `--mode retrain`: Retrains the XGBoost classifier from the training set before running inference.

---

## 5. Output Verification
Validate the generated output files against official competition rules:

```bash
python ../../6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../6ab10eb3b23ba_student_resource/student_resource/dataset/test
```
A status of `PASS` confirms that all formatting and structural rules are satisfied.
