# Amazon ML Challenge 2026

Business entity resolution solution for the Amazon ML Challenge 2026.

## Repository layout

- `code/business_entity_resolution/` - reusable inference and matching pipeline
- `submission_package/` - publishable snapshot of the competition submission package
- `diagnostics/` - profiling, evaluation, and architecture analysis scripts
- `reports/` - generated analysis reports and benchmark data
- `docs/` - solution design, experiment notes, and challenge reference documents
- `scripts/` - exploratory and optimization scripts
- `notebooks/` - experiment notebooks
- `models/` - trained model artifacts

Competition datasets, generated TSV predictions, caches, and packaged archives are intentionally ignored by Git.

## Running the submission pipeline

See `submission_package/code/business_entity_resolution/README.md` for installation, inference, and output validation instructions. The original nested working repository remains locally at `Team_Mani_submission/` and is intentionally excluded from the outer repository.

## Main result

The documented production configuration uses country-partitioned sparse TF-IDF retrieval followed by feature-based XGBoost scoring, with `K=30` selected as the primary efficiency baseline.
