import duckdb
import os
import json
import time

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"

con = duckdb.connect()

files = {
    'train_s1': os.path.join(DATA_DIR, 'train', 'train_source1.tsv'),
    'train_s2': os.path.join(DATA_DIR, 'train', 'train_source2.tsv'),
    'train_s3': os.path.join(DATA_DIR, 'train', 'train_source3.tsv'),
    'train_gt': os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv'),
    'test_s1': os.path.join(DATA_DIR, 'test', 'test_source1.tsv'),
    'test_s2': os.path.join(DATA_DIR, 'test', 'test_source2.tsv'),
    'test_s3': os.path.join(DATA_DIR, 'test', 'test_source3.tsv'),
}

print("Checking dataset files...")
results = {}
for name, fpath in files.items():
    t0 = time.time()
    escaped_path = fpath.replace("\\", "/")
    res = con.execute(f"SELECT count(*) FROM read_csv('{escaped_path}', delim='\\t', header=True, all_varchar=True, quote='')").fetchone()[0]
    elapsed = time.time() - t0
    results[name] = {"rows": res, "path": fpath, "time_sec": round(elapsed, 2)}
    print(f"{name:10s}: {res:>10,d} rows (took {elapsed:.2f}s)")

# Let's inspect ground truth breakdown
print("\nInspecting Ground Truth...")
gt_path = files['train_gt'].replace("\\", "/")
gt_stats = con.execute(f"""
    SELECT 
        count(*) as total_s1,
        count(CASE WHEN matched_entity_ids IS NOT NULL AND length(trim(matched_entity_ids)) > 0 THEN 1 END) as s1_with_matches,
        count(CASE WHEN matched_entity_ids IS NULL OR length(trim(matched_entity_ids)) = 0 THEN 1 END) as singletons
    FROM read_csv('{gt_path}', delim='\\t', header=True, all_varchar=True, quote='')
""").fetchone()

print(f"Total S1 in train_gt: {gt_stats[0]:,}")
print(f"S1 with matches:     {gt_stats[1]:,} ({gt_stats[1]/gt_stats[0]*100:.2f}%)")
print(f"Singletons:          {gt_stats[2]:,} ({gt_stats[2]/gt_stats[0]*100:.2f}%)")

# Let's check sample schema and column names
s1_sample = con.execute(f"SELECT * FROM read_csv('{files['train_s1'].replace('\\', '/')}', delim='\\t', header=True, quote='', all_varchar=True) LIMIT 3").df()
print("\nTrain S1 Columns & Sample:")
print(s1_sample)
