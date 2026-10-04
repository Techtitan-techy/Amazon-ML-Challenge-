import duckdb
import os
import json
import time

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"
con = duckdb.connect()

gt_path = os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv').replace("\\", "/")
s1_path = os.path.join(DATA_DIR, 'train', 'train_source1.tsv').replace("\\", "/")
s2_path = os.path.join(DATA_DIR, 'train', 'train_source2.tsv').replace("\\", "/")
s3_path = os.path.join(DATA_DIR, 'train', 'train_source3.tsv').replace("\\", "/")

print("--- Analyzing Ground Truth Match Cardinalities ---")
con.execute(f"""
    CREATE OR REPLACE TEMP TABLE gt_raw AS 
    SELECT source1_entity_id, matched_entity_ids 
    FROM read_csv('{gt_path}', delim='\\t', header=True, all_varchar=True, quote='')
""")

con.execute(f"""
    CREATE OR REPLACE TEMP TABLE s1_meta AS
    SELECT entity_id, country
    FROM read_csv('{s1_path}', delim='\\t', header=True, all_varchar=True, quote='')
""")

res = con.execute("""
    SELECT 
        m.country,
        count(*) as total_s1,
        count(CASE WHEN g.matched_entity_ids IS NULL OR length(trim(g.matched_entity_ids)) = 0 THEN 1 END) as singletons,
        count(CASE WHEN g.matched_entity_ids IS NOT NULL AND length(trim(g.matched_entity_ids)) > 0 THEN 1 END) as with_matches
    FROM gt_raw g
    JOIN s1_meta m ON g.source1_entity_id = m.entity_id
    GROUP BY m.country
""").df()
print("Country breakdown for S1:")
print(res)

print("\nUnnesting ground truth pairs to count total true pairs and S2 vs S3 distribution...")
t0 = time.time()
gt_pairs = con.execute("""
    SELECT 
        source1_entity_id,
        unnest(string_split(matched_entity_ids, ',')) as target_id
    FROM gt_raw
    WHERE matched_entity_ids IS NOT NULL AND length(trim(matched_entity_ids)) > 0
""").fetch_df()
print(f"Total true pairs: {len(gt_pairs):,} (took {time.time()-t0:.2f}s)")

gt_pairs['prefix'] = gt_pairs['target_id'].str[:2]
print("\nTarget ID prefix distribution in Ground Truth:")
print(gt_pairs['prefix'].value_counts())

match_counts = gt_pairs.groupby('source1_entity_id').size().value_counts().sort_index()
print("\nMatch count per non-singleton S1 distribution (top 10):")
print(match_counts.head(10))
