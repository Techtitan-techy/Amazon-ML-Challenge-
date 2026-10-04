import duckdb
import sys
sys.stdout.reconfigure(encoding='utf-8')
from config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT

con = duckdb.connect()

print("--- Train S1 Sample ---")
df_s1 = con.execute(f"SELECT entity_id, business_name, business_address, country FROM read_csv('{TRAIN_S1}', delim='\\t', header=True, quote='', all_varchar=True) LIMIT 5").df()
for idx, r in df_s1.iterrows():
    print(f"[{r['entity_id']}] ({r['country']})\n  Name: {r['business_name']}\n  Addr: {r['business_address']}")

print("\n--- Train S2 Sample ---")
df_s2 = con.execute(f"SELECT entity_id, business_name, business_address, country FROM read_csv('{TRAIN_S2}', delim='\\t', header=True, quote='', all_varchar=True) LIMIT 5").df()
for idx, r in df_s2.iterrows():
    print(f"[{r['entity_id']}] ({r['country']})\n  Name: {r['business_name']}\n  Addr: {r['business_address']}")

print("\n--- Train S3 Sample ---")
df_s3 = con.execute(f"SELECT entity_id, business_name, business_address, country FROM read_csv('{TRAIN_S3}', delim='\\t', header=True, quote='', all_varchar=True) LIMIT 5").df()
for idx, r in df_s3.iterrows():
    print(f"[{r['entity_id']}] ({r['country']})\n  Name: {r['business_name']}\n  Addr: {r['business_address']}")

print("\n--- Train GT Sample with Join ---")
df_gt_sample = con.execute(f"""
    WITH sample_gt AS (
        SELECT source1_entity_id, string_split(matched_entity_ids, ',')[1] as target_id
        FROM read_csv('{TRAIN_GT}', delim='\\t', header=True, quote='', all_varchar=True)
        WHERE matched_entity_ids IS NOT NULL AND length(matched_entity_ids) > 0
        LIMIT 5
    )
    SELECT 
        g.source1_entity_id, 
        s1.business_name as s1_name, 
        s1.business_address as s1_addr,
        g.target_id,
        COALESCE(s2.business_name, s3.business_name) as target_name,
        COALESCE(s2.business_address, s3.business_address) as target_addr
    FROM sample_gt g
    JOIN read_csv('{TRAIN_S1}', delim='\\t', header=True, quote='', all_varchar=True) s1 ON g.source1_entity_id = s1.entity_id
    LEFT JOIN read_csv('{TRAIN_S2}', delim='\\t', header=True, quote='', all_varchar=True) s2 ON g.target_id = s2.entity_id
    LEFT JOIN read_csv('{TRAIN_S3}', delim='\\t', header=True, quote='', all_varchar=True) s3 ON g.target_id = s3.entity_id
""").df()

for idx, r in df_gt_sample.iterrows():
    print(f"\nMatch Pair #{idx+1}:")
    print(f"  S1 ({r['source1_entity_id']}): {r['s1_name']} || {r['s1_addr']}")
    print(f"  Target ({r['target_id']}): {r['target_name']} || {r['target_addr']}")
