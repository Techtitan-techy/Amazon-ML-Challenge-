import time

t0 = time.time()
input_file = "matching_results_backup_0979.tsv"
output_file = "matching_results.tsv"

# Ultra-conservative: only prune the most extreme outliers
# GT shows: max S2 per entity = 5, max S3 per entity = 6, max total = 11
# Only cap S2 at 5 (GT max), S3 at 5, total at 8
# This targets ONLY the most extreme false-positive entities

total = 0
affected = 0
pruned = 0

with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8', newline='\n') as fout:
    
    header = fin.readline()
    fout.write(header)
    
    for line in fin:
        total += 1
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        s1 = parts[0]
        
        if len(parts) <= 1 or not parts[1].strip():
            fout.write(f"{s1}\t\n")
            continue
        
        matches = [m.strip() for m in parts[1].split(',') if m.strip()]
        orig_len = len(matches)
        
        if orig_len <= 8:
            # Don't touch anything with <=8 matches
            fout.write(f"{s1}\t{','.join(matches)}\n")
            continue
        
        # Only for >8 matches: cap S2 at 5, S3 at 5, total at 8
        s2 = [m for m in matches if m.startswith('S2')]
        s3 = [m for m in matches if m.startswith('S3')]
        
        s2_cap = s2[:5]
        s3_cap = s3[:5]
        new = s2_cap + s3_cap
        new = new[:8]
        
        if len(new) < orig_len:
            affected += 1
            pruned += orig_len - len(new)
        
        fout.write(f"{s1}\t{','.join(new)}\n")

print(f"Done in {time.time()-t0:.1f}s | Entities: {total:,} | Affected: {affected:,} | Pruned: {pruned:,}")
