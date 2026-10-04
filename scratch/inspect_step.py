import json

path = 'C:/Users/marup/.gemini/antigravity-ide/brain/5f465b9c-88f5-49bf-97d0-248cd9bc01cc/.system_generated/logs/transcript_full.jsonl'
with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if '"step_index":545' in line:
            d = json.loads(line)
            print("Step 545:", json.dumps(d, indent=2)[:2000])
