import glob, json

path = 'C:/Users/marup/.gemini/antigravity-ide/brain/5f465b9c-88f5-49bf-97d0-248cd9bc01cc/.system_generated/logs/transcript.jsonl'
with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get('type') == 'USER_INPUT':
                print('--- USER ---')
                print(d.get('content', '')[:300])
        except:
            pass
