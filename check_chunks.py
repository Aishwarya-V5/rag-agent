import json

count = 0
with open('vector_store/checkpoint.jsonl') as f:
    for line in f:
        entry = json.loads(line)
        if entry['metadata']['source_doc'] == 'LLM_Models_Detailed_Pricing_2026-09-15.xlsx':
            count += 1

print('Total chunks for this file:', count)