import json

count = 0
with open('vector_store/checkpoint.jsonl') as f:
    for line in f:
        entry = json.loads(line)
        if entry['metadata']['source_doc'] == 'veeam_backup_13_user_guide.pdf':
            count += 1

print('Total chunks for this file:', count)