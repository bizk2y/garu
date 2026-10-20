"""Add a weak complete-utterance EF filter, preserving the exact existing split texts."""
import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('destination')
parser.add_argument('--data-root', default=str(Path.home() / 'workspace/data'))
args = parser.parse_args()
source, destination = Path(args.source), Path(args.destination)
destination.mkdir(exist_ok=True)
corpus = Path(args.data_root) / 'nikl_mp_2021/SXMP1902008031.json'
data = json.loads(corpus.read_text())
complete = {}
for doc in data['document']:
    for sent in doc['sentence']:
        tags = [m['label'].split('-')[0] for m in (sent.get('MP') or sent.get('morpheme') or [])
                if m['label'].split('-')[0] not in {'SF', 'SP', 'SS', 'SE', 'SO', 'SW'}]
        complete[sent['id']] = bool(tags and tags[-1] == 'EF')
counts = {}
for split in ['train', 'dev', 'test']:
    count = 0
    with (destination / (split + '.jsonl')).open('w') as out:
        for line in (source / (split + '.jsonl')).open():
            row = json.loads(line)
            if row['source'] == 'SX':
                row['sentence_eligible'] = all(complete[sid] for sid in row['sentence_ids'])
                count += row['sentence_eligible']
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    counts[split] = count
manifest = json.loads((source / 'manifest.json').read_text())
manifest['sentence_label_scope'] += '; additionally SX blocks where every source utterance ends in MP EF after punctuation removal; weak completeness labels, not adjudicated sentence gold'
manifest['speech_eligible_rows'] = counts
(destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print(counts)
