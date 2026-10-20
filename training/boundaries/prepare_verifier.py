"""Prepare source-paragraph supervision and conservative spoken sentence pairs."""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from boundary import pack_parts, leak_key, source_sha

MAX_PARAGRAPH_CODEPOINTS = 2048
PUNCTUATION = {'SF', 'SP', 'SS', 'SE', 'SO', 'SW'}
QUOTES = frozenset('\"\'“”‘’「」『』')


def paragraph_groups(sentences):
    groups, previous = [], None
    seen = set()
    for sentence in sentences:
        key, number = sentence['id'].rsplit('.', 1)
        if key != previous:
            if key in seen:
                raise ValueError('Paragraph reappears out of order')
            groups.append([])
            seen.add(key)
            previous = key
        if int(number) != len(groups[-1]) + 1:
            raise ValueError('Nonconsecutive source sentence IDs')
        groups[-1].append(sentence)
    return groups


def complete_utterance(sentence):
    if set(sentence['form']) & QUOTES:
        return False
    tags = [m['label'].split('-')[0] for m in sentence.get('morpheme') or sentence.get('MP') or []]
    lexical = [tag for tag in tags if tag not in PUNCTUATION]
    return bool(lexical and lexical[-1] == 'EF' and lexical.count('EF') == 1
                and tags.count('SF') <= 1 and not any(tag == 'SF' for tag in tags[:-1]))


def spoken_groups(sentences):
    groups, block = [], []
    for sentence in sentences:
        if not complete_utterance(sentence):
            if block:
                groups.append(block)
            block = []
            continue
        block.append(sentence)
        if len(block) == 2:
            groups.append(block)
            block = []
    if block:
        groups.append(block)
    return groups


def document_split(doc_id):
    value = int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) % 10
    return 'test' if value == 0 else 'dev' if value == 1 else 'train'


def group_keys(texts):
    keys = {leak_key(text) for text in texts}
    keys.update(leak_key(pack_parts(texts, stripped)['text']) for stripped in [False, True])
    return keys


def reference_keys(root, references):
    records, by_file = [], {}
    for path in references:
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                records.append(row)
                by_file.setdefault(row['source_file'], set()).update(row['sentence_ids'])
    texts = {}
    for filename, wanted in by_file.items():
        data = json.loads((root / 'nikl_mp_2021' / filename).read_text())
        for doc in data['document']:
            for sentence in (doc or {}).get('sentence') or []:
                if sentence['id'] in wanted:
                    texts[(filename, sentence['id'])] = sentence['form']
    keys = set()
    for row in records:
        keys.update(group_keys([texts[(row['source_file'], sid)] for sid in row['sentence_ids']]))
    return keys


def make_rows(root, repo, out):
    out.mkdir(parents=True, exist_ok=False)
    blocked, exclusion_refs = set(), []
    for path in [repo / 'training/gold_testset/gold_testset.jsonl',
                 repo / 'training/rerank/data/dev_texts.txt']:
        exclusion_refs.append({'path': str(path), 'sha256': source_sha(path)})
        for line in path.open():
            blocked.add(leak_key(json.loads(line)['text'] if path.suffix == '.jsonl' else line))
    for path in sorted((root / 'nikl_mp_2025').glob('*.json')):
        exclusion_refs.append({'path': str(path), 'sha256': source_sha(path)})
        data = json.loads(path.read_text())
        for doc in data['document']:
            for sentence in (doc or {}).get('sentence') or []:
                blocked.add(leak_key(sentence['form']))
                blocked.add(leak_key(sentence.get('original_form', sentence['form'])))
    prior_refs = [root / 'korean_text_boundaries/20261020' / ('spaces-' + split + '.refs.jsonl.gz')
                  for split in ['dev', 'test']]
    prior_boundary_keys = reference_keys(root, prior_refs)
    exclusion_refs.extend({'path': str(path), 'sha256': source_sha(path), 'scope': 'exclude_from_training'}
                          for path in prior_refs)
    pools = {split: [] for split in ['train', 'dev', 'test']}
    sources, counts = [], Counter()
    for path in sorted((root / 'nikl_mp_2021').glob('*.json')):
        source = 'NX' if path.name.startswith('NX') else 'SX'
        sources.append({'path': str(path), 'sha256': source_sha(path)})
        data = json.loads(path.read_text())
        for doc in data['document']:
            split = document_split(doc['id'])
            sentences = doc.get('sentence') or []
            groups = paragraph_groups(sentences) if source == 'NX' else spoken_groups(sentences)
            for parts in groups:
                texts = [sentence['form'] for sentence in parts]
                keys = group_keys(texts)
                row = pack_parts(texts)
                if len(row['text']) < 5 or len(row['text']) > MAX_PARAGRAPH_CODEPOINTS:
                    counts[source + '_length_excluded'] += 1
                    continue
                pools[split].append({'source': source, 'source_file': path.name,
                    'document_id': doc['id'], 'sentence_ids': [s['id'] for s in parts],
                    'parts': texts, 'part_keys': keys,
                    'group_key': leak_key(row['text']),
                    'assembly': 'source_paragraph_ids' if source == 'NX' else 'consecutive_complete_utterance_pair'})
        print('loaded', source, {k: len(v) for k, v in pools.items()}, flush=True)
    split_keys, split_docs, files = {}, {}, {}
    # Held-out partitions take precedence over training when text repeats.
    earlier = set()
    for split in ['test', 'dev', 'train']:
        seen = set()
        forbidden = blocked | earlier
        if split == 'train':
            forbidden.update(prior_boundary_keys)
        split_keys[split], split_docs[split] = set(), set()
        path = out / (split + '.jsonl')
        with path.open('w') as stream:
            for group in pools[split]:
                keys = group['part_keys'] | {group['group_key']}
                if keys & forbidden:
                    counts[split + '_excluded_overlap'] += 1
                    continue
                if group['group_key'] in seen:
                    counts[split + '_duplicate_group'] += 1
                    continue
                seen.add(group['group_key'])
                split_keys[split].update(keys)
                split_docs[split].add(group['document_id'])
                original = pack_parts(group['parts'])
                for stripped in [False, True]:
                    row = pack_parts(group['parts'], stripped)
                    if stripped and row['text'] == original['text']:
                        continue
                    row.update({key: group[key] for key in ['source', 'source_file', 'document_id', 'sentence_ids', 'assembly']})
                    row['id'] = ':'.join([group['sentence_ids'][0], row['variant']])
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    counts[split + '_' + group['source'] + '_rows'] += 1
                    counts[split + '_gold_boundaries'] += len(row['sentences'])
        earlier.update(split_keys[split])
        files[split] = {'path': str(path), 'sha256': source_sha(path)}
    for a in split_keys:
        for b in split_keys:
            if a != b:
                assert not split_keys[a] & split_keys[b]
                assert not split_docs[a] & split_docs[b]
    manifest = {'schema': 1, 'source_files': sources, 'exclusion_sources': exclusion_refs,
                'split': 'sha256(document_id) firstuint32 modulo10: test0,dev1,train2..9',
                'counts': dict(counts), 'datasets': files,
                'documents': {k: len(v) for k, v in split_docs.items()},
                'policy': 'NX original source-record boundaries, including both inside-quote positives and negatives; not consistently quote-preserving top-level boundaries. Paragraph membership from original IDs; original paragraph text unavailable. SX conservative single-final-EF record pairs remain weak labels.',
                'normalization': 'underscore->space; all whitespace deleted; alternative deletes terminal punctuation per source record; exact codepoint targets',
                'source_text_redistributed': False}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path.home() / 'workspace/data')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    make_rows(args.data_root, args.repo, args.out)
