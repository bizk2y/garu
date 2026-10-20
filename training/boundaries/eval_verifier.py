"""Evaluate frozen complete JS paths; never fit a model or select thresholds here."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from eval_natural import score_natural
from eval_natural import load_natural
from boundary import pack_parts, source_sha

MODEL_NAMES = ['baseline', 'baseline_quote', 'full', 'structured', 'no_morph', 'easy']


def prepare_external(lab, root):
    source = root / 'nikl_mp_2025/NXMP2502512313.json'
    rows, info = load_natural([lab / 'data/train.jsonl'], source)
    with (lab / 'natural.jsonl').open('w') as stream:
        for row in rows:
            row.update(id=row['paragraph_id'] + ':' + row['variant'], source='NX')
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    archive = root / 'korean_text_boundaries/20261020-label-review'
    manifest = json.loads((archive / 'manifest.json').read_text())
    refs_path = archive / 'heldout-boundary-review.json'
    if source_sha(refs_path) != manifest['files'][refs_path.name]['sha256']:
        raise ValueError('Held-out references changed')
    refs = json.loads(refs_path.read_text())
    wanted = {r['sentence_id']: r for r in refs['records'] if r['status'] == 'reviewed'}
    corpus = root / 'nikl_mp_2021/SXMP1902008031.json'
    if source_sha(corpus) != manifest['source']['sha256']:
        raise ValueError('Held-out source corpus changed')
    data = json.loads(corpus.read_text())
    found = set()
    with (lab / 'reviewed.jsonl').open('w') as stream:
        for doc in data['document']:
            for sentence in (doc or {}).get('sentence') or []:
                if sentence['id'] not in wanted:
                    continue
                reference = wanted[sentence['id']]
                text = sentence['form']
                if hashlib.sha256(text.encode()).hexdigest() != reference['text_sha256']:
                    raise ValueError('Held-out sentence changed')
                points = [0] + reference['boundaries'] + [len(text)]
                for stripped in [False, True]:
                    row = pack_parts([text[a:b] for a, b in zip(points, points[1:])], stripped)
                    row.update(id=sentence['id'] + ':' + row['variant'], source='SX', sentence_id=sentence['id'])
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                found.add(sentence['id'])
    if found != wanted.keys():
        raise ValueError('Missing held-out source sentences')
    (lab / 'external-manifest.json').write_text(json.dumps({
        'natural': {**info, 'source_sha256': source_sha(source)},
        'reviewed_scope': refs['scope'], 'references_sha256': source_sha(refs_path),
        'used_for_model_selection': False}, ensure_ascii=False, indent=2) + '\n')


def evaluate(lab, runtime, datasets):
    report = {'model_selection': 'All epochs/thresholds frozen on development data. External scores do not choose checkpoints.',
              'models': {}, 'evaluations': {}}
    report['quote_policy'] = 'All verifier paths and baseline_quote preserve matched quotation spans. Original reference labels unchanged. Added after inspecting fixed2025 failures; exploratory policy refinement, not untouched generalization.'
    for name in MODEL_NAMES[2:]:
        path = lab / 'models' / (name + '.json')
        report['models'][name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
    for split in datasets:
        started = time.monotonic()
        source = lab / (split + '.jsonl') if split in ['natural', 'reviewed'] else lab / 'data' / (split + '.jsonl')
        with source.open() as src, (lab / (split + '-verified.jsonl')).open('w') as dst, (lab / (split + '-verified.log')).open('w') as log:
            subprocess.run(['node', str(Path(__file__).with_suffix('.mjs')), str(runtime), str(lab / 'models')],
                           stdin=src, stdout=dst, stderr=log, check=True)
        rows = [json.loads(line) for line in source.open()]
        outputs = [json.loads(line) for line in (lab / (split + '-verified.jsonl')).open()]
        assert len(rows) == len(outputs) and all(a['id'] == b['id'] for a, b in zip(rows, outputs))
        measured = {name: score_natural(rows, [row['boundaries'][name] for row in outputs]) for name in MODEL_NAMES}
        report['evaluations'][split] = {'rows': len(rows), 'invariant_checks': len(rows) * len(MODEL_NAMES),
                                        'scores': measured, 'seconds_all_paths': time.monotonic() - started}
        if split not in ['natural', 'reviewed']:
            report['evaluations'][split]['by_source'] = {
                src: {name: score_natural([r for r in rows if r['source'] == src],
                    [o['boundaries'][name] for r, o in zip(rows, outputs) if r['source'] == src]) for name in MODEL_NAMES}
                for src in ['NX', 'SX']}
        (lab / 'external-evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(split, json.dumps(measured), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lab', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, default=Path(__file__).resolve().parents[2] / 'js/dist')
    parser.add_argument('--datasets', nargs='+', default=['natural', 'reviewed', 'test'])
    parser.add_argument('--data-root', type=Path, default=Path.home() / 'workspace/data')
    args = parser.parse_args()
    prepare_external(args.lab, args.data_root)
    evaluate(args.lab, args.runtime, args.datasets)
