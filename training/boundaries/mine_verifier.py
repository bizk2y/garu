"""Mine candidate mistakes; easy negatives are a matched ablation, never evaluation labels."""
import argparse
import base64
import hashlib
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from boundary import hash_features, source_sha

REQUEST_BATCH = 128


def mine(args):
    source = args.repo / 'js/src/boundary-data.ts'
    text = source.read_text()
    metadata = json.loads(text.split('export const BOUNDARY_MODEL = ')[1].split(' as const;')[0])
    raw = base64.b64decode(text.split('export const BOUNDARY_WEIGHTS: string = ')[1].split(';')[0].strip().strip('"'))
    dimensions = metadata['dimensions']
    q = np.frombuffer(raw, dtype=np.int8)[dimensions:].astype(float)
    head = metadata['heads']['sentences']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise ValueError('Mining output already exists')
    started, total, controls = time.monotonic(), 0, 0
    with args.out.open('w') as output, args.out.with_suffix('.log').open('w') as log:
        process = subprocess.Popen(['node', str(Path(__file__).with_suffix('.mjs')), str(args.runtime), args.features],
                                   stdin=subprocess.PIPE, stdout=output, stderr=log, text=True)
        try:
            with args.dataset.open() as stream:
                while True:
                    lines = list(itertools.islice(stream, REQUEST_BATCH))
                    if not lines:
                        break
                    rows = [json.loads(line) for line in lines]
                    scores, spans = None, None
                    if args.random_controls:
                        matrix, spans = hash_features(rows, dimensions, metadata['features'])
                        scores = (matrix @ q) * head['scale']
                    for index, row in enumerate(rows):
                        extra = []
                        if args.random_controls:
                            a, b = spans[index]
                            values = scores[a:b]
                            gold = set(row['sentences'])
                            hard_count = sum(score > head['threshold'] and i + 1 not in gold
                                             for i, score in enumerate(values))
                            eligible = [i + 1 for i, score in enumerate(values)
                                        if score <= head['threshold'] and i + 1 not in gold]
                            seed = int(hashlib.sha256(row['id'].encode()).hexdigest()[:16], 16)
                            chosen = np.random.default_rng(seed).choice(eligible, min(hard_count, len(eligible)), replace=False)
                            extra = [{'offset': int(i), 'compactOffset': int(i), 'score': float(values[i - 1])} for i in chosen]
                            controls += len(extra)
                        process.stdin.write(json.dumps({'id': row['id'], 'text': row['text'], 'extraCandidates': extra}, ensure_ascii=False) + '\n')
                    total += len(rows)
                    if total % 4096 == 0:
                        print('mined', total, 'seconds', round(time.monotonic() - started, 1), flush=True)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('JS mining failed; inspect ' + str(args.out.with_suffix('.log')))
        except BaseException:
            process.terminate()
            process.wait()
            raise
    report = {'rows': total, 'random_controls': controls, 'seconds': time.monotonic() - started,
              'feature_mode': args.features,
              'dataset_sha256': source_sha(args.dataset), 'features_sha256': source_sha(args.out),
              'base_weights_sha256': hashlib.sha256(raw).hexdigest(),
              'runtime_source_sha256': source_sha(args.runtime / 'sentence-verifier.js')}
    args.out.with_suffix('.meta.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--random-controls', action='store_true')
    parser.add_argument('--features', choices=['morphology', 'context'], default='morphology')
    mine(parser.parse_args())
