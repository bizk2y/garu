"""Small research-only character-boundary model. Offsets count Unicode code points."""
import argparse
import base64
import gzip
import hashlib
import json
import re
import time
from pathlib import Path

import brotli
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit

SEED = 20261020
FEATURES = [(-1, 1), (0, 1), (-2, 2), (-1, 2), (0, 2),
            (-3, 3), (-2, 3), (-1, 3), (0, 3),
            (-3, 4), (-2, 4), (-1, 4), (0, 4)]
TERMINAL = re.compile(r'''[.!?…]+(?=[”’"')\]]*$)''')


def canonical(text):
    return text.replace('_', ' ')


def collapsed(text):
    return ''.join(canonical(text).split())


def leak_key(text):
    # The terminal-deleted augmentation must also stay on one side of a split.
    return collapsed(TERMINAL.sub('', canonical(text).strip()))


def pack_parts(parts, strip_terminal=False):
    chars, spaces, sentences = [], [], []
    for part in parts:
        part = canonical(part).strip()
        if strip_terminal:
            part = TERMINAL.sub('', part)
        if not part:
            continue
        if chars:
            spaces.append(len(chars))
            sentences.append(len(chars))
        in_space = False
        for char in part:
            if char.isspace():
                in_space = True
                continue
            if in_space and chars:
                spaces.append(len(chars))
            chars.append(char)
            in_space = False
    text = ''.join(chars)
    return {'text': text, 'spaces': sorted(set(spaces)),
            'sentences': sentences, 'variant': 'no_terminal' if strip_terminal else 'punctuated'}


def restore(text, boundaries):
    boundaries = set(boundaries)
    return ''.join((' ' if i in boundaries else '') + char for i, char in enumerate(text))


def hash_features(rows, dimensions, features=None):
    if features is None:
        features = FEATURES
    padding = max(4, max(width for _, width in features) - 1)
    codepoints, positions, spans = [], [], []
    count = 0
    for row in rows:
        text = row['text']
        base = len(codepoints) + padding
        codepoints.extend([0] * padding)
        codepoints.extend(map(ord, text))
        codepoints.extend([0] * padding)
        size = max(0, len(text) - 1)
        positions.extend(range(base + 1, base + 1 + size))
        spans.append((count, count + size))
        count += size
    if not count:
        return sparse.csr_matrix((0, dimensions), dtype=np.float64), spans
    chars = np.array(codepoints, dtype=np.uint32)
    pos = np.array(positions, dtype=np.int64)
    columns = [np.zeros(count, dtype=np.int32)]
    for feature_id, (offset, width) in enumerate(features, 1):
        hashes = np.full(count, 2166136261 ^ feature_id, dtype=np.uint32)
        for j in range(width):
            hashes = (hashes ^ chars[pos + offset + j]) * np.uint32(16777619)
        columns.append((hashes % np.uint32(dimensions - 1) + 1).astype(np.int32))
    # Coarse types: zero/pad, Hangul, ASCII upper/lower/digit, punctuation, other.
    def kinds(values):
        out = np.full(values.shape, 7, dtype=np.uint32)
        out[values == 0] = 0
        out[(values >= 0xAC00) & (values <= 0xD7A3)] = 1
        out[(values >= 65) & (values <= 90)] = 2
        out[(values >= 97) & (values <= 122)] = 3
        out[(values >= 48) & (values <= 57)] = 4
        out[np.isin(values, [ord(c) for c in '.!?…'])] = 5
        out[np.isin(values, [ord(c) for c in '\"\'“”‘’()[]{}'])] = 6
        return out
    left, right = kinds(chars[pos - 1]), kinds(chars[pos])
    columns.append((1 + (left * 17 + right * 103 + 147401) % (dimensions - 1)).astype(np.int32))
    idx = np.stack(columns, axis=1).reshape(-1)
    ptr = np.arange(0, len(idx) + 1, len(columns), dtype=np.int64)
    matrix = sparse.csr_matrix((np.ones(len(idx)), idx, ptr), shape=(count, dimensions))
    matrix.sum_duplicates()
    return matrix, spans


def targets(rows, spans, key):
    result = np.zeros(spans[-1][1] if spans else 0, dtype=np.float64)
    for row, (start, end) in zip(rows, spans):
        for boundary in row[key]:
            assert 0 < boundary < len(row['text'])
            result[start + boundary - 1] = 1
    return result


def counts(predicted, expected):
    tp = int(np.count_nonzero(predicted & expected))
    fp = int(np.count_nonzero(predicted & ~expected))
    fn = int(np.count_nonzero(~predicted & expected))
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else 0,
            'recall': tp / (tp + fn) if tp + fn else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}


def metrics(rows, spans, scores, key, threshold, eligibility=None):
    labels = targets(rows, spans, key).astype(bool)
    predictions = scores > threshold
    mask = np.ones(len(labels), dtype=bool)
    selected = []
    if eligibility:
        mask[:] = False
    for row, (start, end) in zip(rows, spans):
        if eligibility and not row[eligibility]:
            continue
        mask[start:end] = True
        selected.append((start, end))
    result = counts(predictions[mask], labels[mask])
    result['records'] = len(selected)
    result['exact'] = sum(np.array_equal(predictions[a:b], labels[a:b]) for a, b in selected) / max(1, len(selected))
    # If pre-existing whitespace is never deleted, only false insertions alter normal input.
    result['normal_preserved'] = sum(not np.any(predictions[a:b] & ~labels[a:b]) for a, b in selected) / max(1, len(selected))
    return result


def source_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_data(args):
    root, repo = Path(args.data_root), Path(args.repo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blocked = set()
    exclusion_sources = []
    for path in [repo / 'training/gold_testset/gold_testset.jsonl',
                 repo / 'training/rerank/data/dev_texts.txt']:
        exclusion_sources.append({'path': str(path), 'sha256': source_sha(path)})
        with path.open() as stream:
            for line in stream:
                text = json.loads(line)['text'] if path.suffix == '.jsonl' else line
                blocked.add(leak_key(text))
    heldout = root / 'nikl_mp_2025/SXMP2502512312.json'
    exclusion_sources.append({'path': str(heldout), 'sha256': source_sha(heldout)})
    data = json.loads(heldout.read_text())
    for doc in data['document']:
        if not doc:
            continue
        for sent in doc.get('sentence') or []:
            blocked.add(leak_key(sent['form']))
            blocked.add(leak_key(sent.get('original_form', sent['form'])))
    del data
    outputs = {name: [] for name in ['train', 'dev', 'test']}
    split_keys = {name: set() for name in outputs}
    split_docs = {name: set() for name in outputs}
    seen = set()
    stats = {'excluded_overlap': 0, 'duplicate': 0, 'too_short_long': 0, 'sources': [], 'eligible_sentences': {}}
    rng = np.random.default_rng(SEED)
    limits = {'train': args.train_per_source, 'dev': args.eval_per_source, 'test': args.eval_per_source}
    if args.fixed_eval:
        for split in ['dev', 'test']:
            outputs[split] = [json.loads(line) for line in (Path(args.fixed_eval) / (split + '.jsonl')).open()]
            limits[split] = 0
            for row in outputs[split]:
                split_docs[split].add(row['document_id'])
                points = [0] + row['sentences'] + [len(row['text'])]
                for a, b in zip(points, points[1:]):
                    key = leak_key(row['text'][a:b])
                    split_keys[split].add(key)
                    seen.add(key)
    for path in sorted((root / 'nikl_mp_2021').glob('*.json')):
        src = 'NX' if path.name.startswith('NX') else 'SX'
        print('loading', path.name, flush=True)
        stat = {'path': str(path), 'sha256': source_sha(path), 'source': src}
        stats['sources'].append(stat)
        data = json.loads(path.read_text())
        docs = data['document']
        order = rng.permutation(len(docs))
        used = {name: 0 for name in outputs}
        for index in order:
            doc = docs[int(index)]
            doc_id = doc['id']
            h = int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) % 10
            split = 'test' if h == 0 else 'dev' if h == 1 else 'train'
            if used[split] >= limits[split]:
                continue
            block = []
            def flush():
                if not block:
                    return
                texts = [s['form'] for s in block]
                sentence_ok = src == 'NX' and all(TERMINAL.search(canonical(t).strip()) for t in texts)
                original = pack_parts(texts)
                for stripped in [False, True]:
                    row = pack_parts(texts, strip_terminal=stripped)
                    if stripped and row['text'] == original['text']:
                        continue
                    assert collapsed(restore(row['text'], row['spaces'])) == row['text']
                    row.update({'source': src, 'document_id': doc_id,
                                'sentence_ids': [s['id'] for s in block],
                                'source_file': path.name, 'sentence_eligible': bool(sentence_ok),
                                'part_count': len(block),
                                'assembly': 'consecutive_source_sentences_with_one_space'})
                    outputs[split].append(row)
                block.clear()
            for sent in doc['sentence']:
                key = leak_key(sent['form'])
                if len(key) < 5 or len(key) > 180:
                    stats['too_short_long'] += 1
                    flush()
                    continue
                if key in blocked:
                    stats['excluded_overlap'] += 1
                    flush()
                    continue
                if key in seen:
                    stats['duplicate'] += 1
                    flush()
                    continue
                if used[split] >= limits[split]:
                    flush()
                    break
                seen.add(key)
                split_keys[split].add(key)
                split_docs[split].add(doc_id)
                block.append(sent)
                used[split] += 1
                if len(block) == 3 or sum(len(s['form']) for s in block) >= 180:
                    flush()
            flush()
        stats['eligible_sentences'][src] = used
        del docs, data
        print(src, used, flush=True)
    for a in outputs:
        assert not (split_keys[a] & blocked)
        for b in outputs:
            if a != b:
                assert not (split_keys[a] & split_keys[b])
                assert not (split_docs[a] & split_docs[b])
    augmented_keys = {}
    for split, records in outputs.items():
        keys = set()
        for row in records:
            offsets = [0] + row['sentences'] + [len(row['text'])]
            keys.update(leak_key(row['text'][a:b]) for a, b in zip(offsets, offsets[1:]))
        assert not (keys & blocked)
        augmented_keys[split] = keys
    for a in outputs:
        for b in outputs:
            if a != b:
                assert not (augmented_keys[a] & augmented_keys[b])
    for name, rows in outputs.items():
        with (out / (name + '.jsonl')).open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {'seed': SEED, 'normalization': 'underscore to space; whitespace removed with exact codepoint boundary labels',
                'split': 'sha256(document_id) first uint32 modulo 10; test0 dev1 train2..9',
                'sentence_label_scope': 'NX consecutive source records ending in terminal punctuation; SX sentence labels excluded',
                'sentence_caveat': '2021 has no original paragraphs here; assembly whitespace is synthetic, adjacency comes from original document order',
                'exclusion_key': 'underscore/whitespace normalized and terminal punctuation removed; applies before augmentation',
                'fixed_eval': args.fixed_eval,
                'excluded_keys': len(blocked), 'exclusion_sources': exclusion_sources,
                'rows': {k: len(v) for k, v in outputs.items()},
                'documents': {k: len(v) for k, v in split_docs.items()}, **stats}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def train(args):
    folder = Path(args.out)
    rows = {split: [json.loads(line) for line in (folder / (split + '.jsonl')).open()]
            for split in ['train', 'dev', 'test']}
    dimensions = 1 << args.bits
    features = FEATURES + ([(-4, 4)] if args.symmetric else [])
    for width in range(5, args.max_width + 1):
        features.extend((offset, width) for offset in range(-width, 1))
    matrices, spans = {}, {}
    for split in rows:
        matrices[split], spans[split] = hash_features(rows[split], dimensions, features)
        print('matrix', split, matrices[split].shape, matrices[split].nnz, flush=True)
    models, report = [], {'dimensions': dimensions, 'features': features, 'l2': args.l2,
                         'speech_weight': args.speech_weight,
                         'sentence_threshold_objective': args.sentence_threshold, 'heads': {}}
    for key in ['spaces', 'sentences']:
        matrix = matrices['train']
        labels = targets(rows['train'], spans['train'], key)
        sample_weights = np.ones(len(labels))
        if key == 'sentences':
            mask = np.zeros(len(labels), dtype=bool)
            for row, (a, b) in zip(rows['train'], spans['train']):
                if row['sentence_eligible']:
                    mask[a:b] = True
                    if row['source'] == 'SX':
                        sample_weights[a:b] = args.speech_weight
            matrix, labels = matrix[mask], labels[mask]
            sample_weights = sample_weights[mask]
        n = len(labels)
        weight_sum = sample_weights.sum()
        initial = np.zeros(dimensions)
        initial[0] = np.log((labels.sum() + 1) / (n - labels.sum() + 1))
        iteration = [0]
        started = time.monotonic()
        def objective(weights):
            logits = matrix @ weights
            loss = np.dot(sample_weights, np.logaddexp(0, logits) - labels * logits) / weight_sum + args.l2 * np.dot(weights[1:], weights[1:]) / 2
            gradient = matrix.T @ ((expit(logits) - labels) * sample_weights) / weight_sum
            gradient[1:] += args.l2 * weights[1:]
            return loss, gradient
        def progress(weights):
            iteration[0] += 1
            if iteration[0] % 10 == 0:
                print('fit', key, iteration[0], 'seconds', round(time.monotonic() - started, 1), flush=True)
        result = minimize(objective, initial, method='L-BFGS-B', jac=True,
                          callback=progress, options={'maxiter': args.iterations, 'ftol': 1e-8, 'gtol': 1e-6, 'maxcor': 10})
        weights = result.x
        # Symmetric int8 per-head quantization; validate the actual delivered weights.
        scale = float(np.max(np.abs(weights)) / 127)
        quantized = np.clip(np.round(weights / scale), -127, 127).astype(np.int8)
        models.append(quantized)
        scores = {split: (matrices[split] @ quantized.astype(np.float64)) * scale for split in rows}
        eligibility = 'sentence_eligible' if key == 'sentences' else None
        thresholds = np.arange(-4.0, 4.01, 0.25)
        trials = [(metrics(rows['dev'], spans['dev'], scores['dev'], key, float(t), eligibility)['f1'], float(t)) for t in thresholds]
        if key == 'sentences' and args.sentence_threshold == 'macro':
            trials = []
            for t in thresholds:
                f1s = []
                for source in ['NX', 'SX']:
                    for row in rows['dev']:
                        row['_source_eligible'] = row['source'] == source and row['sentence_eligible']
                    f1s.append(metrics(rows['dev'], spans['dev'], scores['dev'], key, float(t), '_source_eligible')['f1'])
                trials.append((sum(f1s) / len(f1s), float(t)))
        threshold = max(trials)[1]
        head = {'scale': scale, 'threshold': threshold, 'iterations': result.nit,
                'converged': bool(result.success), 'optimizer_status': result.message,
                'seconds': time.monotonic() - started, 'training_boundaries': n,
                'training_positive': int(labels.sum()), 'metrics': {}, 'baseline_no_boundaries': {}}
        for split in ['dev', 'test']:
            head['metrics'][split] = metrics(rows[split], spans[split], scores[split], key, threshold, eligibility)
            head['baseline_no_boundaries'][split] = metrics(rows[split], spans[split], np.full_like(scores[split], -100), key, 0, eligibility)
            for source in ['NX', 'SX']:
                for row in rows[split]:
                    row['_source_eligible'] = row['source'] == source and (not eligibility or row[eligibility])
                head['metrics'][split + '_' + source] = metrics(rows[split], spans[split], scores[split], key, threshold, '_source_eligible')
            for variant in ['punctuated', 'no_terminal']:
                selected = [i for i, row in enumerate(rows[split]) if row['variant'] == variant and (not eligibility or row[eligibility])]
                mask = np.zeros(len(scores[split]), dtype=bool)
                for i in selected:
                    a, b = spans[split][i]
                    mask[a:b] = True
                truth = targets(rows[split], spans[split], key).astype(bool)
                part = counts((scores[split] > threshold)[mask], truth[mask])
                part['records'] = len(selected)
                part['exact'] = sum(np.array_equal((scores[split][spans[split][i][0]:spans[split][i][1]] > threshold), truth[spans[split][i][0]:spans[split][i][1]]) for i in selected) / max(1, len(selected))
                head['metrics'][split + '_' + variant] = part
        report['heads'][key] = head
        print('RESULT', key, json.dumps(head), flush=True)
        np.savez(folder / (key + '.npz'), weights=quantized, scale=scale, threshold=threshold)
    raw = b''.join(w.tobytes() for w in models)
    compressed = brotli.compress(raw, quality=11)
    (folder / 'weights.int8.br').write_bytes(compressed)
    report['weights_raw_bytes'] = len(raw)
    report['weights_brotli_bytes'] = len(compressed)
    report['weights_gzip_bytes'] = len(gzip.compress(raw, compresslevel=9, mtime=0))
    export = {'dimensions': dimensions, 'features': features,
              'padding_codepoint': 0, 'hash': 'FNV1a over Unicode codepoints, initial=2166136261 xor featureId(1-based), multiplier=16777619 modulo uint32, bucket=1+h%(dimensions-1)',
              'bias_index': 0,
              'category_feature': 'bucket=1+(leftKind*17+rightKind*103+147401)%(dimensions-1)',
              'categories': {'pad': 0, 'hangul_syllable': 1, 'ascii_upper': 2, 'ascii_lower': 3,
                             'ascii_digit': 4, 'terminal_dot_question_exclamation_ellipsis': 5,
                             'quote_bracket': 6, 'other': 7},
              'category_quotes': '\"\'“”‘’()[]{}',
              'heads': {key: {'scale': value['scale'], 'threshold': value['threshold'],
                              'order': i} for i, (key, value) in enumerate(report['heads'].items())},
              'weights_encoding': 'base64 of signed int8, spaces then sentences',
              'weights_sha256': hashlib.sha256(raw).hexdigest()}
    encoded = base64.b64encode(raw).decode()
    (folder / 'weights.b64').write_text(encoded)
    (folder / 'export.json').write_text(json.dumps(export, ensure_ascii=False, indent=2))
    report['base64_gzip_bytes'] = len(gzip.compress(encoded.encode(), compresslevel=9, mtime=0))
    (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('FINAL', json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'train'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--repo', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--data-root', default=str(Path.home() / 'workspace/data'))
    parser.add_argument('--train-per-source', type=int, default=15000)
    parser.add_argument('--eval-per-source', type=int, default=1500)
    parser.add_argument('--bits', type=int, default=17)
    parser.add_argument('--iterations', type=int, default=60)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--fixed-eval')
    parser.add_argument('--symmetric', action='store_true')
    parser.add_argument('--max-width', type=int, default=4)
    parser.add_argument('--speech-weight', type=float, default=1.0)
    parser.add_argument('--sentence-threshold', choices=['micro', 'macro'], default='micro')
    args = parser.parse_args()
    make_data(args) if args.mode == 'prepare' else train(args)


if __name__ == '__main__':
    main()
