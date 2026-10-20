"""Train and export the optional context verifier; no changes to base morphology."""
import base64
import argparse
import hashlib
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from boundary import source_sha

FEATURE_NAMES = ['baseScore', 'leftTerminal', 'leftComma', 'insideQuote', 'bracketDepth',
                 'tokenBoundary', 'tokenCrossing', 'morphUnreliable', 'leftEF', 'leftEC',
                 'leftETM_ETN', 'rightVerb', 'rightParticle', 'rightNoun', 'leftReporting', 'rightReporting']
CONTEXT = 24
EMBEDDING_SIZE = 24
HIDDEN_SIZE = 96
VOCABULARY_LIMIT = 6000
BATCH_SIZE = 1024
MODEL_LIMIT_BYTES = 3_000_000
MORPH_FEATURES = slice(5, 14)


def embed_context(embedding, ids):
    # PyTorch2.8 MPS locally backpropagates into padding_idx; mask its gradient path.
    return embedding(ids) * ids.ne(0).unsqueeze(-1)


def context_ids(text, boundary, vocabulary, context):
    return [vocabulary.get(text[i], 1) if 0 <= i < len(text) else 0
            for i in range(boundary - context, boundary + context)]


def pack_q4(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Q4 tensor must be a finite matrix')
    scales = np.max(np.abs(values), axis=1) / 7
    scales[scales == 0] = 1
    codes = (np.clip(np.rint(values / scales[:, None]), -7, 7) + 8).astype(np.uint8).reshape(-1)
    if len(codes) % 2:
        codes = np.append(codes, np.uint8(8))
    packed = codes[::2] | (codes[1::2] << 4)
    return {'rows': values.shape[0], 'cols': values.shape[1],
            'data': base64.b64encode(packed.tobytes()).decode(),
            'scales': base64.b64encode(scales.astype('<f4').tobytes()).decode()}


def unpack_q4(tensor):
    packed = np.frombuffer(base64.b64decode(tensor['data'], validate=True), dtype=np.uint8)
    codes = np.stack([packed & 15, packed >> 4], axis=1).reshape(-1)
    size = tensor['rows'] * tensor['cols']
    values = codes[:size].astype(np.float32).reshape(tensor['rows'], tensor['cols']) - 8
    scales = np.frombuffer(base64.b64decode(tensor['scales'], validate=True), dtype='<f4')
    return values * scales[:, None]


def float_bytes(values):
    return base64.b64encode(np.asarray(values, dtype='<f4').tobytes()).decode()


def read_bias(tensor):
    return np.frombuffer(base64.b64decode(tensor['bias']), dtype='<f4')


def export_model(model, vocabulary, mode, source_hash):
    embedding = model.embedding.weight.detach().cpu().numpy().copy()
    if np.any(embedding[0] != 0):
        raise ValueError('Padding embedding changed during training')
    hidden = model.hidden.weight.detach().cpu().numpy().copy()
    embedding_size = embedding.shape[1]
    context_size = hidden.shape[1] - len(FEATURE_NAMES)
    context = context_size // (2 * embedding_size)
    if context <= 0 or context * 2 * embedding_size != context_size:
        raise ValueError('Invalid context dimensions')
    if mode == 'structured':
        hidden[:, :context_size] = 0
    if mode == 'no_morph':
        hidden[:, context_size + MORPH_FEATURES.start:context_size + MORPH_FEATURES.stop] = 0
    tensors = {'embedding': pack_q4(embedding), 'hidden': pack_q4(hidden),
               'output': pack_q4(model.output.weight.detach().cpu().numpy())}
    for name in ['hidden', 'output']:
        tensors[name]['bias'] = float_bytes(getattr(model, name).bias.detach().cpu().numpy())
    return {'version': 1, 'context': context, 'embeddingSize': embedding_size,
            'featureNames': FEATURE_NAMES, 'vocabulary': vocabulary, 'threshold': 0.0,
            'sourceHash': source_hash, 'tensors': tensors}


def score_export(model, ids, features):
    embedding = unpack_q4(model['tensors']['embedding'])
    hidden = unpack_q4(model['tensors']['hidden'])
    output = unpack_q4(model['tensors']['output'])
    bias_hidden = read_bias(model['tensors']['hidden'])
    bias_output = read_bias(model['tensors']['output'])
    scores = []
    for start in range(0, len(ids), BATCH_SIZE):
        inputs = np.concatenate([embedding[ids[start:start + BATCH_SIZE]].reshape(-1, hidden.shape[1] - len(FEATURE_NAMES)),
                                 features[start:start + BATCH_SIZE]], axis=1)
        # Local BLAS raises floating-status warnings for bounded, finite products.
        # Validate computed arrays directly rather than silently accepting NaN vetoes.
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            activations = np.maximum(inputs @ hidden.T + bias_hidden, 0).astype(np.float32)
            logits = (activations @ output.T + bias_output).reshape(-1)
        if not np.isfinite(activations).all() or not np.isfinite(logits).all():
            raise ValueError('Non-finite quantized verifier inference')
        scores.extend(logits.tolist())
    return np.asarray(scores, dtype=np.float32)


def build_arrays(folder, split, vocabulary, context=CONTEXT):
    metadata = json.loads((folder / (split + '-features.meta.json')).read_text())
    if (metadata['dataset_sha256'] != source_sha(folder / 'data' / (split + '.jsonl'))
            or metadata['features_sha256'] != source_sha(folder / (split + '-features.jsonl'))):
        raise ValueError('Feature cache does not match the source dataset')
    ids, features, labels, groups, kind = [], [], [], [], []
    totals, seen, unmatched = Counter(), 0, 0
    with (folder / 'data' / (split + '.jsonl')).open() as source, (folder / (split + '-features.jsonl')).open() as mined:
        for source_line, feature_line in itertools.zip_longest(source, mined):
            if source_line is None or feature_line is None:
                raise ValueError('Feature row count differs from source')
            row, extraction = json.loads(source_line), json.loads(feature_line)
            assert row['id'] == extraction['id']
            group = row['source'] + '_' + row['variant']
            gold = set(row['sentences'])
            totals[group] += len(gold)
            if len(extraction.get('extraCandidates', [])) != len(extraction.get('extraFeatures', [])):
                raise ValueError('Random-control feature count mismatch')
            hard = sum(candidate['compactOffset'] not in gold for candidate in extraction['candidates'])
            extra = list(zip(extraction.get('extraCandidates', []), extraction.get('extraFeatures', [])))
            limit = min(hard, len(extra)) if split == 'train' else hard
            unmatched += hard - limit
            negative_count = 0
            assert len(extraction['candidates']) == len(extraction['features'])
            records = [(candidate, feat, 0) for candidate, feat in zip(extraction['candidates'], extraction['features'])]
            records += [(candidate, feat, 1) for candidate, feat in extra[:limit]]
            for candidate, feat, control in records:
                boundary = candidate['compactOffset']
                label = int(boundary in gold)
                if control and label:
                    raise ValueError('Random control overlaps a true boundary')
                if split == 'train' and not control and not label:
                    negative_count += 1
                    if negative_count > limit:
                        continue
                assert len(feat) == len(FEATURE_NAMES) and np.isfinite(feat).all()
                ids.append(context_ids(row['text'], boundary, vocabulary, context))
                features.append(feat)
                labels.append(label)
                groups.append(group)
                kind.append(control)
            seen += 1
    return {'ids': np.asarray(ids, dtype=np.int64), 'features': np.asarray(features, dtype=np.float32),
            'labels': np.asarray(labels, dtype=np.float32), 'groups': np.asarray(groups),
            'kind': np.asarray(kind), 'gold_totals': dict(totals), 'rows': seen,
            'unmatched_hard_negatives': unmatched}


def score_metrics(data, scores, threshold):
    if scores.shape != data['labels'].shape or not np.isfinite(scores).all():
        raise ValueError('Invalid verifier scores')
    report = {}
    prediction = scores > threshold
    truth = data['labels'] > 0
    for group, gold in data['gold_totals'].items():
        selected = data['groups'] == group
        tp = int(np.count_nonzero(prediction & truth & selected))
        fp = int(np.count_nonzero(prediction & ~truth & selected))
        fn = gold - tp
        report[group] = {'tp': tp, 'fp': fp, 'fn': fn,
                         'precision': tp / max(1, tp + fp), 'recall': tp / max(1, gold),
                         'f1': 2 * tp / max(1, 2 * tp + fp + fn)}
    report['macro_f1'] = sum(item['f1'] for item in report.values()) / max(1, len(report))
    return report


def choose_threshold(data, scores):
    trials = [(score_metrics(data, scores, round(float(threshold), 2))['macro_f1'], round(float(threshold), 2))
              for threshold in np.arange(-8, 8.01, 0.1)]
    return max(trials)


def run_training(args):
    import torch
    from torch import nn
    torch.set_num_threads(4)
    vocabulary_counts = Counter()
    for line in (args.lab / 'data/train.jsonl').open():
        vocabulary_counts.update(json.loads(line)['text'])
    vocabulary = sorted([c for c, count in vocabulary_counts.most_common(VOCABULARY_LIMIT) if count >= 2])
    vocabulary_ids = {c: i + 2 for i, c in enumerate(vocabulary)}
    arrays = {split: build_arrays(args.lab, split, vocabulary_ids, args.context) for split in ['train', 'dev', 'test']}
    source_hash = json.loads((args.lab / 'train-features.meta.json').read_text())['base_weights_sha256']
    feature_log = (args.lab / 'train-features.log').read_text()
    assert any(json.loads(line).get('featureNames') == FEATURE_NAMES for line in feature_log.splitlines() if line.startswith('{'))
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print('arrays', {split: {'examples': len(data['labels']), 'rows': data['rows'], 'gold': data['gold_totals'],
                           'unmatched_hard': data['unmatched_hard_negatives']} for split, data in arrays.items()}, flush=True)
    class Model(nn.Module):
        def __init__(self, mode):
            super().__init__()
            self.mode = mode
            self.embedding = nn.Embedding(len(vocabulary) + 2, args.embedding, padding_idx=0)
            nn.init.normal_(self.embedding.weight, std=0.1)
            with torch.no_grad():
                self.embedding.weight[0] = 0
            self.hidden = nn.Linear(args.context * 2 * args.embedding + len(FEATURE_NAMES), args.hidden)
            self.output = nn.Linear(args.hidden, 1)
            self.dropout = nn.Dropout(0.1)

        def forward(self, ids, features):
            chars = embed_context(self.embedding, ids).flatten(1)
            if self.mode == 'structured':
                chars = chars * 0
            if self.mode == 'no_morph':
                features = features.clone()
                features[:, MORPH_FEATURES] = 0
            value = torch.cat([chars, features], dim=1)
            return self.output(self.dropout(torch.relu(self.hidden(value)))).squeeze(-1)

    output = args.out or args.lab / 'models'
    output.mkdir(exist_ok=True)
    report = {'seed': args.seed, 'device': str(device), 'context': args.context,
              'embedding': args.embedding, 'hidden': args.hidden, 'vocabulary': len(vocabulary),
              'architecture': f'Ordered character embeddings + structured features -> ReLU{args.hidden} -> scalar logit',
              'threshold_objective': 'Development macro F1 across NX/SX and punctuation variants, includes missing gold outside frozen candidate set',
              'source_weights_sha256': source_hash, 'datasets': json.loads((args.lab / 'data/manifest.json').read_text()),
              'baseline': {split: score_metrics(data, np.ones(len(data['labels'])), 0) for split, data in arrays.items() if split != 'train'},
              'models': {}}
    for mode in args.modes:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = Model(mode).to(device)
        train = arrays['train']
        selector = ((train['kind'] == 0) & (train['labels'] == 1)) | (train['kind'] == (1 if mode == 'easy' else 0))
        inputs = torch.from_numpy(train['ids'][selector]).to(device)
        features = torch.from_numpy(train['features'][selector]).to(device)
        labels = torch.from_numpy(train['labels'][selector]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        best, stale, history = -1.0, 0, []
        started = time.monotonic()
        for epoch in range(args.epochs):
            model.train()
            indices = np.random.default_rng(args.seed + epoch).permutation(len(labels))
            total_loss = 0.0
            for start in range(0, len(indices), BATCH_SIZE):
                batch = torch.from_numpy(indices[start:start + BATCH_SIZE]).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs[batch], features[batch])
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels[batch])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(batch)
            exported = export_model(model, vocabulary, mode, source_hash)
            scores = score_export(exported, arrays['dev']['ids'], arrays['dev']['features'])
            metric, threshold = choose_threshold(arrays['dev'], scores)
            event = {'epoch': epoch + 1, 'loss': total_loss / len(labels), 'dev_macro_f1': metric,
                     'threshold': threshold, 'seconds': time.monotonic() - started}
            history.append(event)
            print(mode, json.dumps(event), flush=True)
            if metric > best:
                best, stale = metric, 0
                exported['threshold'] = threshold
                blob = (json.dumps(exported, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
                assert len(blob) <= MODEL_LIMIT_BYTES
                (output / (mode + '.json')).write_bytes(blob)
                best_epoch = epoch + 1
            else:
                stale += 1
            if stale >= 5:
                break
        exported = json.loads((output / (mode + '.json')).read_text())
        measurements = {split: score_metrics(arrays[split], score_export(exported, arrays[split]['ids'], arrays[split]['features']), exported['threshold']) for split in ['dev', 'test']}
        report['models'][mode] = {'best_epoch': best_epoch, 'history': history, 'metrics': measurements,
                                 'parameters': sum(p.numel() for p in model.parameters()),
                                 'training_examples': int(selector.sum()), 'training_positive': int(train['labels'][selector].sum()),
                                 'complete_q4_bytes': (output / (mode + '.json')).stat().st_size,
                                 'sha256': hashlib.sha256((output / (mode + '.json')).read_bytes()).hexdigest()}
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print('RESULT', mode, json.dumps(report['models'][mode]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lab', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--context', type=int, default=CONTEXT)
    parser.add_argument('--embedding', type=int, default=EMBEDDING_SIZE)
    parser.add_argument('--hidden', type=int, default=HIDDEN_SIZE)
    parser.add_argument('--modes', nargs='+', choices=['full', 'structured', 'no_morph', 'easy'], default=['structured', 'full', 'no_morph', 'easy'])
    run_training(parser.parse_args())
