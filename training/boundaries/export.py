"""Select compatible heads, emit TypeScript, and preserve references without source text."""
import argparse
import base64
import copy
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import brotli
import numpy as np

CONTRACT_KEYS = ['dimensions', 'features', 'padding_codepoint', 'hash', 'bias_index',
                 'category_feature', 'categories', 'category_quotes', 'weights_encoding']


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def write_json(path, value):
    path.write_bytes(json_bytes(value))


def file_ref(path):
    return {'path': str(path.resolve()), 'sha256': sha_file(path), 'bytes': path.stat().st_size}


def select_heads(spaces, sentences, out):
    inputs = {'spaces': spaces, 'sentences': sentences}
    exports = {head: read_json(folder / 'export.json') for head, folder in inputs.items()}
    for key in CONTRACT_KEYS:
        if exports['spaces'][key] != exports['sentences'][key]:
            raise ValueError('Incompatible model contract: ' + key)
    dimensions = exports['spaces']['dimensions']
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Selection output must be empty; preserve any later threshold edits')
    selected, heads, lineage = [], {}, {}
    for order, (head, folder) in enumerate(inputs.items()):
        exported = exports[head]
        source_raw = base64.b64decode((folder / 'weights.b64').read_text(), validate=True)
        assert sha_bytes(source_raw) == exported['weights_sha256']
        data = np.load(folder / (head + '.npz'))
        weights = data['weights']
        assert weights.dtype == np.int8 and weights.shape == (dimensions,)
        start = exported['heads'][head]['order'] * dimensions
        assert source_raw[start:start + dimensions] == weights.tobytes()
        assert float(data['scale']) == exported['heads'][head]['scale']
        assert float(data['threshold']) == exported['heads'][head]['threshold']
        selected.append(weights.tobytes())
        source_report = read_json(folder / 'report.json')
        heads[head] = copy.deepcopy(source_report['heads'][head])
        heads[head]['order'] = order
        shutil.copyfile(folder / (head + '.npz'), out / (head + '.npz'))
        data_manifest = read_json(folder / 'manifest.json')
        write_json(out / (head + '-source-manifest.json'), data_manifest)
        lineage[head] = {
            'source_model': folder.name,
            'source_model_weights_sha256': exported['weights_sha256'],
            'selected_head_weights_sha256': sha_bytes(weights.tobytes()),
            'selected_head_npz': file_ref(folder / (head + '.npz')),
            'source_export': file_ref(folder / 'export.json'),
            'source_report': file_ref(folder / 'report.json'),
            'source_data_manifest': file_ref(folder / 'manifest.json'),
            'preserved_data_manifest_sha256': sha_file(out / (head + '-source-manifest.json')),
            'datasets': {split: file_ref(folder / (split + '.jsonl')) for split in ['train', 'dev', 'test']},
            'data_lineage': data_manifest,
            'training': {key: source_report.get(key) for key in ['l2', 'speech_weight', 'sentence_threshold_objective']},
            'balanced_threshold': heads[head]['threshold'],
        }
    for split in ['dev', 'test']:
        assert lineage['spaces']['datasets'][split]['sha256'] == lineage['sentences']['datasets'][split]['sha256']
    raw = b''.join(selected)
    encoded = base64.b64encode(raw).decode()
    source_manifest = {
        'schema': 1,
        'selection': 'Spaces full19 and sentences large19, selected per head on development metrics within the shared 2^19/13-feature format.',
        'selection_limit': 'This is not a claim that the sentence head is globally best across incompatible dimensions and feature sets.',
        'threshold_policy': 'Original balanced development thresholds retained; no conservative wrapper threshold chosen here.',
        'source_text_redistributed': False,
        'selected_weights_sha256': sha_bytes(raw),
        'export_script_sha256': sha_file(Path(__file__)),
        'heads': lineage,
    }
    write_json(out / 'source-manifest.json', source_manifest)
    source_sha = sha_file(out / 'source-manifest.json')
    exported = {key: exports['spaces'][key] for key in CONTRACT_KEYS}
    # Preserve optional wrapper policy fields, such as conservative_threshold,
    # while each report/NPZ threshold remains the balanced training threshold.
    runtime_heads = {}
    for order, head in enumerate(heads):
        runtime_heads[head] = copy.deepcopy(exports[head]['heads'][head])
        runtime_heads[head]['order'] = order
    exported.update({'heads': runtime_heads,
                     'weights_sha256': sha_bytes(raw), 'source_manifest_sha256': source_sha,
                     'threshold_policy': 'balanced development thresholds'})
    report = {'dimensions': dimensions, 'features': exported['features'], 'heads': heads,
              'selection': source_manifest['selection'], 'source_manifest_sha256': source_sha,
              'weights_sha256': sha_bytes(raw), 'weights_raw_bytes': len(raw),
              'weights_brotli_bytes': len(brotli.compress(raw, quality=11)),
              'weights_gzip_bytes': len(gzip.compress(raw, compresslevel=9, mtime=0)),
              'base64_gzip_bytes': len(gzip.compress(encoded.encode(), compresslevel=9, mtime=0)),
              'metrics_scope': 'Per-head inherited measurements before JavaScript wrapper protection gates; not a fresh hybrid end-to-end evaluation.'}
    (out / 'weights.b64').write_text(encoded)
    (out / 'weights.int8.br').write_bytes(brotli.compress(raw, quality=11))
    write_json(out / 'export.json', exported)
    write_json(out / 'report.json', report)
    return report


def emit_typescript(model, out):
    metadata = read_json(model / 'export.json')
    encoded = (model / 'weights.b64').read_text()
    assert sha_bytes(base64.b64decode(encoded, validate=True)) == metadata['weights_sha256']
    runtime = {key: metadata[key] for key in ['dimensions', 'features', 'heads']}
    text = '// Generated by training/boundaries/export.py; provenance stays outside runtime code.\n'
    text += 'export const BOUNDARY_MODEL = ' + json.dumps(runtime, ensure_ascii=False, separators=(',', ':')) + ' as const;\n'
    # An explicit type prevents TypeScript from copying the full base64 literal into .d.ts.
    text += 'export const BOUNDARY_WEIGHTS: string = ' + json.dumps(encoded) + ';\n'
    text += '\n/** Unrolled FNV scoring for this model; no runtime code generation. */\n'
    text += ('export function scoreBoundary(\n'
             '  points: readonly number[], i: number, values: Int8Array, offset: number,\n'
             '): number {\n  let score = values[offset];\n')
    for index, (start, width) in enumerate(metadata['features'], 1):
        hashed = str(2166136261 ^ index)
        for delta in range(width):
            hashed = f'Math.imul({hashed} ^ (points[i + {start + delta}] ?? 0), 16777619)'
        text += f"  score += values[offset + 1 + (({hashed}) >>> 0) % {metadata['dimensions'] - 1}];\n"
    text += '  return score;\n}\n'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)


def preserve_refs(model, out):
    source = read_json(model / 'source-manifest.json')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Reference archive output must be empty')
    datasets, by_hash = {}, {}
    for head, details in source['heads'].items():
        datasets[head] = {}
        for split, info in details['datasets'].items():
            if info['sha256'] in by_hash:
                datasets[head][split] = by_hash[info['sha256']]
                continue
            origin = Path(info['path'])
            assert sha_file(origin) == info['sha256']
            filename = head + '-' + split + '.refs.jsonl.gz'
            count = 0
            with (out / filename).open('wb') as raw_stream:
                with gzip.GzipFile(filename='', mode='wb', fileobj=raw_stream, mtime=0, compresslevel=9) as compressed:
                    for line in origin.open():
                        row = json.loads(line)
                        reference = {key: row[key] for key in ['source_file', 'source', 'document_id', 'sentence_ids', 'variant', 'sentence_eligible', 'part_count', 'assembly']}
                        reference['text_codepoints'] = len(row['text'])
                        reference['normalized_text_sha256'] = sha_bytes(row['text'].encode())
                        reference['space_boundaries_sha256'] = sha_bytes(json.dumps(row['spaces'], separators=(',', ':')).encode())
                        reference['sentence_boundaries_sha256'] = sha_bytes(json.dumps(row['sentences'], separators=(',', ':')).encode())
                        compressed.write((json.dumps(reference, ensure_ascii=False, separators=(',', ':')) + '\n').encode())
                        count += 1
            item = {'file': filename, 'rows': count, 'sha256': sha_file(out / filename),
                    'source_dataset_sha256': info['sha256'], 'source_text_included': False}
            by_hash[info['sha256']] = item
            datasets[head][split] = item
    for filename in ['source-manifest.json', 'spaces-source-manifest.json', 'sentences-source-manifest.json', 'export.json']:
        shutil.copyfile(model / filename, out / filename)
    manifest = {'schema': 1, 'purpose': 'Reconstruct and verify training examples from retained licensed original corpora; contains IDs and hashes, no source text or model weights.',
                'selected_source_manifest_sha256': sha_file(model / 'source-manifest.json'),
                'selected_weights_sha256': source['selected_weights_sha256'],
                'datasets': datasets}
    write_json(out / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    selection = commands.add_parser('select')
    selection.add_argument('--spaces', type=Path, required=True)
    selection.add_argument('--sentences', type=Path, required=True)
    selection.add_argument('--out', type=Path, required=True)
    for command in ['typescript', 'refs']:
        sub = commands.add_parser(command)
        sub.add_argument('--model', type=Path, required=True)
        sub.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'select':
        result = select_heads(args.spaces, args.sentences, args.out)
        print(json.dumps({key: result[key] for key in ['weights_sha256', 'weights_brotli_bytes', 'base64_gzip_bytes', 'source_manifest_sha256']}))
    elif args.command == 'typescript':
        emit_typescript(args.model, args.out)
        print(json.dumps({'typescript': str(args.out), 'bytes': args.out.stat().st_size}))
    else:
        result = preserve_refs(args.model, args.out)
        print(json.dumps({'archive': str(args.out), 'datasets': result['datasets']}))


if __name__ == '__main__':
    main()
