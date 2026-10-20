"""Evaluate frozen public JS boundary APIs, including their protection gates."""
import argparse
import json
import subprocess
import time
from pathlib import Path

from boundary import restore
from eval_natural import load_natural, score_natural
from export import sha_file
from infer import JS_WHITESPACE, compact_input

NODE_RUNNER = r'''
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const api=await import(pathToFileURL(input.entry).href);
const data=await import(pathToFileURL(path.join(path.dirname(input.entry),'boundary-data.js')).href);
const start=performance.now();
const sentences=input.sentences.map(text=>api.splitUnspaced(text));
const spacing={};
for(const mode of ['conservative','balanced']){
  spacing[mode]=input.spacing.map(text=>mode==='conservative'
    ?api.restoreSpacing(text):api.restoreSpacing(text,{mode}));
}
process.stdout.write(JSON.stringify({sentences,spacing,model:data.BOUNDARY_MODEL,
  weights_sha256:createHash('sha256').update(Buffer.from(data.BOUNDARY_WEIGHTS,'base64')).digest('hex'),
  node_version:process.version,inference_seconds:(performance.now()-start)/1000}));
'''


def space_positions(text):
    positions, index, pending = set(), 0, False
    for char in text:
        if char in JS_WHITESPACE:
            pending = True
            continue
        if pending and index:
            positions.add(index)
        pending = False
        index += 1
    return positions


def subsequence(source, output):
    cursor = 0
    for char in output:
        if cursor < len(source) and char == source[cursor]:
            cursor += 1
    return cursor == len(source)


def score_spaces(rows, inputs, outputs):
    assert len(rows) == len(inputs) == len(outputs)
    tp = fp = fn = exact = unchanged = chars_ok = whitespace_ok = 0
    for row, original, output in zip(rows, inputs, outputs):
        expected = set(row['spaces'])
        predicted = space_positions(output)
        tp += len(expected & predicted)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
        exact += output == restore(row['text'], row['spaces'])
        unchanged += output == original
        chars_ok += compact_input(output) == compact_input(original)
        whitespace_ok += subsequence(original, output)
    size = len(rows)
    return {'rows': size, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else 0,
            'recall': tp / (tp + fn) if tp + fn else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            'exact': exact / size if size else 0,
            'input_unchanged_rate': unchanged / size if size else 0,
            'non_whitespace_unchanged_count': chars_ok,
            'existing_characters_subsequence_count': whitespace_ok}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--javascript', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path.home() / 'workspace/data')
    args = parser.parse_args()
    started = time.monotonic()
    lineage = json.loads((args.model / 'source-manifest.json').read_text())
    training = []
    for head in ['spaces', 'sentences']:
        info = lineage['heads'][head]['datasets']['train']
        path = Path(info['path'])
        assert sha_file(path) == info['sha256']
        training.append(path)
    source = args.data_root / 'nikl_mp_2025/NXMP2502512313.json'
    natural, natural_info = load_natural(training, source)
    tasks, flat_inputs, slices, dataset_refs = {}, [], {}, {}
    for split in ['dev', 'test']:
        info = lineage['heads']['spaces']['datasets'][split]
        path = Path(info['path'])
        assert sha_file(path) == info['sha256']
        dataset_refs[split] = {'sha256': info['sha256'], 'path': str(path)}
        rows = [json.loads(line) for line in path.open()]
        for condition in ['no_spaces', 'half_spaces_removed', 'normal_spaces']:
            if condition == 'no_spaces':
                inputs = [row['text'] for row in rows]
            elif condition == 'half_spaces_removed':
                inputs = [restore(row['text'], row['spaces'][::2]) for row in rows]
            else:
                inputs = [restore(row['text'], row['spaces']) for row in rows]
            key = split + '_' + condition
            slices[key] = (len(flat_inputs), len(flat_inputs) + len(inputs))
            tasks[key] = (rows, inputs)
            flat_inputs.extend(inputs)
    request = {'entry': str(args.javascript.resolve()), 'sentences': [r['text'] for r in natural], 'spacing': flat_inputs}
    result = subprocess.run(['node', '--input-type=module', '-e', NODE_RUNNER],
                            input=json.dumps(request), text=True, capture_output=True, check=True)
    response = json.loads(result.stdout)
    assert response['weights_sha256'] == lineage['selected_weights_sha256']
    assert len(response['sentences']) == len(natural)
    predictions, segment_invariants = [], 0
    for row, segments in zip(natural, response['sentences']):
        text = row['text']
        valid = ''.join(segment['text'] for segment in segments) == text
        for segment in segments:
            offset = segment['offset']
            valid &= text[offset:offset + len(segment['text'])] == segment['text']
        boundaries = [segment['offset'] for segment in segments[1:]]
        valid &= boundaries == sorted(set(boundaries))
        valid &= all(0 < boundary < len(text) for boundary in boundaries)
        segment_invariants += bool(valid)
        predictions.append(boundaries)
    api_report = {'api': 'public JavaScript splitUnspaced, including wrapper gates',
                  **natural_info, 'source_sha256': sha_file(source),
                  'model_threshold': response['model']['heads']['sentences']['threshold'],
                  'segment_text_and_codepoint_offset_invariants': segment_invariants,
                  'evaluated_variants': len(natural), **score_natural(natural, predictions)}
    spacing_report = {}
    for mode in ['conservative', 'balanced']:
        spacing_report[mode] = {}
        for key, (rows, inputs) in tasks.items():
            a, b = slices[key]
            spacing_report[mode][key] = score_spaces(rows, inputs, response['spacing'][mode][a:b])
    baseline = {key: score_spaces(rows, inputs, inputs) for key, (rows, inputs) in tasks.items()}
    runtime_files = [args.javascript, args.javascript.parent / 'boundaries.js', args.javascript.parent / 'boundary-data.js']
    report = {'evaluation': 'Frozen selected public JS API; no model or threshold selection on these results.',
              'normalization': 'Corpus underscores already normalized during dataset preparation; runtime characters and whitespace are retained.',
              'partial_spacing_policy': 'Keep gold boundaries at indices 0,2,4,...; remove 1,3,5,... for every record.',
              'normal_preservation_scope': 'input_unchanged_rate on normal_spaces means explicit restoreSpacing calls; default analyze is unaffected.',
              'sample_scope': 'Rows include correlated punctuation variants of original blocks; exact rate is per block, not per independent sentence.',
              'source_manifest_sha256': sha_file(args.model / 'source-manifest.json'),
              'selected_weights_sha256': lineage['selected_weights_sha256'],
              'runtime_weights_sha256': response['weights_sha256'],
              'runtime_sha256': {str(path.resolve()): sha_file(path) for path in runtime_files},
              'node_version': response['node_version'], 'runtime_model': response['model'],
              'datasets': dataset_refs, 'natural_sentence_boundaries': api_report,
              'spacing': spacing_report, 'baseline_input_unchanged': baseline,
              'node_inference_seconds': response['inference_seconds'],
              'total_seconds': time.monotonic() - started}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'report': str(args.out), 'natural_sentence_boundaries': api_report,
                      'spacing': spacing_report, 'total_seconds': report['total_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
