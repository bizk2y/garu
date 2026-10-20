"""Replay reviewed lexical targets from their original corpus paragraphs.

Usage: python3 training/lexicon/replay.py --baseline /path/to/base.gmdl
Requires the original corpora under ~/workspace/data and the built JS package.
Scores target word/POS recognition, not full-paragraph morphological F1.
"""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEW = Path(__file__).with_name('review_20261020_context.json')
NODE = """
import {readFileSync} from 'node:fs';
import {Garu} from './js/dist/node.js';
const garu = await Garu.load({modelData: readFileSync(process.argv[1])});
const rows = JSON.parse(readFileSync(process.argv[2], 'utf8'));
console.log(JSON.stringify(rows.map(r => garu.analyze(r.text).tokens
    .map(t => [t.text, t.pos, t.start, t.end]))));
"""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_review(path):
    review = json.loads(path.read_text(encoding='utf-8'))
    if 'case_source' in review:
        reference = review['case_source']
        source = (path.parent / reference['file']).read_bytes()
        if digest(source) != reference['sha256']:
            raise ValueError('Case source changed')
        review['cases'] = json.loads(source)['cases']
    return review


def load_cases(review, data_dir):
    files = defaultdict(list)
    for row in review['cases']:
        files[row['file']].append(row)
    cases = []
    for file, references in files.items():
        wanted = {(r['doc_id'], r['paragraph_id']) for r in references}
        paragraphs = {}
        source = json.loads((data_dir / file).read_text(encoding='utf-8'))
        for doc in source['document']:
            if doc is None:
                continue
            for paragraph in doc.get('paragraph', []):
                key = (doc['id'], paragraph['id'])
                if key in wanted:
                    paragraphs[key] = paragraph['form']
        for ref in references:
            text = paragraphs[(ref['doc_id'], ref['paragraph_id'])]
            if digest(text.encode()) != ref['paragraph_sha256']:
                raise ValueError(f"Corpus changed: {ref['paragraph_id']}")
            mode = ref.get('text_mode', 'strip')
            if mode not in ('strip', 'raw'):
                raise ValueError(f"Unknown text mode: {mode}")
            if mode == 'strip':
                text = text.strip()
            if ref.get('kind') == 'guard':
                if not ref.get('token_guards'):
                    raise ValueError(f"Empty guard: {ref['paragraph_id']}")
                cases.append({**ref, 'text': text})
                continue
            if 'expected_targets' in ref:
                starts = []
                for target in ref['expected_targets']:
                    start, end = target['start'], target['end']
                    if (target['word'] != ref['word'] or target['pos'] != ref['pos']
                            or text[start:end] != ref['word']):
                        raise ValueError(f"Target offset changed: {ref['paragraph_id']}")
                    starts.append(start)
            else:
                starts = [m.start() for m in re.finditer(re.escape(ref['word']), text)]
            if not starts:
                raise ValueError(f"Target missing: {ref['paragraph_id']}")
            cases.append({**ref, 'text': text, 'starts': starts})
    return cases


def analyze(model, input_file):
    result = subprocess.run(['node', '--input-type=module', '-e', NODE,
                             str(model.resolve()), str(input_file)],
                            cwd=ROOT, capture_output=True, encoding='utf-8', check=True)
    return json.loads(result.stdout)


def hits(row, tokens):
    return [any(t[0] == row['word'] and t[1] == row['pos']
                and t[2] <= start and t[3] >= start + len(row['word']) for t in tokens)
            for start in row['starts']]


def check_guards(cases, outputs):
    for row, tokens in zip(cases, outputs):
        if row.get('kind') != 'guard':
            continue
        remaining = iter(tokens)
        for expected in row['token_guards']:
            if not any(token == expected for token in remaining):
                raise ValueError(f"Regression guard failed: {row['paragraph_id']}: {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', type=Path, default=ROOT / 'js/models/base.gmdl')
    parser.add_argument('--review', type=Path, default=REVIEW)
    parser.add_argument('--data-dir', type=Path, default=Path.home() / 'workspace/data')
    args = parser.parse_args()
    review = load_review(args.review)
    for name, path in [('baseline', args.baseline), ('candidate', args.candidate)]:
        if digest(path.read_bytes()) != review['models'][name]['sha256']:
            raise ValueError(f"Wrong {name} model for this recorded experiment")
    for file, expected in review.get('runtime_sha256', {}).items():
        if digest((ROOT / file).read_bytes()) != expected:
            raise ValueError(f"Wrong runtime for this recorded experiment: {file}")
    cases = load_cases(review, args.data_dir)
    with tempfile.TemporaryDirectory(prefix='garu-lexical-replay-') as temporary:
        input_file = Path(temporary) / 'cases.json'
        input_file.write_text(json.dumps(cases, ensure_ascii=False), encoding='utf-8')
        before = analyze(args.baseline, input_file)
        after = analyze(args.candidate, input_file)
    if len(before) != len(cases) or len(after) != len(cases):
        raise ValueError('Analyzer row count mismatch')
    check_guards(cases, after)
    totals = defaultdict(Counter)
    for row, b, a in zip(cases, before, after):
        if row.get('kind') == 'guard':
            continue
        old, new = hits(row, b), hits(row, a)
        totals[row['split']].update(samples=1, occurrences=len(old),
                                   before_samples=int(all(old)), after_samples=int(all(new)),
                                   before_occurrences=sum(old), after_occurrences=sum(new),
                                   target_regressions=sum(x and not y for x, y in zip(old, new)))
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    if set(totals) != set(review['lexical_metrics']):
        raise ValueError('Replay split set differs from the recorded experiment')
    for split, actual in totals.items():
        expected = review['lexical_metrics'][split]['totals']
        if any(value != expected[name] for name, value in actual.items()):
            raise ValueError(f'Replay differs from the recorded {split} result')


if __name__ == '__main__':
    main()
