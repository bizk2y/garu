"""Frozen-threshold external evaluation on exactly aligned 2025 NX original paragraphs."""
import argparse
import json
from pathlib import Path

from boundary import pack_parts, leak_key
from infer import BoundaryModel


def load_natural(training_files, source):
    trained = set()
    for training_file in training_files:
        for line in training_file.open():
            row = json.loads(line)
            points = [0] + row['sentences'] + [len(row['text'])]
            trained.update(leak_key(row['text'][a:b]) for a, b in zip(points, points[1:]))
    data = json.loads(source.read_text())
    rows, excluded, overlapped = [], 0, 0
    for doc in data['document']:
        by_paragraph = {}
        for sentence in doc['sentence']:
            by_paragraph.setdefault(sentence['id'].rsplit('.', 1)[0], []).append(sentence)
        for para in doc['paragraph']:
            sentences = by_paragraph.get(para['id'], [])
            cursor, parts, valid = 0, [], bool(sentences)
            for sentence in sentences:
                part = sentence.get('original_form', '').strip()
                at = para['form'].find(part, cursor)
                if not part or at < 0 or para['form'][cursor:at].strip():
                    valid = False
                    break
                parts.append(part)
                cursor = at + len(part)
            if not valid or para['form'][cursor:].strip():
                excluded += 1
                continue
            if any(leak_key(part) in trained for part in parts):
                overlapped += 1
                continue
            for strip_terminal in [False, True]:
                row = pack_parts(parts, strip_terminal)
                if not strip_terminal:
                    assert row['text'] == ''.join(para['form'].replace('_', ' ').split())
                row['paragraph_id'] = para['id']
                rows.append(row)
    info = {'source': str(source), 'excluded_alignment': excluded,
            'excluded_train_overlap': overlapped,
            'caveat': 'Raw source paragraph alignment is exact before whitespace/terminal deletion; labels are NIKL annotation convention, not manual adjudication.'}
    return rows, info


def score_natural(rows, predictions):
    assert len(rows) == len(predictions)
    report = {}
    for variant in ['punctuated', 'no_terminal']:
        tp = fp = fn = exact = n = 0
        for row, pred in zip(rows, predictions):
            if row['variant'] != variant:
                continue
            p, g = set(pred), set(row['sentences'])
            tp += len(p & g)
            fp += len(p - g)
            fn += len(g - p)
            exact += p == g
            n += 1
        report[variant] = {'paragraphs': n, 'tp': tp, 'fp': fp, 'fn': fn,
                           'precision': tp / (tp + fp) if tp + fp else 0,
                           'recall': tp / (tp + fn) if tp + fn else 0,
                           'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                           'exact': exact / n if n else 0}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--training', required=True)
    parser.add_argument('--data-root', default=str(Path.home() / 'workspace/data'))
    args = parser.parse_args()
    source = Path(args.data_root) / 'nikl_mp_2025/NXMP2502512313.json'
    rows, info = load_natural([Path(args.training) / 'train.jsonl'], source)
    model = BoundaryModel(args.model)
    predicted = model.predict_many([r['text'] for r in rows], 'sentences')
    report = {**info, 'threshold_frozen': model.meta['heads']['sentences']['threshold'],
              **score_natural(rows, predicted)}
    out = Path(args.model) / 'natural-2025-nx.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
