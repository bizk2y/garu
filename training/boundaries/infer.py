"""Batch inference for experiments; does not modify existing whitespace or characters."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from boundary import hash_features

JS_WHITESPACE = frozenset('\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff')


def compact_input(text):
    return ''.join(c for c in text if c not in JS_WHITESPACE)


class BoundaryModel:
    def __init__(self, folder):
        folder = Path(folder)
        self.meta = json.loads((folder / 'report.json').read_text())
        self.weights = {}
        self.scales = {}
        for head in ['spaces', 'sentences']:
            data = np.load(folder / (head + '.npz'))
            self.weights[head] = data['weights'].astype(np.float64)
            self.scales[head] = float(data['scale'])

    def predict_many(self, texts, head='spaces', threshold=None):
        rows = [{'text': compact_input(text)} for text in texts]
        matrix, spans = hash_features(rows, self.meta['dimensions'], self.meta['features'])
        scores = (matrix @ self.weights[head]) * self.scales[head]
        if threshold is None:
            threshold = self.meta['heads'][head]['threshold']
        return [[i + 1 for i, score in enumerate(scores[a:b]) if score > threshold]
                for a, b in spans]

    def restore_many(self, texts, threshold=None):
        predictions = self.predict_many(texts, 'spaces', threshold)
        result = []
        for text, boundaries in zip(texts, predictions):
            boundaries = set(boundaries)
            chars, compact_index, previous_space = [], 0, False
            for char in text:
                if char in JS_WHITESPACE:
                    chars.append(char)
                    previous_space = True
                    continue
                if compact_index in boundaries and not previous_space:
                    chars.append(' ')
                chars.append(char)
                compact_index += 1
                previous_space = False
            output = ''.join(chars)
            assert compact_input(output) == compact_input(text)
            result.append(output)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()
    model = BoundaryModel(args.model)
    texts = [json.loads(line) for line in sys.stdin]
    for output in model.restore_many(texts):
        print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    main()
