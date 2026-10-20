"""Emit the complete selected Q4 model as a separately imported typed module."""
import argparse
import json
from pathlib import Path

from verifier import FEATURE_NAMES, MODEL_LIMIT_BYTES, unpack_q4


def export_typescript(source, destination):
    model = json.loads(source.read_text())
    if model['version'] != 1 or model['featureNames'] != FEATURE_NAMES:
        raise ValueError('Incompatible sentence model contract')
    if unpack_q4(model['tensors']['embedding'])[0].any():
        raise ValueError('Padding embedding must be zero')
    compact = json.dumps(model, ensure_ascii=False, separators=(',', ':'))
    if len((compact + '\n').encode()) > MODEL_LIMIT_BYTES:
        raise ValueError('Complete model exceeds the delivery budget')
    text = "import type { SentenceModelData } from './sentence-verifier.js';\n\n"
    text += '/** Generated from the development-selected local Q4 sentence verifier. */\n'
    text += 'export const SENTENCE_MODEL: SentenceModelData = ' + compact + ';\n'
    destination.write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    export_typescript(args.model, args.out)
