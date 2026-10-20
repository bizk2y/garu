"""Normalize the official 한국어기초사전 full JSON ZIP into lexical proposals.

Usage: python3 training/lexicon/import_krdict.py source.zip --output entries.jsonl

Source: https://krdict.korean.go.kr/download/downloadPopup
Lexical data: 국립국어원 한국어기초사전, CC BY-SA 2.0 KR. This extracts
no definitions, examples, translations or multimedia. NNG/MAG are coarse
mapping proposals: the source does not distinguish NNP or MAG/MAJ. A single
written word is not proof of a single morpheme; no decomposition is supplied.
The source repeats a parent target_code for attached proverbs/idioms, so
source_id is a provenance key, not a unique normalized-record identifier.
"""
import argparse
import json
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path


POS_MAP = {
    '명사': 'NNG', '의존 명사': 'NNB', '대명사': 'NP', '수사': 'NR',
    '동사': 'VV', '형용사': 'VA', '보조 동사': 'VX', '보조 형용사': 'VX',
    '부사': 'MAG', '관형사': 'MM', '감탄사': 'IC',
}
VERBAL_POS = {'동사', '형용사', '보조 동사', '보조 형용사'}
PHRASE_UNITS = {'구', '관용구', '속담', '문법‧표현'}
VERBAL_HEADWORD = re.compile(r'[가-힣]+다\Z')


def as_list(value):
    return value if isinstance(value, list) else [value]


def features(nodes):
    result = {}
    for node in as_list(nodes):
        for feature in as_list(node.get('feat', [])):
            result.setdefault(feature['att'], []).append(feature['val'])
    return result


def normalize_entry(entry):
    metadata = features(entry)
    lemma = features(entry['Lemma'])
    raw_surface, = lemma['writtenForm']
    surface = unicodedata.normalize('NFC', raw_surface).strip()
    raw_pos = '|'.join(dict.fromkeys(metadata.get('partOfSpeech', [])))
    unit = '|'.join(metadata.get('lexicalUnit', []))
    category = '|'.join(metadata.get('semanticCategory', []))
    pos = POS_MAP.get(raw_pos)
    kind = 'atomic'
    flags = []

    if raw_surface != surface:
        flags.append('surface_normalized')
    if '-' in surface:
        flags.append('hyphen_marker')
    if '^' in surface:
        flags.append('spacing_marker')
    if any(character.isspace() for character in surface):
        flags.append('multiword')
    if 'variant' in lemma:
        flags.append('variants_not_expanded')
    if raw_pos == '명사':
        flags.extend(['coarse_noun_pos', 'proper_noun_unresolved'])
    if raw_pos == '부사':
        flags.append('coarse_adverb_pos')
    if any(separator in raw_pos for separator in ('|', '/', ',', '·')):
        flags.append('complex_pos')

    if unit in PHRASE_UNITS or 'multiword' in flags or 'spacing_marker' in flags:
        kind, pos = 'phrase', None
        flags.append('non_atomic_entry')
        if unit in {'관용구', '속담'}:
            flags.append('parent_target_code')
    elif raw_pos == '품사 없음' and 'variant' in lemma and surface.endswith('-'):
        kind, pos = 'inflected', None
        flags.append('inflected_pos_unresolved')
    elif unit != '단어' or not pos or 'hyphen_marker' in flags or not surface:
        kind, pos = 'unsupported', None
        flags.append('unsupported_pos_or_form')
    elif raw_pos in VERBAL_POS:
        if VERBAL_HEADWORD.fullmatch(surface):
            surface = surface[:-1]
            flags.append('dictionary_da_removed')
        else:
            kind, pos = 'unsupported', None
            flags.append('verbal_headword_unrecognized')

    if kind == 'atomic':
        flags.append('compound_boundaries_unavailable')
    row = {'surface': surface, 'pos': pos, 'source': 'krdict',
           'source_id': str(entry['val']), 'kind': kind, 'components': [],
           'raw_pos': raw_pos, 'category': category, 'lexical_unit': unit,
           'flags': flags}
    if raw_surface != surface:
        row['raw_surface'] = raw_surface
    return row


def read_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith('.json'))
        if not names:
            raise ValueError('The archive contains no JSON dictionary files')
        for name in names:
            with archive.open(name) as stream:
                document = json.load(stream)
            entries = document['LexicalResource']['Lexicon']['LexicalEntry']
            for entry in as_list(entries):
                yield normalize_entry(entry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    counts = {key: Counter() for key in ('kind', 'pos', 'raw_pos', 'category',
                                        'lexical_unit', 'flags')}
    identifiers = set()
    total = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as output:
        for row in read_archive(args.archive):
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            total += 1
            identifiers.add(row['source_id'])
            for key, counter in counts.items():
                if key == 'flags':
                    counter.update(row[key])
                else:
                    counter[str(row[key])] += 1
    report = {'source': 'krdict', 'records': total,
              'unique_source_ids': len(identifiers), **counts}
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.report:
        args.report.write_text(encoded, encoding='utf-8')
    print(encoded, end='')


if __name__ == '__main__':
    main()
