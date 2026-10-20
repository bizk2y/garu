"""Normalize a complete MeCab-ko-dic seed/release CSV directory for review.

Writes candidates and counts only; never modifies a Garu dictionary/model.
Pass the upstream ``seed`` directory or a release directory containing CSVs.
Subdirectory user-dic templates are not the compiled source dictionary.
Raw 12-column fields and source line references remain available in every row.
MeCab context IDs/costs are preserved as source metadata, never frequencies.
"""
import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


TAGS = set('NNG NNP NNB NR NP VV VA VX VCP VCN MAG MAJ MM IC '
           'JKS JKC JKG JKO JKB JKV JKQ JX JC EP EF EC ETN ETM '
           'XPN XSN XSV XSA XR SF SP SS SE SO SW SH SL SN'.split())
TAG_MAP = {'NNBC': 'NNB', 'SC': 'SP', 'SSO': 'SS', 'SSC': 'SS', 'SY': 'SW'}
KINDS = {'*': 'atomic', 'Compound': 'compound', 'Inflect': 'inflected',
         'Preanalysis': 'phrase'}
COLUMNS = ['surface', 'left_id', 'right_id', 'cost', 'pos', 'semantic',
           'has_jongseong', 'reading', 'type', 'first_pos', 'last_pos', 'expression']
EXAMPLE_LIMIT = 20


def map_pos(raw):
    return TAG_MAP.get(raw, raw if raw in TAGS else None)


def normalize(row, source_id, category):
    if len(row) != len(COLUMNS):
        raise ValueError(f'field_count:{len(row)}')
    if not row[0]:
        raise ValueError('empty_surface')

    surface = unicodedata.normalize('NFC', row[0])
    pos = map_pos(row[4])
    flags = []
    if row[4] in TAG_MAP:
        flags.append(f'mapped_pos:{row[4]}->{pos}')
    if '+' in row[4]:
        flags.append('complex_pos')
    for tag in sorted(set(row[4].split('+'))):
        if map_pos(tag) is None:
            flags.append(f'unsupported_pos:{tag}')
    if category == 'Wikipedia':
        flags.append('wikipedia')
    if category.split('-')[0] in {'Person', 'Place', 'Group'}:
        flags.append('named_entity_source')
    if category == 'Person-actor':
        flags.append('person_actor_source')

    components = []
    if row[11] != '*':
        for component in row[11].split('+'):
            parts = component.rsplit('/', 2)
            if len(parts) != 3:
                flags.append('malformed_component')
                continue
            form, tag, _ = parts
            components.append([unicodedata.normalize('NFC', form), map_pos(tag)])
            if not form:
                flags.append('empty_component')
            if map_pos(tag) is None:
                flags.append(f'unsupported_component_pos:{tag}')

    kind = KINDS.get(row[8], 'unsupported')
    if row[8] == 'Preanalysis':
        flags.append('preanalysis')
    if row[8] == '*' and components:
        kind = 'compound'
        flags.append('unmarked_components')
    if any(char.isspace() for char in surface):
        kind = 'phrase'
        flags.append('surface_whitespace')
    if kind == 'atomic' and pos is None:
        kind = 'unsupported'
    if row[8] not in KINDS:
        flags.append(f'unsupported_type:{row[8]}')

    result = {'surface': surface, 'pos': pos, 'source': 'mecab',
              'source_id': source_id, 'kind': kind, 'components': components,
              'raw_pos': row[4], 'category': category,
              'flags': sorted(set(flags)), 'raw_fields': row}
    if surface != row[0]:
        result['raw_surface'] = row[0]
    return result


def collect(source_dir, output):
    files = sorted(source_dir.glob('*.csv'))
    if not files:
        raise ValueError(f'No dictionary CSV files in {source_dir}')
    summary = {'source': 'mecab', 'source_dir': str(source_dir.resolve()),
               'raw_columns': COLUMNS, 'tag_mapping': TAG_MAP,
               'raw_records': 0, 'normalized_records': 0, 'atomic_records': 0,
               'exact_duplicate_records': 0, 'exclusions': Counter(),
               'excluded_examples': [], 'kinds': Counter(), 'flags': Counter(),
               'atomic_pos': Counter(), 'categories': Counter(), 'files': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as target:
        for file in files:
            entries = {}
            raw_count = 0
            with file.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.reader(stream, strict=True)
                for row in reader:
                    raw_count += 1
                    summary['raw_records'] += 1
                    source_id = f'{file.name}:{reader.line_num}'
                    try:
                        entry = normalize(row, source_id, file.stem)
                    except ValueError as error:
                        reason = str(error)
                        summary['exclusions'][reason] += 1
                        if len(summary['excluded_examples']) < EXAMPLE_LIMIT:
                            summary['excluded_examples'].append(
                                {'source_id': source_id, 'reason': reason, 'raw_fields': row})
                        continue
                    key = tuple(row)
                    if key in entries:
                        existing = entries[key]
                        existing.setdefault('source_ids', [existing['source_id']]).append(source_id)
                        summary['exact_duplicate_records'] += 1
                        continue
                    entries[key] = entry

            summary['files'].append({'file': file.name, 'bytes': file.stat().st_size,
                                     'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                                     'raw_records': raw_count, 'normalized_records': len(entries)})
            for entry in entries.values():
                target.write(json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n')
                summary['normalized_records'] += 1
                summary['kinds'][entry['kind']] += 1
                summary['flags'].update(entry['flags'])
                summary['categories'][entry['category']] += 1
                if entry['kind'] == 'atomic':
                    summary['atomic_records'] += 1
                    summary['atomic_pos'][entry['pos']] += 1
    summary['output_bytes'] = output.stat().st_size
    with output.open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    summary['output_sha256'] = digest.hexdigest()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    summary = collect(args.source_dir, args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
                            encoding='utf-8')
    print(json.dumps({key: summary[key] for key in
                      ('raw_records', 'normalized_records', 'atomic_records',
                       'exact_duplicate_records', 'kinds', 'exclusions')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
