"""Normalize the official Urimalsaem full APPROVED JSON export for review.

This is not the site's unreviewed contribution pool. Preserve every source row,
including repeated target_code values, with archive member/record provenance.
Only lexical metadata is emitted; definitions, examples and media stay excluded.
NNG/MAG are coarse POS proposals, not verified common-noun/adverb subtypes.
Internal hyphens mark immediate constituents, not a Sejong morpheme analysis.
No frequencies or model costs are inferred; no Garu model is modified.
"""
import argparse
import hashlib
import json
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path


POS_MAP = {'명사': 'NNG', '대명사': 'NP', '수사': 'NR', '동사': 'VV',
           '형용사': 'VA', '관형사': 'MM', '부사': 'MAG', '감탄사': 'IC',
           '의존 명사': 'NNB', '보조 동사': 'VX', '보조 형용사': 'VX'}
VERBAL_POS = {'동사', '형용사', '보조 동사', '보조 형용사'}
PHRASE_UNITS = {'구', '속담', '관용구'}
NAME_CATEGORIES = {'인명', '지명', '책명', '고유명 일반'}
TYPE_FLAGS = {'방언': 'dialect', '지역어(방언)': 'dialect',
              '옛말': 'archaic', '북한어': 'north_korean'}
HASH_CHUNK = 1024 * 1024


def normalize_entry(item):
    source_id = item.get('target_code')
    if source_id is None or str(source_id).strip() == '':
        raise ValueError('missing_source_id')
    word = item['wordinfo']
    sense = item['senseinfo']
    raw_surface = word['word']
    surface = unicodedata.normalize('NFC', raw_surface.strip())
    if not surface:
        raise ValueError(f'empty_surface:{source_id}')
    raw_pos = sense.get('pos', '')
    pos = POS_MAP.get(raw_pos)
    unit = word.get('word_unit', '')
    lexical_type = sense.get('type', '')
    categories = [c['cat'] for c in sense.get('cat_info', [])]
    flags = ['approved_export']
    components = []

    if lexical_type in TYPE_FLAGS:
        flags.append(TYPE_FLAGS[lexical_type])
    if raw_pos == '명사':
        flags.append('proper_noun_unresolved')
    if raw_pos == '부사':
        flags.append('adverb_subtype_unresolved')
    if NAME_CATEGORIES.intersection(categories):
        flags.append('named_entity_source')
    if '·' in raw_pos or '/' in raw_pos:
        flags.append('complex_pos')
    if pos is None:
        flags.append('unmapped_pos')

    has_space = any(c.isspace() for c in surface)
    if '^' in surface:
        flags.append('optional_spacing')
    if has_space:
        flags.append('surface_whitespace')
    if '-' in surface:
        flags.append('hyphen_marker')

    if unit in PHRASE_UNITS or '^' in surface or has_space:
        kind = 'phrase'
        pos = None
        surface = surface.replace('^', ' ')
    elif surface.startswith('-') or surface.endswith('-'):
        kind = 'unsupported'
        pos = None
        flags.append('affix_marker')
    else:
        parts = surface.split('-')
        has_constituents = len(parts) > 1
        if has_constituents:
            kind = 'compound'
            flags.extend(['immediate_constituents_only', 'component_pos_unavailable'])
        else:
            kind = 'atomic' if pos is not None else 'unsupported'
            flags.append('compound_boundaries_unavailable')

        if raw_pos in VERBAL_POS:
            if not surface.endswith('다') or len(surface) < 2:
                pos = None
                kind = 'unsupported'
                flags.append('invalid_verbal_lemma')
            else:
                parts[-1] = parts[-1][:-1]
                flags.append('lemma_to_stem')
                if not parts[-1]:
                    parts.pop()
                    flags.append('constituent_removed_by_stemming')
        surface = ''.join(parts)
        if has_constituents:
            components = [[part, None] for part in parts]

    result = {'surface': surface, 'pos': pos, 'source': 'opendict',
              'source_id': str(source_id), 'kind': kind, 'components': components,
              'raw_pos': raw_pos, 'category': ';'.join(categories),
              'flags': sorted(set(flags)), 'lexical_type': lexical_type,
              'word_unit': unit, 'word_type': word.get('word_type', ''),
              'reviewed_status': 'approved',
              'regions': [r['region'] for r in sense.get('region_info', [])]}
    for key in ('group_code', 'group_order'):
        if key in item:
            result[key] = item[key]
    if 'link' in item:
        result['source_url'] = item['link']
    if 'sense_no' in sense:
        result['sense_no'] = sense['sense_no']
    if surface != raw_surface:
        result['raw_surface'] = raw_surface
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()


def collect(source_zip, output):
    summary = {'source': 'opendict', 'scope': 'official_full_approved_export',
               'includes_unreviewed_contributions': False,
               'source_zip': str(source_zip.resolve()),
               'source_sha256': file_hash(source_zip),
               'tag_mapping': POS_MAP, 'raw_records': 0, 'normalized_records': 0,
               'atomic_records': 0, 'duplicate_source_ids': 0,
               'duplicate_occurrences': [], 'kinds': Counter(), 'flags': Counter(),
               'raw_pos': Counter(), 'atomic_pos': Counter(),
               'lexical_types': Counter(), 'categories': Counter(), 'files': []}
    first_occurrences = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip) as archive, output.open('w', encoding='utf-8') as target:
        members = sorted(name for name in archive.namelist() if name.endswith('.json'))
        if not members:
            raise ValueError('No JSON files in source archive')
        for member in members:
            raw = archive.read(member)
            channel = json.loads(raw)['channel']
            records = channel['item']
            if channel['total'] != len(records):
                raise ValueError(f'Archive count mismatch: {member}')
            summary['files'].append({'file': member, 'bytes': len(raw),
                                     'sha256': hashlib.sha256(raw).hexdigest(),
                                     'declared_total': channel['total'],
                                     'raw_records': len(records),
                                     'lastBuildDate': channel.get('lastBuildDate')})
            del raw
            for number, item in enumerate(records, start=1):
                summary['raw_records'] += 1
                row = normalize_entry(item)
                row.update(source_file=member, source_record=number)
                source_id = row['source_id']
                reference = f'{member}:{number}'
                if source_id in first_occurrences:
                    summary['duplicate_source_ids'] += 1
                    summary['duplicate_occurrences'].append({
                        'source_id': source_id, 'first': first_occurrences[source_id],
                        'repeated': reference})
                    row['flags'].append('duplicate_source_id')
                else:
                    first_occurrences[source_id] = reference
                target.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                summary['normalized_records'] += 1
                summary['kinds'][row['kind']] += 1
                summary['flags'].update(row['flags'])
                summary['raw_pos'][row['raw_pos']] += 1
                summary['lexical_types'][row['lexical_type']] += 1
                summary['categories'][row['category']] += 1
                if row['kind'] == 'atomic':
                    summary['atomic_records'] += 1
                    summary['atomic_pos'][row['pos']] += 1
            del records, channel
    summary['unique_source_ids'] = len(first_occurrences)
    summary['output_bytes'] = output.stat().st_size
    summary['output_sha256'] = file_hash(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-zip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    summary = collect(args.source_zip, args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
                            encoding='utf-8')
    print(json.dumps({key: summary[key] for key in
                      ('raw_records', 'normalized_records', 'unique_source_ids',
                       'duplicate_source_ids', 'atomic_records', 'kinds')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
