"""Lexical fields from NIKL Urimalsaem's 20261004 approved JSON export.

Source: https://opendict.korean.go.kr/member/memberDownloadList
CC BY-SA 2.0 KR; definitions/examples/media are deliberately omitted.
Malformed and Unicode cases are explicitly synthetic.
"""
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


MODULE = Path(__file__).with_name('import_opendict.py')
SPEC = importlib.util.spec_from_file_location('import_opendict', MODULE)
IMPORTER = importlib.util.module_from_spec(SPEC) if MODULE.exists() else None
if IMPORTER is not None:
    SPEC.loader.exec_module(IMPORTER)


def entry(code, word, pos='', unit='어휘', lexical_type='일반어', category=''):
    sense = {'type': lexical_type}
    if pos:
        sense['pos'] = pos
    if category:
        sense['cat_info'] = [{'cat': category}]
    return {'target_code': code, 'wordinfo': {'word': word, 'word_unit': unit},
            'senseinfo': sense}


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(IMPORTER, 'The dictionary importer must exist')

    def test_noun_mapping_keeps_uncertainty_and_approved_scope(self):
        row = IMPORTER.normalize_entry(entry(481443, '검인', '명사'))
        self.assertEqual((row['surface'], row['pos'], row['source_id']),
                         ('검인', 'NNG', '481443'))
        self.assertEqual(row['reviewed_status'], 'approved')
        self.assertIn('proper_noun_unresolved', row['flags'])
        self.assertIn('compound_boundaries_unavailable', row['flags'])

    def test_named_entity_category_does_not_silently_retag_nnp(self):
        row = IMPORTER.normalize_entry(entry(
            1411896, '소이', '명사', category='지명'))
        self.assertEqual((row['category'], row['pos']), ('지명', 'NNG'))
        self.assertIn('named_entity_source', row['flags'])

    def test_hyphen_constituents_have_no_invented_pos(self):
        row = IMPORTER.normalize_entry(entry(426250, '소의-간식', '명사'))
        self.assertEqual((row['surface'], row['kind']), ('소의간식', 'compound'))
        self.assertEqual(row['components'], [['소의', None], ['간식', None]])
        self.assertEqual(row['raw_surface'], '소의-간식')
        self.assertIn('immediate_constituents_only', row['flags'])

    def test_known_verbal_pos_maps_dictionary_form_to_stem(self):
        row = IMPORTER.normalize_entry(entry(14124, '소음-하다', '동사'))
        self.assertEqual((row['surface'], row['pos'], row['kind']),
                         ('소음하', 'VV', 'compound'))
        self.assertEqual(row['components'], [['소음', None], ['하', None]])
        self.assertIn('lemma_to_stem', row['flags'])
        noun = IMPORTER.normalize_entry(entry('synthetic', '바다', '명사'))
        self.assertEqual(noun['surface'], '바다')

    def test_stem_suffix_does_not_leave_an_empty_constituent(self):
        row = IMPORTER.normalize_entry(entry(1280833, '졸르-다', '동사'))
        self.assertEqual((row['surface'], row['pos'], row['kind']),
                         ('졸르', 'VV', 'compound'))
        self.assertEqual(row['components'], [['졸르', None]])
        self.assertIn('constituent_removed_by_stemming', row['flags'])

    def test_phrases_keep_spacing_without_single_token_pos(self):
        cases = [(1028962, '소음^측정^시각', '', '구', '소음 측정 시각'),
                 (607312, '검정개 굿 구경', '', '속담', '검정개 굿 구경'),
                 (607311, '검정개 한 마리 얼씬하지 않다', '', '관용구',
                  '검정개 한 마리 얼씬하지 않다'),
                 ('synthetic', '띄어쓰기없음', '명사', '구', '띄어쓰기없음')]
        for code, word, pos, unit, expected in cases:
            with self.subTest(word=word):
                row = IMPORTER.normalize_entry(entry(code, word, pos, unit))
                self.assertEqual((row['surface'], row['pos'], row['kind']),
                                 (expected, None, 'phrase'))

    def test_dialect_region_and_nonmodern_types_are_preserved(self):
        original = entry(82852, '검자리', '명사', lexical_type='방언')
        original['senseinfo']['region_info'] = [{'region': '전남'}]
        row = IMPORTER.normalize_entry(original)
        self.assertEqual((row['lexical_type'], row['regions']), ('방언', ['전남']))
        self.assertIn('dialect', row['flags'])
        for code, word, lexical_type, flag in [
                (413055, '검잠', '옛말', 'archaic'),
                (82863, '검정간그릇', '북한어', 'north_korean')]:
            row = IMPORTER.normalize_entry(entry(
                code, word, '명사', lexical_type=lexical_type))
            self.assertIn(flag, row['flags'])

    def test_affix_markers_and_multiple_pos_are_not_atomic(self):
        for original in [entry(463215, '-소이', '접사', lexical_type='옛말'),
                         entry(487000, '-것다', '어미')]:
            row = IMPORTER.normalize_entry(original)
            self.assertEqual(row['kind'], 'unsupported')
            self.assertIsNone(row['pos'])
            self.assertEqual(row['components'], [])
            self.assertIn('affix_marker', row['flags'])
        row = IMPORTER.normalize_entry(entry(623448, '소인배-적', '관·명'))
        self.assertIsNone(row['pos'])
        self.assertEqual(row['raw_pos'], '관·명')
        self.assertIn('complex_pos', row['flags'])

    def test_invalid_verbal_form_and_missing_source_id(self):
        row = IMPORTER.normalize_entry(entry('synthetic', '먹어', '동사'))
        self.assertEqual((row['surface'], row['pos'], row['kind']),
                         ('먹어', None, 'unsupported'))
        original = entry('synthetic', '바다', '명사')
        del original['target_code']
        with self.assertRaisesRegex(ValueError, 'source_id'):
            IMPORTER.normalize_entry(original)

    def test_nfc_and_outer_whitespace_keep_original(self):
        row = IMPORTER.normalize_entry(entry('synthetic', ' 바다 ', '명사'))
        self.assertEqual((row['surface'], row['raw_surface']), ('바다', ' 바다 '))

    def test_archive_preserves_duplicates_and_excludes_prose(self):
        original = entry(481443, '검인', '명사')
        original['senseinfo']['example_info'] = [{'example': 'exclude-this'}]
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'source.zip'
            output = Path(directory) / 'normalized.jsonl'
            with zipfile.ZipFile(archive, 'w') as target:
                for name in ('part1.json', 'part2.json'):
                    target.writestr(name, json.dumps({'channel': {
                        'total': 1, 'item': [original],
                        'lastBuildDate': '20261003020742'}}))
            summary = IMPORTER.collect(archive, output)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual((summary['raw_records'], summary['normalized_records']), (2, 2))
        self.assertEqual(summary['duplicate_source_ids'], 1)
        self.assertEqual(summary['unique_source_ids'], 1)
        self.assertEqual([r['source_id'] for r in rows], ['481443', '481443'])
        self.assertEqual([r['source_file'] for r in rows], ['part1.json', 'part2.json'])
        self.assertNotIn('exclude-this', json.dumps(rows))

    def test_archive_declared_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'source.zip'
            with zipfile.ZipFile(archive, 'w') as target:
                target.writestr('part.json', json.dumps({'channel': {
                    'total': 2, 'item': [entry(481443, '검인', '명사')]}}))
            with self.assertRaisesRegex(ValueError, 'count mismatch'):
                IMPORTER.collect(archive, Path(directory) / 'normalized.jsonl')


if __name__ == '__main__':
    unittest.main()
