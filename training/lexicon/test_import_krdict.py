"""Lexical-field fixtures from the official 2026-10 JSON export.

The source is 국립국어원 한국어기초사전 (CC BY-SA 2.0 KR).
Only source lexical fields are retained; malformed/Unicode cases are synthetic.
"""
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


MODULE = Path(__file__).with_name('import_krdict.py')
SPEC = importlib.util.spec_from_file_location('import_krdict', MODULE)
IMPORTER = importlib.util.module_from_spec(SPEC) if MODULE.exists() else None
if IMPORTER is not None:
    SPEC.loader.exec_module(IMPORTER)


def entry(code, word, pos, unit='단어', category=''):
    features = [{'att': 'lexicalUnit', 'val': unit}]
    if pos:
        features.append({'att': 'partOfSpeech', 'val': pos})
    if category:
        features.append({'att': 'semanticCategory', 'val': category})
    return {'att': 'id', 'val': code, 'feat': features,
            'Lemma': {'feat': {'att': 'writtenForm', 'val': word}}}


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(IMPORTER, 'The dictionary importer must exist')

    def test_noun_keeps_category_and_unknown_boundaries(self):
        row = IMPORTER.normalize_entry(entry(
            '27968', '가공식품', '명사', category='식생활 > 음식'))
        self.assertEqual(row['surface'], '가공식품')
        self.assertEqual(row['pos'], 'NNG')
        self.assertEqual(row['source_id'], '27968')
        self.assertEqual(row['category'], '식생활 > 음식')
        self.assertEqual(row['components'], [])
        self.assertIn('compound_boundaries_unavailable', row['flags'])

    def test_coarse_noun_pos_does_not_claim_proper_name_label(self):
        row = IMPORTER.normalize_entry(entry(
            '70041', '대한민국', '명사', category='개념 > 지역'))
        self.assertEqual(row['pos'], 'NNG')
        self.assertIn('proper_noun_unresolved', row['flags'])

    def test_stems_only_known_verbal_pos(self):
        cases = [('14956', '가감하다', '동사', '가감하', 'VV'),
                 ('28031', '가공하다', '형용사', '가공하', 'VA'),
                 ('49985', '듯싶다', '보조 형용사', '듯싶', 'VX'),
                 ('26497', '바다', '명사', '바다', 'NNG'),
                 ('86232', '이다', '조사', '이다', None)]
        for code, word, pos, surface, tag in cases:
            with self.subTest(word=word, pos=pos):
                row = IMPORTER.normalize_entry(entry(code, word, pos))
                self.assertEqual((row['surface'], row['pos']), (surface, tag))
                if surface != word:
                    self.assertEqual(row['raw_surface'], word)

    def test_invalid_verbal_headword_not_blindly_truncated(self):
        row = IMPORTER.normalize_entry(entry('synthetic', '먹어', '동사'))
        self.assertEqual((row['surface'], row['pos'], row['kind']),
                         ('먹어', None, 'unsupported'))

    def test_phrases_never_become_atomic_candidates(self):
        cases = [('15065', '가격 인상', '품사 없음', '구'),
                 ('600515', '수분 크림', '명사', '단어'),
                 ('16391', '가난이 죄다', '', '속담'),
                 ('synthetic', '띄어쓰기없음', '명사', '관용구')]
        for code, word, pos, unit in cases:
            with self.subTest(word=word):
                row = IMPORTER.normalize_entry(entry(code, word, pos, unit))
                self.assertEqual((row['surface'], row['pos'], row['kind']),
                                 (word, None, 'phrase'))

    def test_inflected_lemma_list_is_preserved_without_inferred_pos(self):
        original = entry('16774', '가까워-', '품사 없음')
        original['Lemma'] = [original['Lemma'], {'feat': {
            'att': 'variant', 'val': '가까워, 가까워서, 가까웠다'}}]
        row = IMPORTER.normalize_entry(original)
        self.assertEqual((row['surface'], row['pos'], row['kind']),
                         ('가까워-', None, 'inflected'))
        self.assertIn('hyphen_marker', row['flags'])

    def test_unsupported_or_multiple_pos_remains_unmapped(self):
        for pos in ('접사', '어미', '조사', '품사 없음', '명사/부사'):
            with self.subTest(pos=pos):
                row = IMPORTER.normalize_entry(entry('synthetic', '-거나', pos))
                self.assertIsNone(row['pos'])
                self.assertEqual(row['raw_pos'], pos)
        original = entry('synthetic', '새', '명사')
        original['feat'].append({'att': 'partOfSpeech', 'val': '관형사'})
        row = IMPORTER.normalize_entry(original)
        self.assertIsNone(row['pos'])
        self.assertIn('complex_pos', row['flags'])

    def test_normalizes_nfc_and_outer_space_preserving_original(self):
        raw = ' 바다 '
        row = IMPORTER.normalize_entry(entry('synthetic', raw, '명사'))
        self.assertEqual(row['surface'], '바다')
        self.assertEqual(row['raw_surface'], raw)
        self.assertEqual(row['kind'], 'atomic')

    def test_bulk_archive_keeps_shared_parent_ids_and_no_prose(self):
        records = [entry('16391', '가난', '명사'),
                   entry('16391', '가난이 죄다', '', '속담')]
        records[0]['Sense'] = {'feat': {'att': 'definition', 'val': 'excluded'}}
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'source.zip'
            with zipfile.ZipFile(archive, 'w') as output:
                for number, record in enumerate(records):
                    payload = {'LexicalResource': {'Lexicon': {
                        'LexicalEntry': [record] if number else record}}}
                    output.writestr(f'{number}.json', json.dumps(payload))
            rows = list(IMPORTER.read_archive(archive))
        self.assertEqual(len(rows), 2)
        self.assertEqual([row['source_id'] for row in rows], ['16391', '16391'])
        self.assertEqual([row['kind'] for row in rows], ['atomic', 'phrase'])
        self.assertNotIn('excluded', json.dumps(rows))


if __name__ == '__main__':
    unittest.main()
