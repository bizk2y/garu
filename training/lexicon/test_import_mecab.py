import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name('import_mecab.py')


class MecabImportTests(unittest.TestCase):
    def run_import(self, files):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / 'source'
        source.mkdir()
        for name, rows in files.items():
            with (source / name).open('w', newline='', encoding='utf-8') as stream:
                csv.writer(stream).writerows(rows)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), '--source-dir', str(source),
             '--output', str(root / 'entries.jsonl'),
             '--summary', str(root / 'summary.json')],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        entries = [json.loads(line) for line in
                   (root / 'entries.jsonl').read_text().splitlines()]
        summary = json.loads((root / 'summary.json').read_text())
        return entries, summary

    def test_atomic_preserves_fields_without_cost_frequency(self):
        raw = ['가루', '1780', '3533', '2639', 'NNG', '*', 'F',
               '가루', '*', '*', '*', '*']
        entries, summary = self.run_import({'NNG.csv': [raw]})
        row, = entries
        self.assertEqual(row['surface'], '가루')
        self.assertEqual(row['raw_surface'], '가루')
        self.assertEqual(row['pos'], 'NNG')
        self.assertEqual(row['kind'], 'atomic')
        self.assertEqual(row['source_id'], 'NNG.csv:1')
        self.assertEqual(row['raw_fields'], raw)
        self.assertNotIn('frequency', row)
        self.assertEqual(summary['raw_records'], 1)

    def test_complex_analyses_are_not_atomic(self):
        entries, summary = self.run_import({'NNG.csv': [
            ['한국어', '0', '0', '0', 'NNG', '*', 'F', '한국어',
             'Compound', '*', '*', '한국/NNG/*+어/NNG/*'],
            ['입니다', '0', '0', '0', 'VCP+EF', '*', 'F', '입니다',
             'Inflect', 'VCP', 'EF', '이/VCP/*+ᄇ니다/EF/*'],
            ['반여성주의', '0', '0', '0', 'NNG', '*', 'F', '반여성주의',
             '*', '*', '*', '반/NNG/*+여성/NNG/*+주의/NNG/*'],
            ['월드와이드웹', '0', '0', '0', 'NNP', '*', 'T', '월드와이드웹',
             'Preanalysis', 'NNG', 'NNG', '월드/NNG/*+와이드/NNG/*+웹/NNG/*'],
        ]})
        self.assertEqual([r['kind'] for r in entries],
                         ['compound', 'inflected', 'compound', 'phrase'])
        self.assertEqual(entries[1]['pos'], None)
        self.assertEqual(entries[0]['components'], [['한국', 'NNG'], ['어', 'NNG']])
        self.assertIn('unmarked_components', entries[2]['flags'])
        self.assertEqual(summary['atomic_records'], 0)

    def test_tag_mapping_and_unsupported_tags_are_explicit(self):
        entries, _ = self.run_import({'Symbol.csv': [
            ['개', '0', '0', '0', 'NNBC', '*', 'F', '개', '*', '*', '*', '*'],
            [',', '0', '0', '0', 'SC', '*', '*', ',', '*', '*', '*', '*'],
            ['(', '0', '0', '0', 'SSO', '*', '*', '(', '*', '*', '*', '*'],
            ['몰라', '0', '0', '0', 'UNA', '*', 'F', '몰라', '*', '*', '*', '*'],
        ]})
        self.assertEqual([r['pos'] for r in entries], ['NNB', 'SP', 'SS', None])
        self.assertIn('mapped_pos:NNBC->NNB', entries[0]['flags'])
        self.assertEqual(entries[3]['kind'], 'unsupported')

    def test_risk_categories_whitespace_and_empty_components(self):
        entries, _ = self.run_import({
            'Wikipedia.csv': [['지명', '0', '0', '0', 'NNP', '*', 'T', '지명',
                               '*', '*', '*', '*']],
            'Person.csv': [['존홀', '0', '0', '0', 'NNP', '인명', 'T', '존홀',
                            'Preanalysis', 'NNP', 'NNP', '존/NNP/*+/NNP/*+홀/NNP/*']],
            'Foreign.csv': [['내셔날 ', '0', '0', '0', 'NNG', '*', 'T', '내셔날',
                             '*', '*', '*', '*']],
        })
        rows = {r['category']: r for r in entries}
        self.assertIn('wikipedia', rows['Wikipedia']['flags'])
        self.assertIn('named_entity_source', rows['Person']['flags'])
        self.assertIn('empty_component', rows['Person']['flags'])
        self.assertEqual(rows['Person']['components'][1], ['', 'NNP'])
        self.assertEqual(rows['Foreign']['surface'], '내셔날 ')
        self.assertEqual(rows['Foreign']['kind'], 'phrase')

    def test_exact_dedup_keeps_lineage_and_semantic_variants(self):
        raw = ['가루', '0', '0', '0', 'NNG', '*', 'F', '가루', '*', '*', '*', '*']
        other = raw.copy()
        other[5] = '물질'
        entries, summary = self.run_import({'NNG.csv': [raw, raw, other]})
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]['source_ids'], ['NNG.csv:1', 'NNG.csv:2'])
        self.assertEqual(summary['exact_duplicate_records'], 1)
        self.assertEqual(summary['raw_records'], 3)
        self.assertEqual(summary['normalized_records'], 2)

    def test_bad_schema_is_counted_without_losing_valid_rows(self):
        entries, summary = self.run_import({'NNG.csv': [
            ['broken', 'too', 'short'],
            ['가루', '0', '0', '0', 'NNG', '*', 'F', '가루', '*', '*', '*', '*'],
        ]})
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['source_id'], 'NNG.csv:2')
        self.assertEqual(summary['exclusions']['field_count:3'], 1)
        self.assertEqual(summary['excluded_examples'][0]['source_id'], 'NNG.csv:1')


if __name__ == '__main__':
    unittest.main()
