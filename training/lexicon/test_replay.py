import json
import tempfile
import unittest
from pathlib import Path

import replay
from replay import digest, hits, load_cases


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.text = ' 나이아신아마이드와 나이아신 '
        source = {'document': [{'id': 'doc', 'paragraph': [
            {'id': 'paragraph', 'form': self.text}]}]}
        (self.root / 'source.json').write_text(json.dumps(source), encoding='utf-8')
        self.ref = {
            'file': 'source.json', 'doc_id': 'doc', 'paragraph_id': 'paragraph',
            'paragraph_sha256': digest(self.text.encode()),
            'word': '나이아신', 'pos': 'NNG', 'text_mode': 'raw',
            'expected_targets': [{'word': '나이아신', 'pos': 'NNG', 'start': 11, 'end': 15}],
        }

    def test_explicit_targets_exclude_embedded(self):
        row, = load_cases({'cases': [self.ref]}, self.root)
        self.assertEqual(row['text'], self.text)
        self.assertEqual(row['starts'], [11])
        self.assertEqual(hits(row, [['나이아신', 'NNG', 1, 5]]), [False])
        self.assertEqual(hits(row, [['나이아신', 'NNG', 11, 15]]), [True])

    def test_rejects_changed_source(self):
        self.ref['paragraph_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'Corpus changed'):
            load_cases({'cases': [self.ref]}, self.root)

    def test_rejects_wrong_offset(self):
        self.ref['expected_targets'][0]['start'] = 10
        with self.assertRaisesRegex(ValueError, 'Target offset changed'):
            load_cases({'cases': [self.ref]}, self.root)

    def test_legacy_strip_and_all_matches(self):
        del self.ref['expected_targets']
        del self.ref['text_mode']
        row, = load_cases({'cases': [self.ref]}, self.root)
        self.assertEqual(row['text'], self.text.strip())
        self.assertEqual(row['starts'], [0, 10])

    def test_guard_keeps_morph_spans(self):
        ref = {key: value for key, value in self.ref.items()
               if key not in ('word', 'pos', 'expected_targets')}
        ref.update(kind='guard', token_guards=[['이', 'VCP', 11, 15]])
        row, = load_cases({'cases': [ref]}, self.root)
        self.assertEqual(row['text'], self.text)
        replay.check_guards([row], [[['이', 'VCP', 11, 15]]])

    def test_guard_rejects_pos_change(self):
        row = {'paragraph_id': 'p', 'kind': 'guard',
               'token_guards': [['나', 'NP', 0, 2], ['는', 'JX', 0, 2]]}
        with self.assertRaisesRegex(ValueError, 'Regression guard failed'):
            replay.check_guards([row], [[['나', 'VV', 0, 2], ['는', 'ETM', 0, 2]]])

    def test_guard_rejects_reordering(self):
        row = {'paragraph_id': 'p', 'kind': 'guard',
               'token_guards': [['이', 'VCP', 0, 2], ['며', 'EC', 0, 2]]}
        with self.assertRaisesRegex(ValueError, 'Regression guard failed'):
            replay.check_guards([row], [[['며', 'EC', 0, 2], ['이', 'VCP', 0, 2]]])

    def test_reuses_case_source(self):
        source = self.root / 'cases.json'
        source.write_text(json.dumps({'cases': [self.ref]}), encoding='utf-8')
        review = self.root / 'review.json'
        review.write_text(json.dumps({'case_source': {
            'file': source.name, 'sha256': digest(source.read_bytes()),
        }}), encoding='utf-8')
        self.assertEqual(replay.load_review(review)['cases'], [self.ref])

    def test_changed_case_source(self):
        source = self.root / 'cases.json'
        source.write_text(json.dumps({'cases': [self.ref]}), encoding='utf-8')
        review = self.root / 'review.json'
        review.write_text(json.dumps({'case_source': {
            'file': source.name, 'sha256': 'wrong',
        }}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Case source changed'):
            replay.load_review(review)


if __name__ == '__main__':
    unittest.main()
