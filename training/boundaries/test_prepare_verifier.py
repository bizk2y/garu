import unittest
import gzip
import json
import tempfile
from pathlib import Path

from prepare_verifier import paragraph_groups, complete_utterance, group_keys, reference_keys


class PreparationTests(unittest.TestCase):
    def test_prior_holdout_refs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'nikl_mp_2021').mkdir()
            (root / 'nikl_mp_2021/SX.json').write_text(json.dumps({'document': [{'sentence': [
                {'id': 's1', 'form': '오늘 여기까지 하겠습니다.'}, {'id': 's2', 'form': '다음에 봐요.'}]}]}))
            ref = root / 'test.refs.jsonl.gz'
            with gzip.open(ref, 'wt') as stream:
                stream.write(json.dumps({'source_file': 'SX.json', 'sentence_ids': ['s1', 's2']}) + '\n')
            keys = reference_keys(root, [ref])
            self.assertIn('오늘여기까지하겠습니다', keys)
            self.assertIn('오늘여기까지하겠습니다다음에봐요', keys)

    def test_augmented_overlap_key(self):
        self.assertIn('가요와요', group_keys(['가요.', '와요.']))

    def test_preserve_paragraphs(self):
        rows = [{'id': 'NX.1.1.1'}, {'id': 'NX.1.1.2'}, {'id': 'NX.1.2.1'}]
        self.assertEqual(paragraph_groups(rows), [rows[:2], rows[2:]])

    def test_reject_gap(self):
        with self.assertRaises(ValueError):
            paragraph_groups([{'id': 'NX.1.1.1'}, {'id': 'NX.1.1.3'}])

    def test_single_final_ef(self):
        row = {'form': '밥을 먹었어요.', 'morpheme': [{'label': tag} for tag in ['NNG', 'JKO', 'VV', 'EP', 'EF', 'SF']]}
        self.assertTrue(complete_utterance(row))

    def test_internal_ef_excluded(self):
        row = {'form': '간다 했어요.', 'morpheme': [{'label': tag} for tag in ['VV', 'EF', 'VV', 'EP', 'EF', 'SF']]}
        self.assertFalse(complete_utterance(row))

    def test_fragment_excluded(self):
        row = {'form': '많이 먹었는데.', 'morpheme': [{'label': tag} for tag in ['MAG', 'VV', 'EP', 'EC', 'SF']]}
        self.assertFalse(complete_utterance(row))


if __name__ == '__main__':
    unittest.main()
