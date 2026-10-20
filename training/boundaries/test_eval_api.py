import unittest

from eval_api import score_spaces


class PublicEvaluationTests(unittest.TestCase):
    def test_codepoint_positions(self):
        row = {'text': '😀가나', 'spaces': [1, 2]}
        result = score_spaces([row], ['😀가나'], ['😀 가 나'])
        self.assertEqual(result['tp'], 2)
        self.assertEqual(result['f1'], 1)
        self.assertEqual(result['exact'], 1)
        self.assertEqual(result['non_whitespace_unchanged_count'], 1)

    def test_removed_whitespace(self):
        row = {'text': '가나', 'spaces': [1]}
        result = score_spaces([row], ['가 나'], ['가나'])
        self.assertEqual(result['fn'], 1)
        self.assertEqual(result['input_unchanged_rate'], 0)
        self.assertEqual(result['non_whitespace_unchanged_count'], 1)
        self.assertEqual(result['existing_characters_subsequence_count'], 0)


if __name__ == '__main__':
    unittest.main()
