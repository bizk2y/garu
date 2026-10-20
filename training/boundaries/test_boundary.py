import unittest

from boundary import pack_parts, restore, hash_features
from infer import BoundaryModel, compact_input


class BoundaryTests(unittest.TestCase):
    def test_compaction_and_labels(self):
        row = pack_parts(['  아버지가 방에 ', ' 들어가신다. '])
        self.assertEqual(row['text'], '아버지가방에들어가신다.')
        self.assertEqual(row['spaces'], [4, 6])
        self.assertEqual(row['sentences'], [6])

    def test_underscore_unicode(self):
        row = pack_parts(['😀 A1_가 ', '𠮷B?'])
        self.assertEqual(row['text'], '😀A1가𠮷B?')
        self.assertEqual(row['spaces'], [1, 3, 4])
        self.assertEqual(row['sentences'], [4])
        self.assertEqual(restore(row['text'], row['spaces']), '😀 A1 가 𠮷B?')

    def test_empty_single(self):
        for text in ['', ' ', '\t\n', '😀']:
            row = pack_parts([text])
            self.assertEqual(row['spaces'], [])
            self.assertEqual(row['sentences'], [])
            matrix, slices = hash_features([row], 256)
            self.assertEqual(matrix.shape, (0, 256))
            self.assertEqual(slices, [(0, 0)])

    def test_terminal_removal(self):
        row = pack_parts(['값은 3.14이다.', '다음이다!'], strip_terminal=True)
        self.assertEqual(row['text'], '값은3.14이다다음이다')
        self.assertEqual(row['sentences'], [8])
        self.assertEqual(row['spaces'], [2, 8])

    def test_local_features(self):
        rows = [pack_parts(['나는밥을먹었다']), pack_parts(['나는밥을먹었다'])]
        matrix, spans = hash_features(rows, 256)
        a, b = spans
        self.assertEqual((matrix[a[0]:a[1]] != matrix[b[0]:b[1]]).nnz, 0)

    def test_runtime_keeps_chars(self):
        model = BoundaryModel.__new__(BoundaryModel)
        model.predict_many = lambda texts, head, threshold: [[1, 3], [], [1]]
        texts = ['😀A1_가', '  \t\n', '가 \t나']
        self.assertEqual(model.restore_many(texts), ['😀 A1 _가', '  \t\n', '가 \t나'])

    def test_collision_counts(self):
        # Dimension 2 puts all lexical and category features in the same bucket.
        matrix, _ = hash_features([{'text': '가나'}], 2)
        self.assertEqual(matrix[0, 0], 1)
        self.assertEqual(matrix[0, 1], 14)

    def test_js_whitespace_contract(self):
        self.assertEqual(compact_input('가\ufeff나\u00a0다'), '가나다')
        self.assertEqual(compact_input('가\u0085나'), '가\u0085나')


if __name__ == '__main__':
    unittest.main()
