import unittest
import json
import tempfile
import hashlib
from pathlib import Path

import numpy as np

from verifier import context_ids, pack_q4, unpack_q4, score_export, float_bytes, score_metrics, embed_context, build_arrays, export_model


class VerifierTests(unittest.TestCase):
    def test_small_export_uses_actual_shape(self):
        import torch
        model = torch.nn.Module()
        model.embedding = torch.nn.Embedding(4, 2, padding_idx=0)
        model.hidden = torch.nn.Linear(3 * 2 * 2 + 16, 5)
        model.output = torch.nn.Linear(5, 1)
        result = export_model(model, ['가', '나'], 'no_morph', 'source')
        self.assertEqual((result['context'], result['embeddingSize']), (3, 2))
        hidden = unpack_q4(result['tensors']['hidden'])
        self.assertTrue(np.any(hidden[:, :12] != 0))
        np.testing.assert_array_equal(hidden[:, 17:26], np.zeros((5, 9)))

    def test_padding_never_learns(self):
        import torch
        devices = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                layer = torch.nn.Embedding(5, 3, padding_idx=0).to(device)
                optimizer = torch.optim.AdamW(layer.parameters(), lr=0.001)
                ids = torch.tensor([[0, 2, 3, 0]], device=device)
                embed_context(layer, ids).sum().backward()
                self.assertEqual(layer.weight.grad[0].cpu().tolist(), [0, 0, 0])
                optimizer.step()
                self.assertEqual(layer.weight[0].detach().cpu().tolist(), [0, 0, 0])

    def test_q4_signed_rows(self):
        values = np.array([[-7, 0, 7], [-14, 2, 8]], dtype=np.float32)
        packed = pack_q4(values)
        # Signed values become nibbles 1,8,15,1,9,12 -> bytes 81,1f,c9.
        self.assertEqual(packed['data'], 'gR/J')
        np.testing.assert_array_equal(unpack_q4(packed), values)

    def test_zero_row(self):
        packed = pack_q4(np.zeros((1, 3), dtype=np.float32))
        np.testing.assert_array_equal(unpack_q4(packed), [[0, 0, 0]])

    def test_context_codepoints(self):
        self.assertEqual(context_ids('😀가나', 1, {'😀': 2, '가': 3}, 2), [0, 2, 3, 1])

    def test_context_end_padding(self):
        self.assertEqual(context_ids('가나', 1, {'가': 2, '나': 3}, 3), [0, 0, 2, 3, 0, 0])

    def test_q4_network_score(self):
        hidden = np.zeros((1, 18), dtype=np.float32)
        hidden[0, :2] = [1, 7]
        model = {'tensors': {'embedding': pack_q4([[0], [0], [1], [2]]),
                            'hidden': {**pack_q4(hidden), 'bias': float_bytes([0.5])},
                            'output': {**pack_q4([[2]]), 'bias': float_bytes([0.25])}}}
        actual = score_export(model, np.array([[2, 3]]), np.zeros((1, 16), dtype=np.float32))
        np.testing.assert_allclose(actual, [31.25], rtol=0, atol=1e-5)

    def test_missing_gold_in_recall(self):
        data = {'labels': np.array([1, 0]), 'groups': np.array(['NX', 'NX']), 'gold_totals': {'NX': 3}}
        result = score_metrics(data, np.array([1, 1]), 0)['NX']
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 1, 2))
        self.assertEqual(result['f1'], 0.4)

    def test_nonfinite_score_rejected(self):
        data = {'labels': np.array([1]), 'groups': np.array(['NX']), 'gold_totals': {'NX': 1}}
        with self.assertRaises(ValueError):
            score_metrics(data, np.array([np.nan]), 0)

    def test_score_shape_rejected(self):
        data = {'labels': np.array([1, 0]), 'groups': np.array(['NX', 'NX']), 'gold_totals': {'NX': 1}}
        with self.assertRaises(ValueError):
            score_metrics(data, np.ones((2, 1)), 0)

    def test_extra_features_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'data').mkdir()
            source = root / 'data/train.jsonl'
            source.write_text(json.dumps({'id': 'a', 'text': '가나', 'sentences': [], 'source': 'NX', 'variant': 'punctuated'}) + '\n')
            features = root / 'train-features.jsonl'
            features.write_text(json.dumps({'id': 'a', 'candidates': [], 'features': [],
                                            'extraCandidates': [{'compactOffset': 1}], 'extraFeatures': []}) + '\n')
            (root / 'train-features.meta.json').write_text(json.dumps({
                'dataset_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'features_sha256': hashlib.sha256(features.read_bytes()).hexdigest()}))
            with self.assertRaises(ValueError):
                build_arrays(root, 'train', {'가': 2, '나': 3})


if __name__ == '__main__':
    unittest.main()
