import struct
import unittest

import context_cache as cache
from probe import build_model, score


class ContextCacheTest(unittest.TestCase):
    def test_score_multiplicity(self):
        actual = score([[["가", "JKS"]]],
                       [{"morphemes": [["가", "JKS"], ["가", "JKS"]]}])
        self.assertEqual(actual, {"f1": 2 / 3, "tp": 1, "predicted": 1, "gold": 2})

    def test_feature_boundaries(self):
        self.assertEqual(cache.contexts(None, "번째이며"),
                         [(0, ""), (4, "번"), (5, "번째"), (6, "번째이"),
                          (7, "며"), (8, "이며"), (9, "째이며")])

    def test_block_dev_duplicate(self):
        rows = [{"text": "비 가", "doc": "train"},
                {"text": "비가", "doc": "dev"},
                {"text": "눈이 온다", "doc": "train"}]
        assignments = [(r, cache.split_for(r, {"dev"}, {"비가"}, set())) for r in rows]
        train = [r for r, split in assignments if split == "train"]
        dev = [r for r, split in assignments if split == "dev"]
        self.assertEqual([r["text"] for r in train], ["눈이 온다"])
        self.assertEqual([r["text"] for r in dev], ["비가"])

    def test_patch_keeps_sections(self):
        original = b"GMDL" + struct.pack("<I", 3)
        original += struct.pack("<BI", 13, 3) + b"old"
        original += struct.pack("<BI", 14, 7) + b"weights"
        rule = {"surface": "네", "morphs": [["네", "MM"]],
                "guards": [[5, "번째"]]}
        patched = cache.patch_model(original, [rule], 2.0)
        sections = cache.sections(patched)
        self.assertEqual(sections[13], b"old")
        self.assertEqual(sections[14], b"weights")
        self.assertEqual(sections[16][:9], b"GCAC\x01" + struct.pack("<f", 2.0))
        self.assertEqual(cache.decode(sections[16]), (2.0, [rule]))

    def test_truncated_decode(self):
        blob = cache.encode([{"surface": "네", "morphs": [["네", "MM"]],
                              "guards": [[5, "번째"]]}], 0.0)
        for cut in range(len(blob)):
            with self.assertRaises(ValueError):
                cache.decode(blob[:cut])

    def test_reserved_model_field(self):
        original = b"GMDL" + struct.pack("<I", 3)
        original += struct.pack("<BI", 13, 3) + b"old"
        rule = {"surface": "네", "morphs": [["네", "MM"]], "guards": [[5, "번째"]]}
        model = build_model(original, [rule])
        self.assertEqual(cache.decode(cache.sections(model)[16]), (-1.0, [rule]))
        self.assertEqual(cache.sections(model)[13], b"old")


if __name__ == "__main__":
    unittest.main()
