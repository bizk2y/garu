"""Reproduce the fixed context POS model and measure its native output.

Section 16 retains the legacy -1.0 field for byte-identical model reproduction;
the final POS runtime does not consume that field as a cost or tune it.
"""
import argparse
from collections import Counter
import hashlib
import heapq
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import brotli

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gold_testset/expand"))
from ep_norm import normalize_ep_morphemes
from context_cache import contexts, patch_model, sections

RESERVED_COST = -1.0


def build_model(original, rules):
    return brotli.compress(patch_model(original, rules, RESERVED_COST), quality=11)


def json_rows(path):
    with path.open() as source:
        for line in source:
            yield json.loads(line)


def score(predicted, rows):
    if len(predicted) != len(rows):
        raise ValueError("Prediction/reference count mismatch")
    hit = total_pred = total_gold = 0
    for prediction, row in zip(predicted, rows):
        p = Counter(map(tuple, normalize_ep_morphemes(prediction)))
        g = Counter(map(tuple, normalize_ep_morphemes(row["morphemes"])))
        hit += sum((p & g).values())
        total_pred += sum(p.values())
        total_gold += sum(g.values())
    return {"f1": 2 * hit / (total_pred + total_gold), "tp": hit,
            "predicted": total_pred, "gold": total_gold}


def matches(row, rules):
    words = row["text"].split()
    for i, word in enumerate(words):
        if word not in rules:
            continue
        features = contexts(words[i - 1] if i else None,
                            words[i + 1] if i + 1 < len(words) else None)
        if any(feature in rules[word] for feature in features):
            return True
    return False


def run(binary, model, rows, out, name):
    inputs = out / f"{name}.txt"
    inputs.write_text("".join(row["text"] + "\n" for row in rows))
    start = time.perf_counter()
    result = subprocess.run([str(binary), str(inputs), "--json"], check=True,
                            capture_output=True, text=True,
                            env={**os.environ, "GARU_MODEL": str(model)})
    elapsed = time.perf_counter() - start
    predictions = [json.loads(line) for line in result.stdout.splitlines()]
    if len(predictions) != len(rows):
        raise ValueError("Native output row count mismatch")
    (out / f"{name}.jsonl").write_text(result.stdout)
    return predictions, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    dev = list(json_rows(args.out / "dev.jsonl"))
    rule_list = json.loads((args.out / "rules.json").read_text())
    rules = {r["surface"]: set(map(tuple, r["guards"])) for r in rule_list}
    affected = [i for i, row in enumerate(dev) if matches(row, rules)]
    print(f"dev={len(dev)}, matched={len(affected)}, rules={len(rules)}", flush=True)
    baseline, base_time = run(args.baseline_binary, args.model, dev, args.out, "dev_base")
    inactive, _ = run(args.binary, args.model, dev, args.out, "dev_inactive")
    assert inactive == baseline, "No-Section16 output changed"
    result = {"dev": len(dev), "matched": len(affected), "rules": len(rules),
              "baseline": score(baseline, dev), "baseline_wall_seconds": base_time,
              "default_equal_rows": len(dev), "reserved_field_used_by_runtime": False}
    original = args.model.read_bytes()
    selected = args.out / "selected.gmdl"
    compressed = build_model(original, rule_list)
    assert all(sections(compressed)[s] == blob for s, blob in sections(original).items() if s != 16)
    selected.write_bytes(compressed)
    final, _ = run(args.binary, selected, dev, args.out, "dev_selected")
    affected_set = set(affected)
    assert all(final[i] == baseline[i] for i in range(len(dev)) if i not in affected_set)
    assert all([t[0] for t in before] == [t[0] for t in after] for before, after in zip(baseline, final))
    result["selected"] = {**score(final, dev),
                          "changed": sum(a != b for a, b in zip(final, baseline)),
                          "model_bytes": len(compressed),
                          "delta_bytes": len(compressed) - len(original),
                          "sha256": hashlib.sha256(compressed).hexdigest()}
    result["sources"] = {}
    for src in ["NX", "SX"]:
        indices = [i for i, row in enumerate(dev) if row["src"] == src]
        rows = [dev[i] for i in indices]
        result["sources"][src] = {"baseline": score([baseline[i] for i in indices], rows),
                                   "selected": score([final[i] for i in indices], rows)}
    sample = heapq.nsmallest(1000, json_rows(args.out / "train.jsonl"),
                            key=lambda r: hashlib.sha256(r["id"].encode()).digest())
    train_base, _ = run(args.baseline_binary, args.model, sample, args.out, "train_base")
    train_new, _ = run(args.binary, selected, sample, args.out, "train_selected")
    result["train_sample"] = {"n": len(sample), "baseline": score(train_base, sample),
                               "selected": score(train_new, sample)}
    changes = [{"id": row["id"], "doc": row["doc"], "src": row["src"], "text": row["text"],
                "gold": row["morphemes"], "baseline": old, "selected": new,
                "baseline_f1": score([old], [row])["f1"],
                "selected_f1": score([new], [row])["f1"]}
               for row, old, new in zip(dev, baseline, final) if old != new]
    (args.out / "dev_changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2))
    (args.out / "probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
