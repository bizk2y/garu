"""Train-only, document-split probe for context-gated cache alternatives."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "gold_testset/expand")]
from ep_norm import normalize_ep_morphemes
from eval_nikl_mp import normalize_pos
from patch_cache_add_candidates import parse_cache
from context_cache import collapse, contexts, sections, split_for


def normalized(morphs):
    return tuple(tuple(m) for m in normalize_ep_morphemes(morphs))


def documents(path):
    """Stream the retained, indented NIKL 2021 original one document at a time."""
    lines = None
    with path.open() as source:
        for line in source:
            if line == "        {\n":
                lines = []
            if lines is None:
                continue
            lines.append(line)
            if line in ("        },\n", "        }\n"):
                yield json.loads("".join(lines).rstrip().rstrip(","))
                lines = None
    if lines is not None:
        raise ValueError(f"Incomplete document: {path}")


def json_rows(path):
    with path.open() as source:
        for line in source:
            yield json.loads(line)


def row_for(sent, doc, src):
    text = sent["form"]
    if text != text.strip() or any(c in text for c in "\n\r\t"):
        return None
    source = sent.get("MP") or sent.get("morpheme") or []
    by_word = defaultdict(list)
    gold = []
    for morph in source:
        if not morph.get("form", "").strip():
            continue
        pair = [morph["form"], normalize_pos(morph["label"])]
        by_word[morph.get("word_id")].append(pair)
        gold.append(pair)
    words = []
    for word in sent.get("word") or []:
        if text[word["begin"]:word["end"]] != word["form"]:
            return None
        words.append([word["form"], word["begin"], word["end"], by_word[word["id"]]])
    if normalized([m for word in words for m in word[3]]) != normalized(gold):
        return None
    if [w[0] for w in words] != text.split():
        return None
    return {"id": sent["id"], "doc": doc, "src": src, "text": text,
            "morphemes": gold, "words": words}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-rules", type=int, default=64)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = Path.home() / "workspace/data"
    train_source = HERE.parent / "rerank/data/train.jsonl"
    allowed = {(r["src"], r["text"]) for r in json_rows(train_source)}
    excluded = {collapse(r["text"]) for path in [HERE.parent / "rerank/data/dev.jsonl",
                HERE.parent / "gold_testset/gold_testset.jsonl"] for r in json_rows(path)}
    # Only text is inspected: no 2025 gold labels enter candidate/condition learning.
    spoken_path = data / "nikl_mp_2025/SXMP2502512312.json"
    spoken = json.loads(spoken_path.read_text())
    excluded.update(collapse(s.get("form", "").replace("_", " "))
                    for d in spoken["document"] if d
                    for s in d.get("sentence") or [])
    del spoken
    counts, dev_keys, dev_docs = Counter(), set(), set()
    eligible = args.out / "eligible.jsonl"
    with eligible.open("w") as output:
        for path in sorted((data / "nikl_mp_2021").glob("*.json")):
            for doc in documents(path):
                if not doc:
                    continue
                did, src = doc["id"], path.name[:2]
                is_dev = int(hashlib.sha256(did.encode()).hexdigest()[:8], 16) % 10 == 0
                if is_dev:
                    dev_docs.add(did)
                for sent in doc.get("sentence") or []:
                    text = sent["form"]
                    if (src, text) not in allowed or collapse(text) in excluded:
                        continue
                    row = row_for(sent, did, src)
                    if row is None:
                        counts["alignment_excluded"] += 1
                        continue
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if is_dev:
                        dev_keys.add(collapse(text))
            print(f"Read {path.name}", flush=True)
    del allowed
    _, cache_entries = parse_cache(sections(args.model.read_bytes())[13])
    cache = dict(cache_entries)
    cache_norm = {word: normalized(morphs) for word, morphs in cache.items()}
    variants, support_docs = defaultdict(Counter), defaultdict(set)
    duplicate_counts, split_docs = Counter(), defaultdict(set)
    with (args.out / "train.jsonl").open("w") as train, (args.out / "dev.jsonl").open("w") as dev:
        for row in json_rows(eligible):
            key = collapse(row["text"])
            split = split_for(row, dev_docs, dev_keys, excluded)
            if split is None:
                counts["cross_split_duplicates"] += 1
                continue
            # One copy per source/text: recurring boilerplate cannot multiply evidence.
            duplicate_key = (split, row["src"], key)
            if duplicate_counts[duplicate_key]:
                counts["within_split_duplicates"] += 1
                continue
            duplicate_counts[duplicate_key] += 1
            split_docs[split].add(row["doc"])
            (train if split == "train" else dev).write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
            if split != "train":
                continue
            for word, _, _, morphs in row["words"]:
                if word not in cache_norm:
                    continue
                target = normalized(morphs)
                variants[word][target] += 1
                support_docs[word, target].add(row["doc"])
    eligible.unlink()
    candidates = []
    for word, analyses in variants.items():
        base = cache_norm[word]
        for alt, support in analyses.items():
            if len(base) != len(alt) or [m[0] for m in base] != [m[0] for m in alt]:
                continue
            changes = [i for i, (a, b) in enumerate(zip(base, alt)) if a[1] != b[1]]
            if len(changes) != 1 or support < 10 or len(support_docs[word, alt]) < 3:
                continue
            at = changes[0]
            if base[at][1].startswith("S") or alt[at][1].startswith("S"):
                continue
            morphs = [list(m) for m in cache[word]]
            if len(morphs) != len(base):
                continue
            morphs[at][1] = alt[at][1]
            candidates.append((support, word, alt, morphs))
    chosen = {}
    for support, word, alt, morphs in sorted(candidates, key=lambda x: (-x[0], x[1])):
        chosen.setdefault(word, {"target": alt, "morphs": morphs, "support": support})
        if len(chosen) == args.max_rules:
            break
    # A rule's negative count includes every other reference, not only the base cache.
    stats = defaultdict(lambda: [0, 0, set()])
    for row in json_rows(args.out / "train.jsonl"):
        words = row["words"]
        for i, (word, _, _, morphs) in enumerate(words):
            if word not in chosen:
                continue
            good = normalized(morphs) == chosen[word]["target"]
            features = contexts(words[i - 1][0] if i else None,
                                words[i + 1][0] if i + 1 < len(words) else None)
            for kind, value in features:
                stat = stats[word, kind, value]
                stat[0 if good else 1] += 1
                if good:
                    stat[2].add(row["doc"])
    guards = defaultdict(list)
    for (word, kind, value), (good, bad, docs) in stats.items():
        if good >= 5 and len(docs) >= 3 and good / (good + bad) >= 0.95:
            guards[word].append((good, bad, len(docs), kind, value))
    rules = []
    evidence = []
    for word in sorted(guards):
        selected = sorted(guards[word], key=lambda x: (-x[0], x[1], x[3], x[4]))[:4]
        rules.append({"surface": word, "morphs": chosen[word]["morphs"],
                      "guards": [[kind, value] for _, _, _, kind, value in selected]})
        evidence.append({"surface": word, "support": chosen[word]["support"],
                         "conditions": selected})
    (args.out / "rules.json").write_text(json.dumps(rules, ensure_ascii=False, indent=2))
    (args.out / "rule_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    manifest = {"counts": dict(counts), "documents": {k: len(v) for k, v in split_docs.items()},
                "candidate_surfaces": len(chosen), "rules": len(rules),
                "split": "sha256(doc_id)[:8] mod 10 == 0 => dev; two-pass collapsed-text exclusion",
                "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                "heldout_labels_used": False}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
