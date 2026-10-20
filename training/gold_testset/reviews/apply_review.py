"""Apply the reviewed POS changes to a separate benchmark copy."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    review = json.loads((HERE / 'corrections.json').read_text())
    source = Path(review['source'])
    original = source.read_bytes()
    assert hashlib.sha256(original).hexdigest() == review['source_sha256']
    records = [json.loads(line) for line in original.decode().splitlines()]
    for item in review['patches']:
        record = records[item['gold_line'] - 1]
        assert record['text'] == item['text']
        assert record['morphemes'][item['token_index']] == item['before']
        record['morphemes'][item['token_index']] = item['after']
    target = HERE / 'gold_corrected.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    assert source.read_bytes() == original
    print(f'{len(review["patches"])} corrections written to {target}; source unchanged')


if __name__ == '__main__':
    main()
