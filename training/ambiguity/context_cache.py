"""Small context-gated alternative cache, GMDL optional Section 16."""
import struct

import brotli

POS = "NNG NNP NNB NR NP VV VA VX VCP VCN MAG MAJ MM IC JKS JKC JKG JKO JKB JKV JKQ JX JC EP EF EC ETN ETM XPN XSN XSV XSA XR SF SP SS SE SO SW SH SL SN".split()
MAGIC = b"GCAC\x01"


def contexts(previous, following):
    out = [(0, "")] if previous is None else []
    if previous is not None:
        out += [(n, previous[-n:]) for n in range(1, 4)]
        out += [(9 + n, previous[:n]) for n in range(1, 4)]
    if following is None:
        return out + [(13, "")]
    out += [(3 + n, following[:n]) for n in range(1, 4)]
    return out + [(6 + n, following[-n:]) for n in range(1, 4)]


def collapse(text):
    return "".join(text.split())


def split_for(row, dev_docs, dev_keys, excluded):
    key = collapse(row["text"])
    if key in excluded:
        return None
    if row["doc"] in dev_docs:
        return "dev"
    return None if key in dev_keys else "train"


def sections(model):
    if not model.startswith(b"GMDL"):
        model = brotli.decompress(model)
    if model[:8] != b"GMDL" + struct.pack("<I", 3):
        raise ValueError("Expected GMDL v3")
    out, offset = {}, 8
    while offset < len(model):
        kind, size = struct.unpack_from("<BI", model, offset)
        offset += 5
        if offset + size > len(model):
            raise ValueError("Truncated GMDL section")
        out[kind] = model[offset:offset + size]
        offset += size
    return out


def patch_model(model, rules, cost):
    old = sections(model)
    old[16] = encode(rules, cost)
    raw = b"GMDL" + struct.pack("<I", 3)
    return raw + b"".join(struct.pack("<BI", kind, len(blob)) + blob for kind, blob in old.items())


def encode(rules, cost):
    def string(value):
        data = value.encode("utf-8")
        return struct.pack("<H", len(data)) + data

    out = bytearray(MAGIC + struct.pack("<fI", cost, len(rules)))
    for rule in rules:
        out += string(rule["surface"]) + struct.pack("<B", len(rule["morphs"]))
        for form, pos in rule["morphs"]:
            out += string(form) + struct.pack("<B", POS.index(pos))
        out += struct.pack("<H", len(rule["guards"]))
        for kind, value in rule["guards"]:
            out += struct.pack("<B", kind) + string(value)
    return bytes(out)


def decode(blob):
    offset = 0

    def take(size):
        nonlocal offset
        if offset + size > len(blob):
            raise ValueError("Truncated context cache")
        value = blob[offset:offset + size]
        offset += size
        return value

    def number(fmt):
        return struct.unpack(fmt, take(struct.calcsize(fmt)))[0]

    def string():
        return take(number("<H")).decode("utf-8")

    if take(5) != MAGIC:
        raise ValueError("Unsupported context cache")
    cost, count = number("<f"), number("<I")
    rules = []
    for _ in range(count):
        surface = string()
        morphs = [[string(), POS[number("<B")]] for _ in range(number("<B"))]
        guards = [[number("<B"), string()] for _ in range(number("<H"))]
        rules.append({"surface": surface, "morphs": morphs, "guards": guards})
    if offset != len(blob):
        raise ValueError("Trailing context cache bytes")
    return cost, rules
