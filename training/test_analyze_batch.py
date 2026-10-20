"""Exercise the actual analyze_batch CLI and its line-oriented output formats."""
import json
import os
from pathlib import Path
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def binary():
    subprocess.run(["cargo", "build", "--quiet", "--example", "analyze_batch"],
                   cwd=REPO, check=True)
    metadata = subprocess.run(["cargo", "metadata", "--format-version", "1", "--no-deps"],
                              cwd=REPO, capture_output=True, text=True, check=True)
    target_dir = Path(json.loads(metadata.stdout)["target_directory"])
    return target_dir / "debug/examples/analyze_batch"


def run_batch(binary, tmp_path, text, *args):
    input_path = tmp_path / "input.txt"
    input_path.write_text(text, encoding="utf-8")
    result = subprocess.run(
        [str(binary), str(input_path), *args], cwd=REPO,
        env={**os.environ, "GARU_MODEL": str(REPO / "js/models/base.gmdl")},
        capture_output=True, text=True, check=True,
    )
    return result.stdout


@pytest.mark.parametrize("text", ['\\', '"', '한글', 'C:\\temp\\data', '\x00'])
def test_json_roundtrip(binary, tmp_path, text):
    output = run_batch(binary, tmp_path, text + "\n", "--json")
    lines = output.splitlines()
    assert len(lines) == 1
    tokens = json.loads(lines[0])
    assert "".join(token[0] for token in tokens) == text
    assert all(len(token) == 2 and isinstance(token[1], str) for token in tokens)


@pytest.mark.parametrize("args", [(), ("--json",)])
def test_empty_lines(binary, tmp_path, args):
    assert run_batch(binary, tmp_path, "", *args) == ""
    assert run_batch(binary, tmp_path, "\n \t\n", *args) == "[]\n[]\n"


def test_tsv_output(binary, tmp_path):
    assert run_batch(binary, tmp_path, "한글\n\n") == "한글\tNNG\n---\n[]\n"
