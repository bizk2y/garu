import base64
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from export import emit_typescript


class ExportTests(unittest.TestCase):
    def test_compiled_export(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            weights = bytes([1, 255, 2, 254]) * (1 << 18)
            metadata = {
                'dimensions': len(weights) // 2,
                'features': [[-1, 1], [-1, 2], [0, 3], [-3, 4]],
                'heads': {'spaces': {'scale': 0.1, 'threshold': -0.5,
                                     'conservative_threshold': 1.0, 'order': 0}},
                'weights_sha256': hashlib.sha256(weights).hexdigest(),
            }
            (folder / 'export.json').write_text(json.dumps(metadata))
            (folder / 'weights.b64').write_text(base64.b64encode(weights).decode())
            emit_typescript(folder, folder / 'model.ts')
            repo = Path(__file__).resolve().parents[2]
            compiler = next(path for path in [repo / 'node_modules/typescript/bin/tsc',
                                               repo / 'js/node_modules/typescript/bin/tsc']
                            if path.exists())
            compiled = subprocess.run(['node', str(compiler), str(folder / 'model.ts'),
                                       '--target', 'ES2020', '--module', 'commonjs',
                                       '--declaration', '--skipLibCheck', '--outDir', str(folder / 'out')],
                                      capture_output=True, text=True, cwd=folder)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            script = """const m = require(process.argv[1]);
const hash = require('node:crypto').createHash('sha256');
hash.update(Buffer.from(m.BOUNDARY_WEIGHTS, 'base64'));
const values = new Int8Array(Buffer.from(m.BOUNDARY_WEIGHTS, 'base64'));
const points = [65, 128512, 44032, 66];
const scores = [1, 2, 3].map(i => m.scoreBoundary(points, i, values, 3));
console.log(JSON.stringify({head: m.BOUNDARY_MODEL.heads.spaces, sha256: hash.digest('hex'), scores}));"""
            output = subprocess.check_output(['node', '-e', script,
                                               str(folder / 'out/model.js')], text=True)
            actual = json.loads(output)
            self.assertEqual(actual['sha256'], metadata['weights_sha256'])
            self.assertEqual(actual['head']['threshold'], -0.5)
            self.assertEqual(actual['head']['conservative_threshold'], 1.0)
            # Independent scalar oracle covers head offsets, Unicode, padding and widths 1..4.
            values = [w if w < 128 else w - 256 for w in weights]
            points = [65, 128512, 44032, 66]
            expected = []
            for position in [1, 2, 3]:
                score = values[3]
                for index, (start, width) in enumerate(metadata['features'], 1):
                    hashed = 2166136261 ^ index
                    for delta in range(width):
                        at = position + start + delta
                        point = points[at] if 0 <= at < len(points) else 0
                        hashed = ((hashed ^ point) * 16777619) & 0xffffffff
                    score += values[3 + 1 + hashed % (metadata['dimensions'] - 1)]
                expected.append(score)
            self.assertEqual(actual['scores'], expected)
            self.assertLess((folder / 'out/model.d.ts').stat().st_size, 4096)

    def test_changed_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'export.json').write_text(json.dumps({'weights_sha256': '0' * 64}))
            (folder / 'weights.b64').write_text(base64.b64encode(b'changed').decode())
            with self.assertRaises(AssertionError):
                emit_typescript(folder, folder / 'model.ts')


if __name__ == '__main__':
    unittest.main()
