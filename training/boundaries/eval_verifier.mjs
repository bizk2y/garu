// Frozen end-to-end splitter comparison. No training or threshold adjustment.
import fs from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';

const [runtimeArg, modelDir] = process.argv.slice(2);
const runtime = resolve(runtimeArg);
const { Garu } = await import(pathToFileURL(join(runtime, 'node.js')).href);
const { splitUnspaced, sentenceCandidates, splitAtBoundaries } = await import(pathToFileURL(join(runtime, 'boundaries.js')).href);
const { loadSentenceVerifier, splitVerified, keepOutsideQuotes } = await import(pathToFileURL(join(runtime, 'sentence-verifier.js')).href);
const models = Object.fromEntries(['full', 'structured', 'no_morph', 'easy'].map(name => [
  name, loadSentenceVerifier(JSON.parse(fs.readFileSync(join(modelDir, name + '.json'), 'utf8'))),
]));
const analyzer = await Garu.load();
try {
  for await (const line of createInterface({ input: process.stdin, crlfDelay: Infinity })) {
    if (!line.trim()) continue;
    const { id, text } = JSON.parse(line);
    const outputs = { baseline: splitUnspaced(text) };
    outputs.baseline_quote = splitAtBoundaries(text, keepOutsideQuotes(text, sentenceCandidates(text)).map(c => c.offset));
    for (const [name, model] of Object.entries(models)) outputs[name] = splitVerified(text, model, analyzer);
    const chars = [...text];
    for (const segments of Object.values(outputs)) {
      if (segments.some(s => chars.slice(s.offset, s.offset + [...s.text].length).join('') !== s.text)
          || segments.map(s => s.text).join('') !== text) throw new Error('Original substring invariant failed');
    }
    const boundaries = Object.fromEntries(Object.entries(outputs).map(([name, segments]) => [name, segments.slice(1).map(s => s.offset)]));
    if (!process.stdout.write(JSON.stringify({ id, boundaries }) + '\n')) {
      await new Promise(resolveDrain => process.stdout.once('drain', resolveDrain));
    }
  }
} finally {
  analyzer.destroy();
}
