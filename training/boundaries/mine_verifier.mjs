// Extract features through the same compiled JS/WASM path used by the optional API.
import { createInterface } from 'node:readline';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';

const runtime = resolve(process.argv[2]);
const { Garu } = await import(pathToFileURL(join(runtime, 'node.js')).href);
const { sentenceCandidates } = await import(pathToFileURL(join(runtime, 'boundaries.js')).href);
const { extractSentenceFeatures, extractContextFeatures, FEATURE_NAMES } = await import(pathToFileURL(join(runtime, 'sentence-verifier.js')).href);
const analyzer = process.argv[3] === 'context' ? undefined : await Garu.load();
process.stderr.write(JSON.stringify({ featureNames: FEATURE_NAMES }) + '\n');
try {
  for await (const line of createInterface({ input: process.stdin, crlfDelay: Infinity })) {
    if (!line.trim()) continue;
    const { id, text, extraCandidates = [] } = JSON.parse(line);
    const candidates = sentenceCandidates(text);
    const allCandidates = [...candidates, ...extraCandidates];
    const features = analyzer
      ? extractSentenceFeatures(text, allCandidates, analyzer)
      : extractContextFeatures(text, allCandidates);
    const result = { id, candidates, features: features.slice(0, candidates.length) };
    if (extraCandidates.length) {
      result.extraCandidates = extraCandidates;
      result.extraFeatures = features.slice(candidates.length);
    }
    if (!process.stdout.write(JSON.stringify(result) + '\n')) {
      await new Promise(resolveDrain => process.stdout.once('drain', resolveDrain));
    }
  }
} finally {
  analyzer?.destroy();
}
