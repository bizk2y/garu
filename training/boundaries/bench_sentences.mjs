// Loaded latency and boundary F1 on paired output from eval_natural.load_natural.
// node bench_sentences.mjs js/dist natural.json report.json [frozen-baseline/dist]
import fs from 'node:fs';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';

const [runtimeArg, inputFile, outputFile, baselineArg] = process.argv.slice(2);
if (!outputFile) throw new Error('Expected runtime, natural JSON, report JSON, optional baseline runtime');
const SAMPLE_COUNT = 1250;
const PARAGRAPH_STRIDE = 7;
const WARMUP_COUNT = 50;
const ROUNDS = 7;
const rows = JSON.parse(fs.readFileSync(inputFile, 'utf8'));
if (rows.length % 2 || rows.some((r, i) => r.variant !== (i % 2 ? 'no_terminal' : 'punctuated'))) {
  throw new Error('Expected alternating punctuated/no_terminal rows');
}
const inputs = [];
for (let i = 0; i < rows.length / 2 && inputs.length < SAMPLE_COUNT; i += PARAGRAPH_STRIDE) {
  inputs.push(rows[2 * i + (inputs.length % 2)].text);
}
const load = (root, file) => import(pathToFileURL(join(resolve(root), file)).href);
let start = performance.now();
const { splitSentences } = await load(runtimeArg, 'sentences.js');
const importMs = performance.now() - start;
start = performance.now();
splitSentences(inputs[0]);
const firstCallMs = performance.now() - start;
const paths = { fast: text => splitSentences(text) };
let baselineAnalyzer;
if (baselineArg) {
  const { Garu } = await load(baselineArg, 'node.js');
  baselineAnalyzer = await Garu.load();
  const previous = await load(baselineArg, 'sentences.js');
  const { sentenceCandidates, splitAtBoundaries } = await load(baselineArg, 'boundaries.js');
  const { keepOutsideQuotes } = await load(baselineArg, 'sentence-verifier.js');
  paths.previousFull = text => previous.splitSentences(text, baselineAnalyzer);
  paths.previousQuote = text => splitAtBoundaries(text,
    keepOutsideQuotes(text, sentenceCandidates(text)).map(c => c.offset));
}
const timings = Object.fromEntries(Object.keys(paths).map(name => [name, []]));
let checksum = 0;
try {
  for (const fn of Object.values(paths)) {
    for (const text of inputs.slice(0, WARMUP_COUNT)) checksum += fn(text).length;
  }
  const entries = Object.entries(paths);
  for (let round = 0; round < ROUNDS; round++) {
    for (let k = 0; k < entries.length; k++) {
      const [name, fn] = entries[(round + k) % entries.length];
      start = performance.now();
      for (const text of inputs) checksum += fn(text).length;
      timings[name].push((performance.now() - start) * 1000 / inputs.length);
    }
  }
} finally {
  baselineAnalyzer?.destroy();
}
const latencies = [];
for (const text of inputs) {
  start = performance.now();
  checksum += splitSentences(text).length;
  latencies.push((performance.now() - start) * 1000);
}
const metrics = {};
const predictions = createHash('sha256');
for (const row of rows) {
  const segments = splitSentences(row.text);
  const chars = [...row.text];
  if (segments.some(s => chars.slice(s.offset, s.offset + [...s.text].length).join('') !== s.text)
      || segments.map(s => s.text).join('') !== row.text) throw new Error('Original span invariant failed');
  const boundaries = segments.slice(1).map(s => s.offset);
  predictions.update(JSON.stringify(boundaries) + '\n');
  const gold = new Set(row.sentences);
  const tp = boundaries.filter(at => gold.has(at)).length;
  const count = metrics[row.variant] ??= { rows: 0, tp: 0, fp: 0, fn: 0 };
  count.rows++;
  count.tp += tp;
  count.fp += boundaries.length - tp;
  count.fn += gold.size - tp;
}
for (const count of Object.values(metrics)) {
  count.f1 = 2 * count.tp / (2 * count.tp + count.fp + count.fn);
}
const quantile = (values, q) => [...values].sort((a, b) => a - b)[Math.floor((values.length - 1) * q)];
const report = {
  environment: { node: process.version, platform: process.platform, arch: process.arch, cpu: os.cpus()[0].model },
  inputSha256: createHash('sha256').update(fs.readFileSync(inputFile)).digest('hex'),
  sample: { count: inputs.length, stride: PARAGRAPH_STRIDE, warmup: WARMUP_COUNT, rounds: ROUNDS,
    meanCodepoints: inputs.reduce((n, text) => n + [...text].length, 0) / inputs.length },
  cold: { importMs, firstCallMs, includesFullAndFastModuleParsing: true, osFileCacheControlled: false },
  loaded: Object.fromEntries(Object.entries(timings).map(([name, us]) =>
    [name, { roundMeanUs: us, medianMeanUs: quantile(us, 0.5) }])),
  fastIndividualUs: { p50: quantile(latencies, 0.5), p95: quantile(latencies, 0.95), max: Math.max(...latencies) },
  metrics, predictionsSha256: predictions.digest('hex'), checksum,
};
fs.writeFileSync(outputFile, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
