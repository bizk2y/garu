import { BOUNDARY_MODEL, BOUNDARY_WEIGHTS, scoreBoundary } from './boundary-data.js';
import type { Segment } from './normalize.js';

const SPACE = /\s/u;
const HANGUL = /[가-힣]/u;
const ASCII_RUN = /[A-Za-z0-9._~:/?#@!$&'()*+,;=%+-]+/gu;
const DELIMITED_URL = /(?:^|\s)((?:https?:\/\/|www\.)[^\s]+)/gu;
const TERMINALS = '.!?…';
const QUOTES = '"\'“”‘’()[]{}';
let weights: Int8Array | undefined;

function loadWeights(): Int8Array {
  if (weights) return weights;
  const encoded = atob(BOUNDARY_WEIGHTS);
  weights = new Int8Array(encoded.length);
  for (let i = 0; i < encoded.length; i++) weights[i] = encoded.charCodeAt(i);
  return weights;
}

function charKind(value: number): number {
  if (value === 0) return 0;
  if (value >= 0xac00 && value <= 0xd7a3) return 1;
  if (value >= 65 && value <= 90) return 2;
  if (value >= 97 && value <= 122) return 3;
  if (value >= 48 && value <= 57) return 4;
  if (TERMINALS.includes(String.fromCodePoint(value))) return 5;
  if (QUOTES.includes(String.fromCodePoint(value))) return 6;
  return 7;
}

function protectedRanges(text: string): number[][] {
  const ranges: number[][] = [];
  // A whitespace-delimited URL can include Unicode. Without that delimiter,
  // protect its ASCII portion so a following Korean sentence is still visible.
  for (const match of text.matchAll(DELIMITED_URL)) {
    const start = match.index! + match[0].length - match[1].length;
    ranges.push([start, start + match[1].length]);
  }
  for (const match of text.matchAll(ASCII_RUN)) {
    const run = match[0];
    const start = match.index!;
    if (run.includes('://') || run.startsWith('www.') || run.includes('@')
        || /^[0-9.,:/_-]+$/u.test(run)) {
      ranges.push([start, start + run.length]);
      continue;
    }
    for (const word of run.matchAll(/[A-Za-z0-9]+/gu)) {
      ranges.push([start + word.index!, start + word.index! + word[0].length]);
    }
  }
  return ranges.sort((a, b) => a[0] - b[0] || b[1] - a[1]);
}

export interface SentenceCandidate {
  offset: number;
  compactOffset: number;
  score: number;
}

function predictBoundaries(
  text: string, head: 'spaces' | 'sentences', threshold?: number,
): SentenceCandidate[] {
  if (!HANGUL.test(text)) return [];
  const points: number[] = [];
  const sourceOffsets: number[] = [];
  const utf16Offsets: number[] = [];
  let sourceOffset = 0;
  let utf16Offset = 0;
  for (const char of text) {
    if (!SPACE.test(char)) {
      points.push(char.codePointAt(0)!);
      sourceOffsets.push(sourceOffset);
      utf16Offsets.push(utf16Offset);
    }
    sourceOffset++;
    utf16Offset += char.length;
  }

  const ranges = protectedRanges(text);
  const model = BOUNDARY_MODEL.heads[head];
  const dimensions = BOUNDARY_MODEL.dimensions;
  const offset = model.order * dimensions;
  const values = loadWeights();
  const boundaries: SentenceCandidate[] = [];
  let rangeIndex = 0;
  for (let i = 1; i < points.length; i++) {
    const position = utf16Offsets[i];
    while (rangeIndex < ranges.length && ranges[rangeIndex][1] <= position) {
      rangeIndex++;
    }
    const range = ranges[rangeIndex];
    if (range && range[0] < position && position < range[1]) continue;

    let score = scoreBoundary(points, i, values, offset);
    const category = 1 + (charKind(points[i - 1]) * 17 + charKind(points[i]) * 103 + 147401)
      % (dimensions - 1);
    score += values[offset + category];
    const scaled = score * model.scale;
    if (scaled > (threshold ?? model.threshold)) {
      boundaries.push({ offset: sourceOffsets[i], compactOffset: i, score: scaled });
    }
  }
  return boundaries;
}

export interface SpacingOptions {
  /** Conservative inserts fewer uncertain spaces; balanced recovers more. */
  mode?: 'conservative' | 'balanced';
}

/** Internal candidates with the same threshold and protection gates as splitUnspaced. */
export function sentenceCandidates(text: string): SentenceCandidate[] {
  return predictBoundaries(text, 'sentences');
}

/** Infer missing Korean spaces while retaining original characters and whitespace. */
export function restoreSpacing(text: string, opts: SpacingOptions = {}): string {
  const threshold = opts.mode === 'balanced'
    ? BOUNDARY_MODEL.heads.spaces.threshold
    : BOUNDARY_MODEL.heads.spaces.conservative_threshold;
  const boundaries = new Set(predictBoundaries(text, 'spaces', threshold).map(value => value.offset));
  if (boundaries.size === 0) return text;
  const chars = [...text];
  return chars.map((char, i) => (
    boundaries.has(i) && i > 0 && !SPACE.test(chars[i - 1]) ? ' ' + char : char
  )).join('');
}

/** Infer Korean sentence boundaries, retaining original text and codepoint offsets. */
export function splitUnspaced(text: string): Segment[] {
  const boundaries = sentenceCandidates(text).map(value => value.offset);
  return splitAtBoundaries(text, boundaries);
}

/** Internal splitter for sorted original-codepoint boundaries. */
export function splitAtBoundaries(text: string, boundaries: number[]): Segment[] {
  const chars = [...text];
  const segments: Segment[] = [];
  let cursor = 0;
  for (const boundary of [...boundaries, chars.length]) {
    let start = cursor;
    let end = boundary;
    while (start < end && SPACE.test(chars[start])) start++;
    while (end > start && SPACE.test(chars[end - 1])) end--;
    if (start < end) segments.push({ text: chars.slice(start, end).join(''), offset: start });
    cursor = boundary;
  }
  return segments.length ? segments : [{ text: text.trim(), offset: 0 }];
}
