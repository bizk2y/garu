/**
 * Optional English analysis. Import from `garu-ko/english`.
 *
 * Ports NLTK's `NLTKWordTokenizer` (the tokenizer behind `word_tokenize`) and
 * `PorterStemmer` in its default NLTK_EXTENSIONS mode. No data files are needed.
 */
import type { AnalyzeResult, POS, Token } from './core.js';

export interface EnglishTokenizeOptions {
  /** Skip sentence splitting, matching NLTK `word_tokenize(text, preserve_line=True)`. */
  preserveLine?: boolean;
}

export interface EnglishToken {
  /** Original substring; NLTK's `` and '' quote tokens map back to the source quote. */
  text: string;
  pos: POS;
  /** Porter stem for SL tokens; other tokens keep their surface. */
  stem: string;
  /** Unicode code point offsets in the original input. */
  start: number;
  end: number;
}

type Rule = [RegExp, string];

const W = '[\\p{L}\\p{N}_]';
const re = (source: string, flags = 'gu') => new RegExp(source, flags);

const STARTING_QUOTES: Rule[] = [
  [/([«“‘„]|`+)/gu, ' $1 '],
  [/^"/u, '``'],
  [/(``)/gu, ' $1 '],
  [/([ ([{<])("|'{2})/gu, '$1 `` '],
  [re(`(')(?!re|ve|ll|m|t|s|d|n)(${W})(?!${W})`, 'giu'), '$1 $2'],
];

const PUNCTUATION: Rule[] = [
  [/([^.])(\.)([\])}>"'»”’ ]*)\s*$/u, '$1 $2 $3 '],
  [/([:,])([^\p{Nd}])/gu, ' $1 $2'],
  // Python's `$` also matches before a final newline.
  [/([:,])(?=\n?$)/u, ' $1 '],
  [/\.{2,}/gu, ' $& '],
  [/[;@#$%&]/gu, ' $& '],
  [/([^.])(\.)([\])}>"']*)\s*$/u, '$1 $2$3 '],
  [/[?!]/gu, ' $& '],
  [/([^'])' /gu, "$1 ' "],
  [/[*]/gu, ' $& '],
];

const PARENS_BRACKETS: Rule = [/[\][(){}<>]/gu, ' $& '];
const DOUBLE_DASHES: Rule = [/--/gu, ' -- '];

const ENDING_QUOTES: Rule[] = [
  [/([»”’])/gu, ' $1 '],
  [/''/gu, " '' "],
  [/"/gu, " '' "],
  [/\s+/gu, ' '],
  [/([^' ])('[sS]|'[mM]|'[dD]|') /gu, '$1 $2 '],
  [/([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) /gu, '$1 $2 '],
];

const CONTRACTIONS: RegExp[] = [
  ...[
    ['can', 'not'],
    ['d', "'ye"],
    ['gim', 'me'],
    ['gon', 'na'],
    ['got', 'ta'],
    ['lem', 'me'],
    ['more', "'n"],
  ].map(([a, b]) => re(`(?<!${W})(${a})(${b})(?!${W})`, 'giu')),
  re(`(?<!${W})(wan)(na)(?=\\s)`, 'giu'),
  re(` ('t)(is)(?!${W})`, 'giu'),
  re(` ('t)(was)(?!${W})`, 'giu'),
];

function applyRules(text: string, rules: Rule[]): string {
  for (const [pattern, replacement] of rules) {
    text = text.replace(pattern, replacement);
  }
  return text;
}

function tokenizeSentence(text: string): string[] {
  text = applyRules(text, STARTING_QUOTES);
  text = applyRules(text, PUNCTUATION);
  text = applyRules(text, [PARENS_BRACKETS, DOUBLE_DASHES]);
  text = applyRules(` ${text} `, ENDING_QUOTES);
  for (const pattern of CONTRACTIONS) {
    text = text.replace(pattern, ' $1 $2 ');
  }
  return text.split(/\s+/u).filter(Boolean);
}

const ABBREVIATIONS = new Set([
  'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'mt', 'vs', 'etc', 'inc', 'ltd', 'co',
  'corp', 'no', 'fig', 'approx', 'dept', 'gen', 'col', 'lt', 'sgt', 'capt', 'rev',
  'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
]);

/** Heuristic stand-in for NLTK's Punkt model: splits before a capitalized word. */
function splitEnglishSentences(text: string): string[] {
  const sentences: string[] = [];
  const boundary = /([.?!]+)(["'”’)\]]*)\s+(?=["'“‘([]*\p{Lu})/gu;
  let start = 0;
  for (const m of text.matchAll(boundary)) {
    const index = m.index!;
    if (m[1] === '.') {
      const word = text.slice(0, index).split(/\s/u).pop()!.replace(/^[^\p{L}]+/u, '').toLowerCase();
      if (ABBREVIATIONS.has(word) || /^\p{L}$/u.test(word) || /^(\p{L}\.)+\p{L}$/u.test(word)) {
        continue;
      }
    }
    const end = index + m[1].length + m[2].length;
    sentences.push(text.slice(start, end).trim());
    start = index + m[0].length;
  }
  const rest = text.slice(start).trim();
  if (rest) sentences.push(rest);
  return sentences;
}

/** NLTK-compatible `word_tokenize`. Sentence splitting approximates Punkt. */
export function tokenizeEnglish(text: string, options?: EnglishTokenizeOptions): string[] {
  const sentences = options?.preserveLine ? [text] : splitEnglishSentences(text);
  return sentences.flatMap(tokenizeSentence);
}

// ---------------------------------------------------------------------------
// Porter stemmer (NLTK_EXTENSIONS mode)
// ---------------------------------------------------------------------------

const IRREGULAR_FORMS: Record<string, string[]> = {
  sky: ['sky', 'skies'],
  die: ['dying'],
  lie: ['lying'],
  tie: ['tying'],
  news: ['news'],
  inning: ['innings', 'inning'],
  outing: ['outings', 'outing'],
  canning: ['cannings', 'canning'],
  howe: ['howe'],
  proceed: ['proceed'],
  exceed: ['exceed'],
  succeed: ['succeed'],
};

const POOL = new Map<string, string>(
  Object.entries(IRREGULAR_FORMS).flatMap(([key, forms]) => forms.map((f): [string, string] => [f, key])),
);

type Condition = ((stem: string) => boolean) | null;
type StemRule = [suffix: string, replacement: string, condition: Condition];

function isConsonant(word: string, i: number): boolean {
  const c = word[i];
  if (c === 'a' || c === 'e' || c === 'i' || c === 'o' || c === 'u') return false;
  if (c === 'y') return i === 0 ? true : !isConsonant(word, i - 1);
  return true;
}

function measure(stem: string): number {
  let count = 0;
  for (let i = 1; i < stem.length; i++) {
    if (!isConsonant(stem, i - 1) && isConsonant(stem, i)) count++;
  }
  return count;
}

const positiveMeasure = (stem: string) => measure(stem) > 0;
const measureGt1 = (stem: string) => measure(stem) > 1;

function containsVowel(stem: string): boolean {
  for (let i = 0; i < stem.length; i++) {
    if (!isConsonant(stem, i)) return true;
  }
  return false;
}

function endsDoubleConsonant(word: string): boolean {
  const n = word.length;
  return n >= 2 && word[n - 1] === word[n - 2] && isConsonant(word, n - 1);
}

function endsCvc(word: string): boolean {
  const n = word.length;
  return (
    (n >= 3 &&
      isConsonant(word, n - 3) &&
      !isConsonant(word, n - 2) &&
      isConsonant(word, n - 1) &&
      !'wxy'.includes(word[n - 1])) ||
    (n === 2 && !isConsonant(word, 0) && isConsonant(word, 1))
  );
}

function applyRuleList(word: string, rules: StemRule[]): string {
  for (const [suffix, replacement, condition] of rules) {
    if (suffix === '*d' && endsDoubleConsonant(word)) {
      const stem = word.slice(0, -2);
      return condition === null || condition(stem) ? stem + replacement : word;
    }
    if (word.endsWith(suffix)) {
      const stem = word.slice(0, word.length - suffix.length);
      return condition === null || condition(stem) ? stem + replacement : word;
    }
  }
  return word;
}

function step1a(word: string): string {
  if (word.endsWith('ies') && word.length === 4) return word.slice(0, -3) + 'ie';
  return applyRuleList(word, [
    ['sses', 'ss', null],
    ['ies', 'i', null],
    ['ss', 'ss', null],
    ['s', '', null],
  ]);
}

function step1b(word: string): string {
  if (word.endsWith('ied')) {
    return word.slice(0, -3) + (word.length === 4 ? 'ie' : 'i');
  }
  if (word.endsWith('eed')) {
    const stem = word.slice(0, -3);
    return measure(stem) > 0 ? stem + 'ee' : word;
  }
  let intermediate: string | undefined;
  for (const suffix of ['ed', 'ing']) {
    if (word.endsWith(suffix)) {
      const candidate = word.slice(0, -suffix.length);
      if (containsVowel(candidate)) {
        intermediate = candidate;
        break;
      }
    }
  }
  if (intermediate === undefined) return word;
  const last = intermediate[intermediate.length - 1];
  return applyRuleList(intermediate, [
    ['at', 'ate', null],
    ['bl', 'ble', null],
    ['iz', 'ize', null],
    ['*d', last, () => !'lsz'.includes(last)],
    ['', 'e', (stem) => measure(stem) === 1 && endsCvc(stem)],
  ]);
}

function step1c(word: string): string {
  return applyRuleList(word, [
    ['y', 'i', (stem) => stem.length > 1 && isConsonant(stem, stem.length - 1)],
  ]);
}

function step2(word: string): string {
  if (word.endsWith('alli') && positiveMeasure(word.slice(0, -4))) {
    return step2(word.slice(0, -4) + 'al');
  }
  return applyRuleList(word, [
    ['ational', 'ate', positiveMeasure],
    ['tional', 'tion', positiveMeasure],
    ['enci', 'ence', positiveMeasure],
    ['anci', 'ance', positiveMeasure],
    ['izer', 'ize', positiveMeasure],
    ['bli', 'ble', positiveMeasure],
    ['alli', 'al', positiveMeasure],
    ['entli', 'ent', positiveMeasure],
    ['eli', 'e', positiveMeasure],
    ['ousli', 'ous', positiveMeasure],
    ['ization', 'ize', positiveMeasure],
    ['ation', 'ate', positiveMeasure],
    ['ator', 'ate', positiveMeasure],
    ['alism', 'al', positiveMeasure],
    ['iveness', 'ive', positiveMeasure],
    ['fulness', 'ful', positiveMeasure],
    ['ousness', 'ous', positiveMeasure],
    ['aliti', 'al', positiveMeasure],
    ['iviti', 'ive', positiveMeasure],
    ['biliti', 'ble', positiveMeasure],
    ['fulli', 'ful', positiveMeasure],
    // NLTK keeps the 'l' with the stem so short stems like 'geo' still qualify.
    ['logi', 'log', () => positiveMeasure(word.slice(0, -3))],
  ]);
}

function step3(word: string): string {
  return applyRuleList(word, [
    ['icate', 'ic', positiveMeasure],
    ['ative', '', positiveMeasure],
    ['alize', 'al', positiveMeasure],
    ['iciti', 'ic', positiveMeasure],
    ['ical', 'ic', positiveMeasure],
    ['ful', '', positiveMeasure],
    ['ness', '', positiveMeasure],
  ]);
}

function step4(word: string): string {
  return applyRuleList(word, [
    ['al', '', measureGt1],
    ['ance', '', measureGt1],
    ['ence', '', measureGt1],
    ['er', '', measureGt1],
    ['ic', '', measureGt1],
    ['able', '', measureGt1],
    ['ible', '', measureGt1],
    ['ant', '', measureGt1],
    ['ement', '', measureGt1],
    ['ment', '', measureGt1],
    ['ent', '', measureGt1],
    ['ion', '', (stem) => measure(stem) > 1 && 'st'.includes(stem[stem.length - 1])],
    ['ou', '', measureGt1],
    ['ism', '', measureGt1],
    ['ate', '', measureGt1],
    ['iti', '', measureGt1],
    ['ous', '', measureGt1],
    ['ive', '', measureGt1],
    ['ize', '', measureGt1],
  ]);
}

function step5a(word: string): string {
  if (word.endsWith('e')) {
    const stem = word.slice(0, -1);
    const m = measure(stem);
    if (m > 1 || (m === 1 && !endsCvc(stem))) return stem;
  }
  return word;
}

function step5b(word: string): string {
  return applyRuleList(word, [['ll', 'l', () => measure(word.slice(0, -1)) > 1]]);
}

/** NLTK `PorterStemmer().stem(word)` in the default NLTK_EXTENSIONS mode. */
export function stemEnglish(word: string): string {
  const lower = word.toLowerCase();
  // NLTK checks the irregular pool with the original casing.
  const irregular = POOL.get(word);
  if (irregular !== undefined) return irregular;
  if (word.length <= 2) return lower;
  let stem = step1a(lower);
  stem = step1b(stem);
  stem = step1c(stem);
  stem = step2(stem);
  stem = step3(stem);
  stem = step4(stem);
  stem = step5a(stem);
  return step5b(stem);
}

// ---------------------------------------------------------------------------
// Analysis with Garu's Sejong symbol tags
// ---------------------------------------------------------------------------

const QUOTE_TOKENS = new Set(['``', "''", '"']);
const BRACKETS_AND_QUOTES = new Set('()[]{}<>"\'`«»“”‘’„―—「」『』〈〉《》');

function tagToken(token: string): POS {
  if (/\p{L}/u.test(token)) return 'SL';
  if (/^\p{N}[\p{N}.,:/-]*$/u.test(token)) return 'SN';
  if (/^(\.{2,}|…+)$/u.test(token)) return 'SE';
  if (QUOTE_TOKENS.has(token) || [...token].every((c) => BRACKETS_AND_QUOTES.has(c))) return 'SS';
  if (/^(-+|~|–|−)$/u.test(token)) return 'SO';
  if (/^[.!?]$/u.test(token)) return 'SF';
  if (/^[,;:/·]$/u.test(token)) return 'SP';
  return 'SW';
}

/**
 * Tokenize English text the NLTK way and attach Sejong symbol tags, Porter
 * stems and code point offsets. English words are tagged SL, as in `Garu.analyze`.
 */
export function analyzeEnglish(text: string, options?: EnglishTokenizeOptions): EnglishToken[] {
  const tokens = tokenizeEnglish(text, options);
  const result: EnglishToken[] = [];
  let cursor = 0;
  let cursorCp = 0;
  for (const token of tokens) {
    let index: number;
    let surface = token;
    if (QUOTE_TOKENS.has(token)) {
      const quote = /``|''|"/gu;
      quote.lastIndex = cursor;
      const m = quote.exec(text);
      index = m ? m.index : -1;
      if (m) surface = m[0];
    } else {
      index = text.indexOf(token, cursor);
    }
    if (index < 0) continue;
    const start = cursorCp + [...text.slice(cursor, index)].length;
    const end = start + [...surface].length;
    const pos = tagToken(token);
    result.push({ text: surface, pos, stem: pos === 'SL' ? stemEnglish(token) : surface, start, end });
    cursor = index + surface.length;
    cursorCp = end;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Mixed Korean/English analysis
// ---------------------------------------------------------------------------

export interface KoreanAnalyzer {
  analyze(text: string): AnalyzeResult | AnalyzeResult[];
}

export type Language = 'ko' | 'en';

export interface MixedToken extends Token {
  lang: Language;
  /** Present on English tokens: the Porter stem for SL, otherwise the surface. */
  stem?: string;
}

interface Run {
  lang: Language;
  /** UTF-16 range, for slicing. */
  from: number;
  to: number;
  /** Code point offset of `from`. */
  cp: number;
}

const HANGUL = /[ᄀ-ᇿ㄰-㆏가-힣]/u;
const LATIN = /\p{Script=Latin}/u;

function splitRuns(text: string): Run[] {
  const runs: Run[] = [];
  let pendingNeutral: { from: number; to: number; cp: number } | undefined;
  let utf16 = 0;
  let cp = 0;
  for (const m of text.matchAll(/\S+/gu)) {
    const from = m.index!;
    cp += [...text.slice(utf16, from)].length;
    const to = from + m[0].length;
    const wordCp = cp;
    cp += [...m[0]].length;
    utf16 = to;

    const lang: Language | undefined = HANGUL.test(m[0]) ? 'ko' : LATIN.test(m[0]) ? 'en' : undefined;
    const last = runs[runs.length - 1];
    if (lang === undefined) {
      if (last) last.to = to;
      else if (pendingNeutral) pendingNeutral.to = to;
      else pendingNeutral = { from, to, cp: wordCp };
    } else if (last && last.lang === lang) {
      last.to = to;
    } else {
      const start = pendingNeutral ?? { from, cp: wordCp };
      pendingNeutral = undefined;
      runs.push({ lang, from: start.from, to, cp: start.cp });
    }
  }
  if (pendingNeutral) runs.push({ lang: 'ko', ...pendingNeutral });
  return runs;
}

/**
 * Analyze text that mixes Korean and English. Whitespace-separated words
 * containing Hangul go to `analyzer` (a loaded `Garu`); words with Latin letters
 * use `analyzeEnglish`. Numbers, symbols and emoji join the neighbouring run.
 * Offsets are Unicode code points in `text`; Korean tokens keep Garu's eojeol spans.
 */
export function analyzeMixed(text: string, analyzer: KoreanAnalyzer): MixedToken[] {
  const tokens: MixedToken[] = [];
  for (const run of splitRuns(text)) {
    const chunk = text.slice(run.from, run.to);
    if (run.lang === 'en') {
      for (const t of analyzeEnglish(chunk)) {
        tokens.push({ ...t, start: t.start + run.cp, end: t.end + run.cp, lang: 'en' });
      }
    } else {
      const result = analyzer.analyze(chunk) as AnalyzeResult;
      for (const t of result.tokens) {
        tokens.push({ ...t, start: t.start + run.cp, end: t.end + run.cp, lang: 'ko' });
      }
    }
  }
  return tokens;
}
