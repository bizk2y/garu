import { describe, it, expect } from 'vitest';
import * as boundaries from '../src/boundaries';
import { BOUNDARY_MODEL } from '../src/boundary-data';
import {
  extractSentenceFeatures, extractContextFeatures, FEATURE_NAMES, loadSentenceVerifier,
  scoreSentenceCandidates, verifySentenceCandidates, splitVerified, keepOutsideQuotes,
} from '../src/sentence-verifier';
import type { SentenceModelData } from '../src/sentence-verifier';
import { Garu } from '../src/node';

describe('sentence candidates', () => {
  it('retains the existing candidate gates and original codepoint offsets', () => {
    const text = '  😀안녕하세요\t반갑습니다  ';
    const candidates = boundaries.sentenceCandidates(text);
    expect(candidates.map(({ offset, compactOffset }) => ({ offset, compactOffset })))
      .toEqual([{ offset: 9, compactOffset: 6 }]);
    expect(candidates[0].score).toBeGreaterThan(BOUNDARY_MODEL.heads.sentences.threshold);
    expect(boundaries.splitUnspaced(text)).toEqual([
      { text: '😀안녕하세요', offset: 2 },
      { text: '반갑습니다', offset: 9 },
    ]);
  });

  it('protects a whitespace-delimited Unicode URL and ASCII addresses', () => {
    const text = '주소는 https://example.com/한글검색 입니다.다음입니다.';
    expect(boundaries.sentenceCandidates(text).map(candidate => candidate.offset)).toEqual([33]);
    for (const value of ['', ' \n\t', 'https://example.com/a?q=1.2', 'name@example.com', '3.14']) {
      expect(boundaries.sentenceCandidates(value)).toEqual([]);
    }
  });

  it('preserves representative unsplit and multi-sentence outputs', () => {
    expect(boundaries.splitUnspaced('문장은끝났다다음문장이다')).toEqual([
      { text: '문장은끝났다', offset: 0 },
      { text: '다음문장이다', offset: 6 },
    ]);
    expect(boundaries.splitUnspaced('')).toEqual([{ text: '', offset: 0 }]);
  });
});

describe('sentence morphology features', () => {
  it('extracts only context features when morphology is unavailable', () => {
    const [values] = extractContextFeatures('😀“왔다.”라고했다다음이다', [
      { offset: 6, compactOffset: 6, score: 2.5 },
    ]);
    expect(values).toEqual([2.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
    expect(extractContextFeatures('', [])).toEqual([]);
  });

  it('maps restored spaces and emoji spans to compact boundary positions', () => {
    const analyzer = {
      analyze(text: string) {
        expect(text).toBe('  😀 먹었다\t간다');
        return { tokens: [
          { text: '😀', pos: 'SW', start: 2, end: 3 },
          { text: '먹', pos: 'VV', start: 4, end: 5 },
          { text: '었', pos: 'EP', start: 5, end: 6 },
          { text: '다', pos: 'EF', start: 6, end: 7 },
          { text: '가', pos: 'VV', start: 8, end: 9 },
          { text: 'ㄴ다', pos: 'EF', start: 9, end: 10 },
        ] };
      },
    };
    const [values] = extractSentenceFeatures('  😀먹었다\t간다', [
      { offset: 7, compactOffset: 4, score: 2.5 },
    ], analyzer);
    const features = Object.fromEntries(FEATURE_NAMES.map((name, i) => [name, values[i]]));
    expect(features).toMatchObject({
      baseScore: 2.5, tokenBoundary: 1, tokenCrossing: 0, morphUnreliable: 0,
      leftEF: 1, leftEC: 0, rightVerb: 1, rightParticle: 0,
    });
  });

  it('marks a crossing token as unreliable instead of trusting its POS', () => {
    const [values] = extractSentenceFeatures('왔다간다', [
      { offset: 2, compactOffset: 2, score: 1 },
    ], { analyze: () => ({ tokens: [{ text: '왔다간다', pos: 'EF', start: 0, end: 4 }] }) });
    const features = Object.fromEntries(FEATURE_NAMES.map((name, i) => [name, values[i]]));
    expect(features).toMatchObject({ tokenCrossing: 1, morphUnreliable: 1, leftEF: 0 });
  });

  it('distinguishes quoted reporting from a completed outer sentence', () => {
    const candidates = [
      { offset: 3, compactOffset: 3, score: 1 },
      { offset: 4, compactOffset: 4, score: 1 },
    ];
    const values = extractSentenceFeatures('“좋다”고말했다', candidates,
      { analyze: () => ({ tokens: [] }) });
    const quote = FEATURE_NAMES.indexOf('insideQuote');
    const report = FEATURE_NAMES.indexOf('rightReporting');
    expect(values.map(row => row[quote])).toEqual([1, 0]);
    expect(values.map(row => row[report])).toEqual([1, 1]);
  });

  it('does not analyze when there are no candidates', () => {
    expect(extractSentenceFeatures('', [], {
      analyze() { throw new Error('empty input must not need morphology'); },
    })).toEqual([]);
  });

  it('rejects N-best results and masks invalid morphology spans', () => {
    const candidates = [{ offset: 2, compactOffset: 2, score: 1 }];
    expect(() => extractSentenceFeatures('왔다간다', candidates,
      { analyze: () => [] })).toThrow('single analysis');
    const [values] = extractSentenceFeatures('왔다간다', candidates, {
      analyze: () => ({ tokens: [{ text: '다', pos: 'EF', start: -1, end: 2 }] }),
    });
    expect(values[FEATURE_NAMES.indexOf('morphUnreliable')]).toBe(1);
    expect(values[FEATURE_NAMES.indexOf('leftEF')]).toBe(0);
  });

  it('uses the real analyzer to identify an ending and following verb', async () => {
    const analyzer = await Garu.load();
    try {
      const text = '안녕하세요반갑습니다';
      const [values] = extractSentenceFeatures(text, boundaries.sentenceCandidates(text), analyzer);
      expect(values[FEATURE_NAMES.indexOf('leftEF')]).toBe(1);
      expect(values[FEATURE_NAMES.indexOf('rightVerb')]).toBe(1);
      expect(values[FEATURE_NAMES.indexOf('morphUnreliable')]).toBe(0);
    } finally {
      analyzer.destroy();
    }
  });
});

function floats(values: number[]): string {
  const bytes = Buffer.alloc(values.length * 4);
  values.forEach((value, i) => bytes.writeFloatLE(value, i * 4));
  return bytes.toString('base64');
}

function tinyModel(): SentenceModelData {
  // Context=1 and embeddingSize=2 give 4 character inputs + the16 ABI features.
  // Embeddings: pad=[0,0], unknown=[.5,-.5], 가=[1,-1], 나=[-2,1].
  // Hidden0 = ReLU(2*left0 - right1 + baseScore + .5).
  // Hidden1 = ReLU(-left1 + right0 - .5); output=.5*h0-h1+.25.
  const hidden = new Uint8Array(20).fill(0x88);
  hidden[0] = 0x8a;
  hidden[1] = 0x78;
  hidden[2] = 0x89;
  hidden[10] = 0x78;
  hidden[11] = 0x89;
  return {
    version: 1, context: 1, embeddingSize: 2,
    featureNames: [...FEATURE_NAMES], vocabulary: ['가', '나'], threshold: 1, sourceHash: 'fixture',
    tensors: {
      embedding: { rows: 4, cols: 2, data: Buffer.from([0x88, 0x79, 0x6a, 0x96]).toString('base64'),
        scales: floats([0, .5, .5, 1]) },
      hidden: { rows: 2, cols: 20, data: Buffer.from(hidden).toString('base64'),
        scales: floats([1, 1]), bias: floats([.5, -.5]) },
      output: { rows: 1, cols: 2, data: Buffer.from([0x69]).toString('base64'),
        scales: floats([.5]), bias: floats([.25]) },
    },
  };
}

describe('Q4 sentence verifier', () => {
  const candidate = { offset: 1, compactOffset: 1, score: 1 };
  const features = [[1, ...Array(15).fill(0)]];

  it('decodes low-first signed nibbles, float32 scales and biases', () => {
    const model = loadSentenceVerifier(tinyModel());
    // 가나: h0=2-1+1+.5=2.5; h1=max(0,1-2-.5)=0; score=1.5.
    expect(scoreSentenceCandidates('가나', [candidate], features, model)).toEqual([1.5]);
    // 나가: h0=max(0,-4+1+1+.5)=0; h1=max(0,-1+1-.5)=0.
    expect(scoreSentenceCandidates('나가', [candidate], features, model)).toEqual([.25]);
    // 가가: h0=4.5, h1=1.5, so the negative output weight contributes.
    expect(scoreSentenceCandidates('가가', [candidate], features, model)).toEqual([1]);
  });

  it('ignores whitespace and counts supplementary characters as one context position', () => {
    const model = loadSentenceVerifier(tinyModel());
    expect(scoreSentenceCandidates(' 가\t나 ', [candidate], features, model)).toEqual([1.5]);
    // Unknown😀 left=[.5,-.5], 나 right=[-2,1]: h0=1-1+1+.5=1.5; h1=0.
    expect(scoreSentenceCandidates('😀나', [candidate], features, model)).toEqual([1]);
  });

  it('uses zero padding and keeps only scores strictly above the threshold', () => {
    const data = tinyModel();
    data.context = 2;
    data.tensors.hidden.cols = 24;
    const hidden = new Uint8Array(24).fill(0x88);
    // Only outer pad positions contribute, so both h values are zero.
    hidden[0] = 0x89;
    hidden[15] = 0x89;
    data.tensors.hidden.data = Buffer.from(hidden).toString('base64');
    data.tensors.hidden.bias = floats([0, 0]);
    expect(scoreSentenceCandidates('가나', [candidate], features, loadSentenceVerifier(data)))
      .toEqual([.25]);
    const model = loadSentenceVerifier(tinyModel());
    expect(verifySentenceCandidates('😀나', [candidate], features, model)).toEqual([]);
    expect(verifySentenceCandidates('가나', [candidate], features, model)).toEqual([candidate]);
    expect(scoreSentenceCandidates('', [], [], model)).toEqual([]);
  });

  it('rejects incompatible feature order, dimensions and truncated scales', () => {
    const order = tinyModel();
    order.featureNames.reverse();
    expect(() => loadSentenceVerifier(order)).toThrow();
    const shape = tinyModel();
    shape.tensors.hidden.cols--;
    expect(() => loadSentenceVerifier(shape)).toThrow();
    const truncated = tinyModel();
    truncated.tensors.embedding.scales = floats([0]);
    expect(() => loadSentenceVerifier(truncated)).toThrow();
    const model = loadSentenceVerifier(tinyModel());
    expect(() => scoreSentenceCandidates('가나', [candidate], [[]], model)).toThrow();
  });

  it('packs odd-width rows continuously and applies a distinct scale per row', () => {
    const data: SentenceModelData = {
      version: 1, context: 1, embeddingSize: 1,
      featureNames: [...FEATURE_NAMES], vocabulary: ['가', '나'], threshold: 0, sourceHash: 'odd',
      tensors: {
        // q=[0,1,2,-1], scales=[0,1,.25,.5] -> embedding=[0,1,.5,-.5].
        embedding: { rows: 4, cols: 1, data: Buffer.from([0x98, 0x7a]).toString('base64'),
          scales: floats([0, 1, .25, .5]) },
        hidden: { rows: 1, cols: 18, data: Buffer.from([0x79, ...Array(8).fill(0x88)]).toString('base64'),
          scales: floats([1]) },
        output: { rows: 1, cols: 1, data: Buffer.from([0x89]).toString('base64'),
          scales: floats([.25]) },
      },
    };
    // h0=.5-(-.5)=1; output=.25. High padding nibble of output is unused.
    expect(scoreSentenceCandidates('가나', [candidate], features, loadSentenceVerifier(data)))
      .toEqual([.25]);
  });
});

function constantModel(score: number) {
  const data = tinyModel();
  data.threshold = 0;
  data.tensors.output.data = Buffer.from([0x88]).toString('base64');
  data.tensors.output.bias = floats([score]);
  return loadSentenceVerifier(data);
}

describe('verified sentence splitting', () => {
  const analyzer = { analyze: () => ({ tokens: [] }) };

  it('retains candidate substrings with original Unicode and whitespace offsets', () => {
    const text = '  😀안녕하세요\t반갑습니다  ';
    expect(splitVerified(text, constantModel(1), analyzer)).toEqual([
      { text: '😀안녕하세요', offset: 2 },
      { text: '반갑습니다', offset: 9 },
    ]);
  });

  it('preserves whitespace inside a segment when the candidate is vetoed', () => {
    const text = '  😀안녕하세요\t반갑습니다  ';
    expect(splitVerified(text, constantModel(-1), analyzer)).toEqual([
      { text: '😀안녕하세요\t반갑습니다', offset: 2 },
    ]);
    expect(splitVerified(text, constantModel(0), analyzer)).toEqual([
      { text: '😀안녕하세요\t반갑습니다', offset: 2 },
    ]);
  });

  it('keeps protected URLs intact through the complete verification path', () => {
    expect(splitVerified('주소는 https://example.com/한글검색 입니다.다음입니다.',
      constantModel(1), analyzer)).toEqual([
      { text: '주소는 https://example.com/한글검색 입니다.', offset: 0 },
      { text: '다음입니다.', offset: 33 },
    ]);
  });

  it('retains empty-input fallback and skips morphology without candidates', () => {
    const noAnalysis = { analyze() { throw new Error('no candidates'); } };
    const model = constantModel(1);
    expect(splitVerified('', model, noAnalysis)).toEqual([{ text: '', offset: 0 }]);
    expect(splitVerified(' \n\t', model, noAnalysis)).toEqual([{ text: '', offset: 0 }]);
    expect(splitVerified('  https://example.com/a?q=1.2 ', model, noAnalysis)).toEqual([
      { text: 'https://example.com/a?q=1.2', offset: 2 },
    ]);
  });

  it('splits at supplied codepoint boundaries and trims only segment edges', () => {
    expect(boundaries.splitAtBoundaries(' \t😀가 \n나 다  ', [6])).toEqual([
      { text: '😀가', offset: 2 },
      { text: '나 다', offset: 6 },
    ]);
  });
});

function candidatesAt(offsets: number[]) {
  return offsets.map(offset => ({ offset, compactOffset: offset, score: 1 }));
}

describe('balanced top-level quotations', () => {
  it('protects nested curved and CJK families through their closing marker', () => {
    const candidates = candidatesAt([1, 2, 3, 4, 5, 6, 7]);
    for (const text of ['“가‘나’다”라', '「가『나』다」라', '"가\'나\'다"라']) {
      expect(keepOutsideQuotes(text, candidates)).toEqual(candidatesAt([7]));
    }
  });

  it('pairs mixed curved and straight marks of the same family', () => {
    const candidates = candidatesAt([1, 2, 3, 4, 5, 6, 7]);
    expect(keepOutsideQuotes('“가"나"다”라', candidates)).toEqual(candidatesAt([3, 4, 7]));
    expect(keepOutsideQuotes('‘가\'나\'다’라', candidates)).toEqual(candidatesAt([3, 4, 7]));
  });

  it('does not extend protection after unmatched or crossed directional marks', () => {
    expect(keepOutsideQuotes('“가나다', candidatesAt([1, 2, 3]))).toEqual(candidatesAt([1, 2, 3]));
    expect(keepOutsideQuotes('가”나다', candidatesAt([1, 2, 3]))).toEqual(candidatesAt([1, 2, 3]));
    expect(keepOutsideQuotes('“가‘나”다’라', candidatesAt([1, 2, 3, 4, 5, 6, 7])))
      .toEqual(candidatesAt([1, 2, 3, 4, 5, 6, 7]));
    // An unfinished outer quote does not erase a complete nested pair.
    expect(keepOutsideQuotes('“가‘나’다라', candidatesAt([1, 2, 3, 4, 5, 6])))
      .toEqual(candidatesAt([1, 2, 5, 6]));
    expect(keepOutsideQuotes('“가』나”라', candidatesAt([1, 2, 3, 4, 5])))
      .toEqual(candidatesAt([5]));
  });

  it('ignores ASCII word apostrophes while preserving a quoted span', () => {
    const candidates = candidatesAt([3, 4, 7, 8, 16, 17, 18]);
    expect(keepOutsideQuotes("can't O'Reilly '가'나", candidates)).toEqual(candidatesAt([3, 4, 7, 8, 18]));
    expect(keepOutsideQuotes('‘O’Reilly가’나', candidatesAt([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])))
      .toEqual(candidatesAt([11]));
  });

  it('uses codepoint offsets and allows a boundary immediately after the closer', () => {
    expect(keepOutsideQuotes('😀“가”나', candidatesAt([1, 2, 3, 4])))
      .toEqual(candidatesAt([1, 4]));
    expect(keepOutsideQuotes('', [])).toEqual([]);
  });

  it('does not pair a Latin possessive apostrophe with a later quotation', () => {
    const text = "James' 작품을 봤다. 그는 '안녕'이라고 했다.";
    expect(keepOutsideQuotes(text, candidatesAt([14]))).toEqual(candidatesAt([14]));
    expect(keepOutsideQuotes("'hello' world", candidatesAt([1, 6, 7, 8])))
      .toEqual(candidatesAt([7, 8]));
  });

  it('ignores apostrophes between Unicode Latin letters but closes before Korean particles', () => {
    const text = "L'été라고 썼다. 그는 '안녕'이라고 했다.";
    expect(keepOutsideQuotes(text, candidatesAt([11]))).toEqual(candidatesAt([11]));
    expect(keepOutsideQuotes('L’été라고 썼다. 그는 ‘안녕’이라고 했다.', candidatesAt([11])))
      .toEqual(candidatesAt([11]));
    expect(keepOutsideQuotes("'안녕'이라고", candidatesAt([1, 2, 3, 4, 5])))
      .toEqual(candidatesAt([4, 5]));
  });

  it('guards before morphology and leaves the original text splitter unchanged', () => {
    const text = '“안녕하세요반갑습니다”';
    expect(boundaries.splitUnspaced(text)).toEqual([
      { text: '“안녕하세요', offset: 0 },
      { text: '반갑습니다”', offset: 6 },
    ]);
    expect(splitVerified(text, constantModel(1), {
      analyze() { throw new Error('guard removed every candidate'); },
    })).toEqual([{ text, offset: 0 }]);
  });
});
