import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Garu } from '../src/node';
import { splitSentences } from '../src/sentences';
import type { SentenceAnalyzer } from '../src/sentences';
import { splitSentences as splitPunctuation } from '../src/normalize';

describe('optional sentence verifier entry', () => {
  let garu: Garu;
  let analyzer: SentenceAnalyzer;

  beforeAll(async () => {
    garu = await Garu.load();
    analyzer = garu;
  });

  afterAll(() => {
    garu?.destroy();
  });

  it('splits offline without loading a morphological analyzer', () => {
    expect(splitSentences('안녕하세요반갑습니다')).toEqual([
      { text: '안녕하세요', offset: 0 },
      { text: '반갑습니다', offset: 5 },
    ]);
    expect(splitSentences('  😀안녕하세요\t반갑습니다  ')).toEqual([
      { text: '😀안녕하세요', offset: 2 },
      { text: '반갑습니다', offset: 9 },
    ]);
    expect(splitSentences('“안녕하세요반갑습니다”')).toEqual([
      { text: '“안녕하세요반갑습니다”', offset: 0 },
    ]);
  });

  it('uses the selected local model with a real Garu analyzer', () => {
    expect(splitSentences('안녕하세요반갑습니다', analyzer)).toEqual([
      { text: '안녕하세요', offset: 0 },
      { text: '반갑습니다', offset: 5 },
    ]);
    expect(splitPunctuation('안녕하세요반갑습니다')).toEqual([
      { text: '안녕하세요반갑습니다', offset: 0 },
    ]);
  });

  it('returns original text spans and codepoint offsets around emoji and whitespace', () => {
    const text = '  😀안녕하세요\t반갑습니다  ';
    expect(splitSentences(text, analyzer)).toEqual([
      { text: '😀안녕하세요', offset: 2 },
      { text: '반갑습니다', offset: 9 },
    ]);
  });

  it('preserves a Unicode URL while separating the following sentence', () => {
    expect(splitSentences('주소는 https://example.com/한글검색 입니다.다음입니다.', analyzer)).toEqual([
      { text: '주소는 https://example.com/한글검색 입니다.', offset: 0 },
      { text: '다음입니다.', offset: 33 },
    ]);
  });

  it('does not call the analyzer when no candidates remain', () => {
    const unused: SentenceAnalyzer = { analyze() { throw new Error('no candidates'); } };
    expect(splitSentences('', unused)).toEqual([{ text: '', offset: 0 }]);
    expect(splitSentences(' \n\t', unused)).toEqual([{ text: '', offset: 0 }]);
    expect(splitSentences('https://example.com/한글검색', unused)).toEqual([
      { text: 'https://example.com/한글검색', offset: 0 },
    ]);
    expect(splitSentences('“안녕하세요반갑습니다”', unused)).toEqual([
      { text: '“안녕하세요반갑습니다”', offset: 0 },
    ]);
  });
});
