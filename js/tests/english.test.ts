import { describe, it, expect } from 'vitest';
import { analyzeEnglish, stemEnglish, tokenizeEnglish } from '../src/english';

// Expected values are NLTK 3.x outputs from its documentation and doctests.
describe('tokenizeEnglish (NLTK word_tokenize)', () => {
  it('splits sentences and final periods', () => {
    const s = 'Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks.';
    expect(tokenizeEnglish(s)).toEqual([
      'Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York', '.',
      'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.',
    ]);
  });

  it('matches TreebankWordTokenizer when preserving lines', () => {
    const s = 'Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.';
    expect(tokenizeEnglish(s, { preserveLine: true })).toEqual([
      'Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York.',
      'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks', '.',
    ]);
  });

  it('keeps numbers with separators and splits currency', () => {
    expect(tokenizeEnglish('On a $50,000 mortgage of 30 years at 8 percent, the monthly payment would be $366.88.')).toEqual([
      'On', 'a', '$', '50,000', 'mortgage', 'of', '30', 'years', 'at', '8', 'percent', ',',
      'the', 'monthly', 'payment', 'would', 'be', '$', '366.88', '.',
    ]);
  });

  it('converts double quotes to PTB quote tokens', () => {
    expect(tokenizeEnglish('"We beat some pretty good teams to get here," Slocum said.')).toEqual([
      '``', 'We', 'beat', 'some', 'pretty', 'good', 'teams', 'to', 'get', 'here', ',', "''",
      'Slocum', 'said', '.',
    ]);
  });

  it('splits clitics, parentheses and quotes but keeps hyphenated words', () => {
    const s = 'Well, we couldn\'t have this predictable, cliche-ridden, "Touched by an Angel" '
      + '(a show creator John Masius worked on) wanna-be if she didn\'t.';
    expect(tokenizeEnglish(s)).toEqual([
      'Well', ',', 'we', 'could', "n't", 'have', 'this', 'predictable', ',', 'cliche-ridden', ',',
      '``', 'Touched', 'by', 'an', 'Angel', "''", '(', 'a', 'show', 'creator', 'John', 'Masius',
      'worked', 'on', ')', 'wanna-be', 'if', 'she', 'did', "n't", '.',
    ]);
  });

  it('splits MacIntyre contractions', () => {
    expect(tokenizeEnglish('I cannot cannot work under these conditions!')).toEqual([
      'I', 'can', 'not', 'can', 'not', 'work', 'under', 'these', 'conditions', '!',
    ]);
    expect(tokenizeEnglish("I'm gonna go, wanna come?")).toEqual([
      'I', "'m", 'gon', 'na', 'go', ',', 'wan', 'na', 'come', '?',
    ]);
  });

  it('handles possessive and single-quote edge cases', () => {
    expect(tokenizeEnglish("The company spent $30,000,000 to promote the new product, but it's not making a profit.")).toEqual([
      'The', 'company', 'spent', '$', '30,000,000', 'to', 'promote', 'the', 'new', 'product', ',',
      'but', 'it', "'s", 'not', 'making', 'a', 'profit', '.',
    ]);
    expect(tokenizeEnglish("The 'v', I've been fooled but I'll seek revenge.")).toEqual([
      'The', "'", 'v', "'", ',', 'I', "'ve", 'been', 'fooled', 'but', 'I', "'ll", 'seek', 'revenge', '.',
    ]);
    expect(tokenizeEnglish("'v' 're'")).toEqual(["'", 'v', "'", "'re", "'"]);
  });

  // Sentence splitting is a heuristic stand-in for Punkt, not an NLTK parity case.
  it('does not split sentences after common abbreviations or initials', () => {
    expect(tokenizeEnglish('Mr. Smith met J. K. Rowling in the U.S. Then he left.')).toEqual([
      'Mr.', 'Smith', 'met', 'J.', 'K.', 'Rowling', 'in', 'the', 'U.S.', 'Then', 'he', 'left', '.',
    ]);
  });

  it('returns no tokens for empty or blank input', () => {
    expect(tokenizeEnglish('')).toEqual([]);
    expect(tokenizeEnglish('   \n')).toEqual([]);
  });
});

describe('stemEnglish (NLTK PorterStemmer)', () => {
  it('matches the NLTK stem doctest', () => {
    const plurals = [
      'caresses', 'flies', 'dies', 'mules', 'denied', 'died', 'agreed', 'owned', 'humbled', 'sized',
      'meeting', 'stating', 'siezing', 'itemization', 'sensational', 'traditional', 'reference',
      'colonizer', 'plotted',
    ];
    expect(plurals.map(stemEnglish).join(' ')).toBe(
      'caress fli die mule deni die agre own humbl size meet state siez item sensat tradit refer colon plot',
    );
  });

  it('applies NLTK extensions', () => {
    expect(stemEnglish('skies')).toBe('sky');
    expect(stemEnglish('dying')).toBe('die');
    expect(stemEnglish('happy')).toBe('happi');
    expect(stemEnglish('enjoy')).toBe('enjoy');
    expect(stemEnglish('generously')).toBe('gener');
  });

  it('lowercases and leaves short words unstemmed', () => {
    expect(stemEnglish('Running')).toBe('run');
    expect(stemEnglish('IS')).toBe('is');
    expect(stemEnglish('connections')).toBe('connect');
  });
});

describe('analyzeEnglish', () => {
  it('tags symbols with Sejong tags and stems SL tokens', () => {
    const tokens = analyzeEnglish('The runners aren\'t running (yet)...');
    expect(tokens.map((t) => [t.text, t.pos, t.stem])).toEqual([
      ['The', 'SL', 'the'],
      ['runners', 'SL', 'runner'],
      ['are', 'SL', 'are'],
      ["n't", 'SL', "n't"],
      ['running', 'SL', 'run'],
      ['(', 'SS', '('],
      ['yet', 'SL', 'yet'],
      [')', 'SS', ')'],
      ['...', 'SE', '...'],
    ]);
  });

  it('maps PTB quote tokens back to source quotes with code point offsets', () => {
    const text = '😀 "Hi," she said.';
    const tokens = analyzeEnglish(text);
    const cps = [...text];
    for (const t of tokens) {
      expect(cps.slice(t.start, t.end).join('')).toBe(t.text);
    }
    expect(tokens.map((t) => [t.text, t.pos])).toEqual([
      ['😀', 'SW'], ['"', 'SS'], ['Hi', 'SL'], [',', 'SP'], ['"', 'SS'],
      ['she', 'SL'], ['said', 'SL'], ['.', 'SF'],
    ]);
  });

  it('tags numbers as SN', () => {
    expect(analyzeEnglish('It costs $3.88 today.').map((t) => [t.text, t.pos])).toEqual([
      ['It', 'SL'], ['costs', 'SL'], ['$', 'SW'], ['3.88', 'SN'], ['today', 'SL'], ['.', 'SF'],
    ]);
  });
});
