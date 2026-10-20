/** Optional local sentence verifier. Import from `garu-ko/sentences`. */
import { SENTENCE_MODEL } from './sentence-data.js';
import { SENTENCE_MODEL as FAST_MODEL } from './sentence-fast-data.js';
import { loadSentenceVerifier, splitVerified } from './sentence-verifier.js';
import type { SentenceAnalyzer, SentenceVerifier } from './sentence-verifier.js';
import type { Segment } from './normalize.js';

export type { SentenceAnalyzer } from './sentence-verifier.js';

let model: SentenceVerifier | undefined;
let fastModel: SentenceVerifier | undefined;

/** Fast offline splitting; supplying an analyzer opts into the slower morphology verifier. */
export function splitSentences(text: string, analyzer?: SentenceAnalyzer): Segment[] {
  if (!analyzer) {
    fastModel ??= loadSentenceVerifier(FAST_MODEL);
    return splitVerified(text, fastModel);
  }
  model ??= loadSentenceVerifier(SENTENCE_MODEL);
  return splitVerified(text, model, analyzer);
}
