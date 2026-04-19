import { describe, it, expect } from 'vitest';
import { cleanDescription, looksLikeJunk, renderDescription } from './sceneDescription';

describe('looksLikeJunk', () => {
  it('flags dict repr', () => expect(looksLikeJunk("{'raw_response': 'hi'}")).toBe(true));
  it('flags json leak', () => expect(looksLikeJunk('{"description":"x"}')).toBe(true));
  it('flags errors', () => expect(looksLikeJunk('Error: connection refused')).toBe(true));
  it('flags refusals', () => expect(looksLikeJunk('I cannot analyze this image.')).toBe(true));
  it('accepts real prose', () => expect(looksLikeJunk('Two people at a desk, one gesturing.')).toBe(false));
  it('accepts synthetic markers', () => {
    expect(looksLikeJunk('Key moment 3 at 1:23')).toBe(false);
    expect(looksLikeJunk('Key moment 5: the speaker introduces the topic')).toBe(false);
  });
  it('rejects too-short', () => expect(looksLikeJunk('ok')).toBe(true));
});

describe('cleanDescription', () => {
  it('collapses whitespace', () => expect(cleanDescription('  hi   there  ')).toBe('hi there'));
  it('strips control chars', () => expect(cleanDescription('A\u0000B\u0001C')).toBe('A B C'));
  it('returns empty for falsy', () => expect(cleanDescription(null)).toBe(''));
});

describe('renderDescription', () => {
  it('substitutes for junk', () => {
    expect(renderDescription("{'x':1}", 83, 4)).toBe('Key moment 5 at 1:23');
  });
  it('passes valid through cleaned', () => {
    expect(renderDescription('  Two people talking.\n\n', 10, 0)).toBe('Two people talking.');
  });
});
