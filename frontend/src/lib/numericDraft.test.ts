import { describe, expect, it } from 'vitest';
import { commitNumericDraft, resolveDraftCommit } from './numericDraft';

describe('commitNumericDraft', () => {
  it('keeps an in-range value exactly', () => {
    expect(commitNumericDraft('1500', 100, 5000)).toBe(1500);
  });
  it('clamps below min and above max', () => {
    expect(commitNumericDraft('1', 100, 5000)).toBe(100);
    expect(commitNumericDraft('9999', 100, 5000)).toBe(5000);
    expect(commitNumericDraft('-5', 0, 10)).toBe(0);
  });
  it('returns null for empty or non-numeric input', () => {
    expect(commitNumericDraft('abc', 0, 10)).toBeNull();
    expect(commitNumericDraft('', 0, 10)).toBeNull();
    expect(commitNumericDraft('   ', 0, 10)).toBeNull();
    expect(commitNumericDraft('Infinity', 0, 10)).toBeNull();
  });
  it('parses exponent notation', () => {
    expect(commitNumericDraft('1e3', 0, 5000)).toBe(1000);
  });
});

describe('resolveDraftCommit', () => {
  it('does not commit an untouched draft (tab-through of 1.2345 shown as 1.23)', () => {
    expect(resolveDraftCommit('1.23', '1.23', 1.2345, 0, 10)).toBeNull();
  });
  it('commits an edited draft that differs from the value', () => {
    expect(resolveDraftCommit('2', '1.23', 1.2345, 0, 10)).toBe(2);
  });
  it('does not commit when the clamped result equals the value', () => {
    expect(resolveDraftCommit('50', '5', 10, 0, 10)).toBeNull();
  });
  it('does not commit invalid input', () => {
    expect(resolveDraftCommit('abc', '1.23', 1.2345, 0, 10)).toBeNull();
  });
});
