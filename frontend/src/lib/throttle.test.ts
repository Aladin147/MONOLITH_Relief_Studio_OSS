import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createThrottle } from './throttle';

describe('createThrottle', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('fires on the leading edge and once on the trailing edge with latest args', () => {
    const fn = vi.fn();
    const t = createThrottle(fn, 50);
    for (let i = 1; i <= 10; i++) {
      t.call(i);
      if (i < 10) vi.advanceTimersByTime(5);
    }
    expect(fn).toHaveBeenCalledTimes(1);
    expect(fn).toHaveBeenLastCalledWith(1);
    vi.advanceTimersByTime(100);
    expect(fn).toHaveBeenCalledTimes(2);
    expect(fn).toHaveBeenLastCalledWith(10);
  });

  it('cancel drops the pending trailing call', () => {
    const fn = vi.fn();
    const t = createThrottle(fn, 50);
    t.call(1);
    t.call(2);
    t.cancel();
    vi.advanceTimersByTime(200);
    expect(fn).toHaveBeenCalledTimes(1);
    expect(fn).toHaveBeenCalledWith(1);
  });

  it('fires again immediately after the window has fully elapsed', () => {
    const fn = vi.fn();
    const t = createThrottle(fn, 50);
    t.call('a');
    vi.advanceTimersByTime(200);
    t.call('b');
    expect(fn).toHaveBeenCalledTimes(2);
    expect(fn).toHaveBeenLastCalledWith('b');
  });
});
