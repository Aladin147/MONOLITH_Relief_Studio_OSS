import { describe, expect, it } from 'vitest';
import { createLatestRunner } from './latestRunner';

function deferred() {
  let resolve!: () => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<void>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}
const tick = () => new Promise((r) => setTimeout(r, 0));

describe('createLatestRunner', () => {
  it('collapses submits during an in-flight run into one follow-up with the latest arg', async () => {
    const gates: ReturnType<typeof deferred>[] = [];
    const seen: number[] = [];
    const r = createLatestRunner<number>((n) => {
      seen.push(n);
      const d = deferred();
      gates.push(d);
      return d.promise;
    });
    r.submit(1);
    r.submit(2);
    r.submit(3);
    expect(seen).toEqual([1]);
    expect(r.busy).toBe(true);
    gates[0].resolve();
    await tick();
    expect(seen).toEqual([1, 3]);
    gates[1].resolve();
    await tick();
    expect(seen).toEqual([1, 3]);
    expect(r.busy).toBe(false);
  });

  it('cancel drops the pending arg but lets the in-flight run finish', async () => {
    const gates: ReturnType<typeof deferred>[] = [];
    const seen: number[] = [];
    const r = createLatestRunner<number>((n) => {
      seen.push(n);
      const d = deferred();
      gates.push(d);
      return d.promise;
    });
    r.submit(1);
    r.submit(2);
    r.cancel();
    gates[0].resolve();
    await tick();
    expect(seen).toEqual([1]);
    expect(r.busy).toBe(false);
  });

  it('a rejected run does not stall later submits', async () => {
    const seen: number[] = [];
    const r = createLatestRunner<number>((n) => {
      seen.push(n);
      return n === 1 ? Promise.reject(new Error('boom')) : Promise.resolve();
    });
    r.submit(1);
    r.submit(2);
    await tick();
    r.submit(3);
    await tick();
    expect(seen).toEqual([1, 2, 3]);
    expect(r.busy).toBe(false);
  });
});
