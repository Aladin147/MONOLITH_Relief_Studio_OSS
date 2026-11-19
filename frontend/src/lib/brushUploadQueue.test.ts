import { describe, expect, it } from 'vitest';
import { BrushUploadQueue, type BrushJob } from './brushUploadQueue';

function deferred() {
  let resolve!: () => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<void>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const job = (projectId: string, values: number[], width = values.length, height = 1): BrushJob => ({
  projectId,
  width,
  height,
  delta: Float32Array.from(values),
});

const tick = () => new Promise((r) => setTimeout(r, 0));

describe('BrushUploadQueue', () => {
  it('merges jobs queued behind the in-flight upload into one summed delta', async () => {
    const gates = [deferred(), deferred()];
    const calls: BrushJob[] = [];
    const queue = new BrushUploadQueue(
      async (j) => {
        calls.push(j);
        await gates[calls.length - 1].promise;
      },
      () => undefined
    );

    queue.enqueue(job('p', [1, 0, 0]));
    queue.enqueue(job('p', [0, 2, 0]));
    queue.enqueue(job('p', [0, 0, 4]));
    queue.enqueue(job('p', [0.5, 0.5, 0.5]));
    expect(queue.pending).toBe(2);

    gates[0].resolve();
    await tick();
    gates[1].resolve();
    await queue.whenIdle();

    expect(calls).toHaveLength(2);
    expect(Array.from(calls[0].delta)).toEqual([1, 0, 0]);
    expect(Array.from(calls[1].delta)).toEqual([0.5, 2.5, 4.5]);
    expect(queue.pending).toBe(0);
  });

  it('does not mutate the in-flight job or the caller arrays', async () => {
    const gate = deferred();
    const first = job('p', [1, 1]);
    const second = job('p', [2, 2]);
    const third = job('p', [3, 3]);
    const queue = new BrushUploadQueue(
      async () => gate.promise,
      () => undefined
    );
    queue.enqueue(first);
    queue.enqueue(second);
    queue.enqueue(third);
    gate.resolve();
    await queue.whenIdle();
    expect(Array.from(first.delta)).toEqual([1, 1]);
    expect(Array.from(second.delta)).toEqual([2, 2]);
    expect(Array.from(third.delta)).toEqual([3, 3]);
  });

  it('never merges across projects or dimensions', async () => {
    const gate = deferred();
    const calls: BrushJob[] = [];
    const queue = new BrushUploadQueue(
      async (j) => {
        calls.push(j);
        if (calls.length === 1) await gate.promise;
      },
      () => undefined
    );
    queue.enqueue(job('a', [1, 1]));
    queue.enqueue(job('b', [1, 1]));
    queue.enqueue(job('a', [1, 1]));
    queue.enqueue(job('a', [1, 1, 1, 1], 4, 1));
    gate.resolve();
    await queue.whenIdle();
    expect(calls.map((c) => [c.projectId, c.delta.length])).toEqual([
      ['a', 2],
      ['b', 2],
      ['a', 2],
      ['a', 4],
    ]);
  });

  it('whenIdle resolves only after every job has settled', async () => {
    const gate = deferred();
    const queue = new BrushUploadQueue(
      async () => gate.promise,
      () => undefined
    );
    queue.enqueue(job('p', [1]));
    queue.enqueue(job('q', [1]));
    let idle = false;
    const waiting = queue.whenIdle().then(() => {
      idle = true;
    });
    await tick();
    expect(idle).toBe(false);
    gate.resolve();
    await waiting;
    expect(idle).toBe(true);
    await queue.whenIdle(); // already idle: resolves immediately
  });

  it('reports a failed upload and keeps going', async () => {
    const errors: Array<[unknown, BrushJob]> = [];
    const uploaded: string[] = [];
    const boom = new Error('boom');
    const queue = new BrushUploadQueue(
      async (j) => {
        if (j.projectId === 'bad') throw boom;
        uploaded.push(j.projectId);
      },
      (err, j) => errors.push([err, j])
    );
    queue.enqueue(job('bad', [1]));
    queue.enqueue(job('good', [1]));
    await queue.whenIdle();
    expect(errors).toHaveLength(1);
    expect(errors[0][0]).toBe(boom);
    expect(errors[0][1].projectId).toBe('bad');
    expect(uploaded).toEqual(['good']);
    expect(queue.pending).toBe(0);
  });

  it('notifies pending changes', async () => {
    const seen: number[] = [];
    const queue = new BrushUploadQueue(
      async () => undefined,
      () => undefined
    );
    queue.onPendingChange = (n) => seen.push(n);
    queue.enqueue(job('p', [1]));
    await queue.whenIdle();
    expect(seen[0]).toBe(1);
    expect(seen[seen.length - 1]).toBe(0);
  });
});
