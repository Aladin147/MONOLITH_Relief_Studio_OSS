export type BrushJob = {
  projectId: string;
  width: number;
  height: number;
  delta: Float32Array;
};

/**
 * Serial brush-upload queue. Jobs still waiting (not yet started) for the same
 * project and grid size are summed into one delta, so a burst of strokes costs
 * one upload instead of one per stroke. The in-flight job is never touched.
 *
 * Summing already-clipped deltas matches the server applying them in sequence
 * except for extra clipping at 0 or 1, which the server-side clip handles.
 */
export class BrushUploadQueue {
  onPendingChange?: (n: number) => void;

  private queued: BrushJob[] = [];
  private inFlight = false;
  private idleWaiters: Array<() => void> = [];

  constructor(
    private readonly upload: (job: BrushJob) => Promise<void>,
    private readonly onError: (err: unknown, job: BrushJob) => void
  ) {}

  get pending(): number {
    return this.queued.length + (this.inFlight ? 1 : 0);
  }

  enqueue(job: BrushJob): void {
    const last = this.queued[this.queued.length - 1];
    if (
      last &&
      last.projectId === job.projectId &&
      last.width === job.width &&
      last.height === job.height &&
      last.delta.length === job.delta.length
    ) {
      // Fresh array: never mutate a caller's buffer.
      const sum = new Float32Array(last.delta.length);
      for (let i = 0; i < sum.length; i++) sum[i] = last.delta[i] + job.delta[i];
      this.queued[this.queued.length - 1] = { ...last, delta: sum };
    } else {
      this.queued.push(job);
    }
    this.notify();
    void this.pump();
  }

  whenIdle(): Promise<void> {
    if (this.pending === 0) return Promise.resolve();
    return new Promise((resolve) => this.idleWaiters.push(resolve));
  }

  private notify(): void {
    this.onPendingChange?.(this.pending);
  }

  private async pump(): Promise<void> {
    if (this.inFlight) return;
    const job = this.queued.shift();
    if (!job) return;
    this.inFlight = true;
    try {
      await this.upload(job);
    } catch (err) {
      try {
        this.onError(err, job);
      } catch {
        // an error handler must never wedge the queue
      }
    }
    this.inFlight = false;
    this.notify();
    if (this.queued.length > 0) {
      void this.pump();
    } else {
      const waiters = this.idleWaiters;
      this.idleWaiters = [];
      waiters.forEach((w) => w());
    }
  }
}
