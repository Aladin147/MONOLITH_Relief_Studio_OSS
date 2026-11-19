/** Runs at most one async job at a time. Submits made while a job is in flight
 * collapse into a single follow-up run with the latest argument. A rejected run
 * never stalls later submits. */
export function createLatestRunner<T>(run: (arg: T) => Promise<unknown>): {
  submit(arg: T): void;
  cancel(): void;
  readonly busy: boolean;
} {
  let running = false;
  let pending: { arg: T } | null = null;

  const start = (arg: T) => {
    running = true;
    let p: Promise<unknown>;
    try {
      p = run(arg);
    } catch (e) {
      p = Promise.reject(e);
    }
    p.catch(() => {}).then(() => {
      running = false;
      if (pending) {
        const next = pending.arg;
        pending = null;
        start(next);
      }
    });
  };

  return {
    submit(arg: T) {
      if (running) pending = { arg };
      else start(arg);
    },
    cancel() {
      pending = null;
    },
    get busy() {
      return running || pending !== null;
    },
  };
}
