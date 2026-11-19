/** Leading + trailing throttle: fires immediately, then at most once per
 * `waitMs`, with the latest arguments on the trailing edge. */
export function createThrottle<A extends unknown[]>(
  fn: (...a: A) => void,
  waitMs: number,
): { call(...a: A): void; cancel(): void } {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let pending: A | null = null;

  const open = () => {
    timer = setTimeout(() => {
      timer = null;
      if (pending) {
        const args = pending;
        pending = null;
        fn(...args);
        open(); // trailing call starts a new window
      }
    }, waitMs);
  };

  return {
    call(...a: A) {
      if (timer === null) {
        fn(...a);
        open();
      } else {
        pending = a;
      }
    },
    cancel() {
      if (timer !== null) clearTimeout(timer);
      timer = null;
      pending = null;
    },
  };
}
