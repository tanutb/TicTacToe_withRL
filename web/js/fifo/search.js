import { moveValues } from "./dqn.js";

// One worker per controller; retain only the newest queued position.
export function createSearch(network, WorkerClass = globalThis.Worker) {
  if (!WorkerClass) return { evaluate: (state, depth) => moveValues(network, state, depth) };
  let worker;
  let started = false;
  let running = null;
  let queued = null;
  let sequence = 0;
  const send = (job) => {
    running = job;
    worker.postMessage({ id: job.id, state: job.state, depth: job.depth });
  };
  const fail = () => {
    worker?.terminate();
    worker = null;
    for (const job of [running, queued]) {
      if (job) job.reject(new Error("The agent stopped. Please retry."));
    }
    running = queued = null;
  };
  function start() {
    started = true;
    try {
      worker = new WorkerClass(new URL("./search-worker.js", import.meta.url), { type: "module" });
      worker.postMessage({ network });
      worker.onerror = fail;
      worker.onmessage = ({ data }) => {
        if (!running || data.id !== running.id) return;
        const job = running;
        running = null;
        if (data.error) job.reject(new Error(data.error));
        else job.resolve(data.values);
        if (queued) { const next = queued; queued = null; send(next); }
      };
    } catch {
      worker?.terminate();
      worker = null;
    }
  }
  return {
    evaluate(state, depth) {
      if (!started) start();
      if (!worker) return moveValues(network, state, depth);
      return new Promise((resolve, reject) => {
        const job = { id: ++sequence, state, depth, resolve, reject };
        if (!running) send(job);
        else {
          queued?.resolve(null); // obsolete position; its controller ignores it
          queued = job;
        }
      });
    },
  };
}
