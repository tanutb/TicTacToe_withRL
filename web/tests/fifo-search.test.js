import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { Worker } from "node:worker_threads";
import { createSearch } from "../js/fifo/search.js";
import { loadNetwork, encode, activations, moveValues } from "../js/fifo/dqn.js";
import { initialState, legalMoves, playMove } from "../js/fifo/game.js";
import { quizState } from "../js/fifo/learn.js";

const network = loadNetwork(JSON.parse(await readFile(new URL("../data/FIFO-DeepQLearning.json", import.meta.url))));
function dense(state) {
  const layers = [encode(state)];
  for (const [index, layer] of network.layers.entries()) {
    const x = layers.at(-1), y = new Float32Array(layer.out);
    for (let o = 0; o < layer.out; o++) {
      let sum = layer.bias[o];
      for (let i = 0; i < layer.in; i++) if (x[i]) sum += layer.weight[o * layer.in + i] * x[i];
      y[o] = index < network.layers.length - 1 ? Math.max(0, sum) : sum;
    }
    layers.push(y);
  }
  return layers;
}
function exhaustive(state, depth) {
  const scores = Array(16).fill(-Infinity);
  const q = depth === 0 ? dense(state).at(-1) : null;
  for (const cell of legalMoves(state)) {
    if (q) { scores[cell] = q[cell]; continue; }
    const next = playMove(state, cell);
    scores[cell] = next.winner !== null ? (next.winner === state.player ? 1 : 0)
      : -network.gamma * Math.max(...exhaustive(next, depth - 1));
  }
  return scores;
}

test("optimized inference and pruning preserve every action score", () => {
  const states = [initialState(), quizState()];
  let state = initialState();
  for (const cell of [1, 5, 10, 14, 0, 7, 12, 15, 3, 8]) {
    state = playMove(state, cell);
    if ([3, 7, 10].includes(state.moves)) states.push(state);
  }
  states.push({ ...quizState(), moves: 999 });
  for (const state of states) {
    assert.deepEqual(activations(network, state), dense(state));
    for (let depth = 0; depth <= 3; depth++) {
      assert.deepEqual(moveValues(network, state, depth), exhaustive(state, depth));
    }
  }
});

test("worker queue discards superseded work and recovers from failure", async () => {
  let worker;
  class FakeWorker {
    messages = [];
    constructor() { worker = this; }
    postMessage(message) { this.messages.push(message); }
    terminate() { this.terminated = true; }
  }
  const search = createSearch(network, FakeWorker);
  assert.equal(worker, undefined, "do not start a worker before it is needed");
  const first = search.evaluate(initialState(), 3);
  const stale = search.evaluate(quizState(), 2);
  const latest = search.evaluate(quizState(), 1);
  assert.equal(await stale, null);
  assert.equal(worker.messages.length, 2, "one model upload and one running job");
  const result = Array(16).fill(0);
  worker.onmessage({ data: { id: 1, values: result } });
  assert.deepEqual(await first, result);
  assert.equal(worker.messages.at(-1).depth, 1);
  const failed = assert.rejects(latest, /agent stopped/);
  worker.onerror();
  await failed;
  assert.equal(worker.terminated, true);
  assert.deepEqual(search.evaluate(quizState(), 0), moveValues(network, quizState(), 0));
});

test("real background worker returns the same search scores", async (t) => {
  let thread;
  class BrowserWorker {
    constructor(url) {
      thread = new Worker(`const { parentPort } = require('node:worker_threads');
        globalThis.self = { postMessage: message => parentPort.postMessage(message) };
        import(${JSON.stringify(url.href)}).then(() => {
          parentPort.on('message', data => self.onmessage({ data }));
        });`, { eval: true });
      thread.on("message", data => this.onmessage?.({ data }));
      thread.on("error", error => this.onerror?.(error));
    }
    postMessage(data) { thread.postMessage(data); }
    terminate() { thread.terminate(); }
  }
  t.after(() => thread?.terminate());
  const result = createSearch(network, BrowserWorker).evaluate(quizState(), 3);
  assert.ok(result instanceof Promise);
  assert.deepEqual(await result, moveValues(network, quizState(), 3));
});
