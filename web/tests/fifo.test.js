import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { LINES, MAX_MOVES, initialState, boardOf, legalMoves, playMove, chooseMove } from "../js/fifo/game.js";
import { ENCODING, RULES, encode, loadNetwork, qValues } from "../js/fifo/dqn.js";

const pack = (values) => Buffer.from(new Float32Array(values).buffer).toString("base64");
// a small repeatable random number generator for the opponent
function seededRandom(seed) {
  let value = seed >>> 0;
  return () => {
    value = (Math.imul(1664525, value) + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

test("four marks stay; the fifth removes only that player's oldest", () => {
  let state = initialState();
  for (const cell of [0, 1, 5, 2, 10, 4, 14, 7]) state = playMove(state, cell);
  assert.deepEqual(state.queues, [[0, 5, 10, 14], [1, 2, 4, 7]]);
  const next = playMove(state, 15);
  assert.equal(next.removed, 0);
  assert.deepEqual(next.queues, [[5, 10, 14, 15], [1, 2, 4, 7]]);
  assert.equal(boardOf(next)[0], null);
  assert.equal(next.winner, null, "the removed 0 must not count toward the diagonal");
  assert.equal(state.queues[0][0], 0, "previous states remain intact for undo");
  const reply = playMove(next, 0);
  assert.equal(reply.removed, 1);
  assert.deepEqual(reply.queues[0], next.queues[0]);
});

test("all rows, columns and full diagonals win for either player", () => {
  for (const line of LINES) for (const player of [0, 1]) {
    const state = initialState();
    state.player = player;
    state.queues[player] = line.slice(0, 3);
    assert.equal(playMove(state, line[3]).winner, player);
  }
});

test("a replacement can win, but three in a row cannot", () => {
  const state = initialState();
  state.queues[0] = [8, 0, 1, 2];
  assert.equal(playMove(state, 3).winner, 0);
  state.queues[0] = [0, 1];
  assert.equal(playMove(state, 2).winner, null);
});

test("reject occupied cells including oldest, invalid cells, and moves after a win", () => {
  let state = initialState();
  state.queues[0] = [0, 5, 10, 14];
  for (const cell of [0, -1, 16, 1.5]) assert.throws(() => playMove(state, cell));
  state.queues[0] = [0, 1, 2];
  state = playMove(state, 3);
  assert.deepEqual(legalMoves(state), []);
  assert.throws(() => playMove(state, 9));
});

test("100 moves draws, while a win on move 100 takes priority", () => {
  const state = initialState();
  state.moves = MAX_MOVES - 1;
  assert.equal(playMove(state, 0).winner, "draw");
  state.queues[0] = [0, 1, 2];
  assert.equal(playMove(state, 3).winner, 0);
});

test("the encoding splits each side's marks by turns until they disappear", () => {
  const state = { ...initialState(), queues: [[0, 5, 10, 14], [1, 2]], player: 1, moves: 6 };
  const x = encode(state);
  // O moves: its 2 marks leave after 3 and 4 more O turns (planes 2, 3)
  assert.equal(x[2 * 16 + 1], 1);
  assert.equal(x[3 * 16 + 2], 1);
  // X's full queue: cell 0 leaves on X's next turn (plane 4), cell 14 last (plane 7)
  assert.equal(x[4 * 16 + 0], 1);
  assert.equal(x[5 * 16 + 5], 1);
  assert.equal(x[6 * 16 + 10], 1);
  assert.equal(x[7 * 16 + 14], 1);
  assert.equal(x.slice(0, 128).reduce((a, b) => a + b), 6);
  assert.ok(Math.abs(x[128] - 0.94) < 1e-6);
});

test("the network runs ReLU layers and rejects files for other rules or shapes", () => {
  // 129 -> 2 (ReLU) -> 16: hidden[0] copies the moves-left input, hidden[1] is always 0 after ReLU
  const w1 = Array(2 * 129).fill(0); w1[128] = 1; w1[129 + 128] = -1;
  const w2 = Array(16 * 2).fill(0); w2[3 * 2] = 2;
  const model = { rules: RULES, encoding: ENCODING, episodes: 7, layers: [
    { in: 129, out: 2, weight: pack(w1), bias: pack([0, 0]) },
    { in: 2, out: 16, weight: pack(w2), bias: pack(Array(16).fill(0.5)) },
  ] };
  const q = qValues(loadNetwork(model), initialState());
  assert.ok(Math.abs(q[3] - 2.5) < 1e-6);
  assert.ok(Math.abs(q[0] - 0.5) < 1e-6);
  assert.throws(() => loadNetwork({ ...model, rules: "3x3" }));
  assert.throws(() => loadNetwork({ ...model, layers: model.layers.slice(1) }));
});

test("the shipped Deep Q model loads, plays only legal moves, and takes a win", async () => {
  const model = JSON.parse(await readFile(new URL("../data/FIFO-DeepQLearning.json", import.meta.url), "utf8"));
  const network = loadNetwork(model);
  assert.ok(network.episodes > 0);
  const random = seededRandom(54);
  for (let i = 0; i < 20; i++) {
    let state = initialState();
    while (state.winner === null) {
      const legal = legalMoves(state);
      const action = state.player === 0 ? chooseMove(state, qValues(network, state)) : legal[Math.floor(random() * legal.length)];
      assert.ok(legal.includes(action));
      state = playMove(state, action);
    }
  }
  const state = { ...initialState(), queues: [[0, 1, 2], [8, 9, 12]], moves: 6 };
  assert.equal(chooseMove(state, qValues(network, state)), 3, "completes the top row");
});
