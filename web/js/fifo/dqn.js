// Runs the FIFO Deep Q-Learning network exported by `python trainer.py --game fifo`.
// encode() must match encode() in fifo/dqn.py.
import { MAX_MOVES, legalMoves, playMove } from "./game.js";

export const RULES = "4x4-four-marks-four-in-row-100-ply";
export const ENCODING = "fifo-planes-v1";
export const LABEL = "Deep Q-Learning";
// The page's agent looks this many moves ahead: in testing it never lost at 3.
export const DEPTH = 3;

// base64 little-endian float32 -> Float32Array
function floats(text, length) {
  const bytes = Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
  const values = new Float32Array(bytes.buffer);
  if (values.length !== length) throw new Error("The FIFO model file is damaged.");
  return values;
}

// Fetch the trained network the page plays with.
export async function fetchNetwork() {
  const response = await fetch("data/FIFO-DeepQLearning.json");
  if (!response.ok) throw new Error(`Cannot load the Deep Q agent (${response.status}).`);
  return loadNetwork(await response.json());
}

export function loadNetwork(model) {
  if (model.rules !== RULES || model.encoding !== ENCODING) throw new Error("The FIFO model does not match this game.");
  const layers = model.layers.map((layer) => ({
    in: layer.in, out: layer.out, weight: floats(layer.weight, layer.in * layer.out), bias: floats(layer.bias, layer.out),
  }));
  if (layers[0]?.in !== 129 || layers.at(-1).out !== 16) throw new Error("The FIFO model has the wrong shape.");
  return { episodes: model.episodes, gamma: model.gamma ?? 0.9, layers };
}

// 8 planes of 16 squares from the mover's side: their marks, then the opponent's, each by how
// many of the owner's turns are left before the mark disappears. Then moves left / 100.
export function encode(state) {
  const x = new Float32Array(129);
  [state.player, 1 - state.player].forEach((side, s) => {
    const queue = state.queues[side];
    queue.forEach((cell, i) => { x[(s * 4 + 4 - queue.length + i) * 16 + cell] = 1; });
  });
  x[128] = (MAX_MOVES - state.moves) / MAX_MOVES;
  return x;
}

// Hidden-layer activations and output scores, for drawing the network at work.
export function activations(network, state) {
  const layers = [encode(state)];
  network.layers.forEach((layer, index) => {
    const x = layers.at(-1);
    const y = new Float32Array(layer.out);
    for (let o = 0; o < layer.out; o++) {
      let sum = layer.bias[o];
      const row = o * layer.in;
      for (let i = 0; i < layer.in; i++) if (x[i]) sum += layer.weight[row + i] * x[i];
      y[o] = index < network.layers.length - 1 ? Math.max(0, sum) : sum;
    }
    layers.push(y);
  });
  return layers;
}

// One Q-value per square for the player to move (occupied squares are meaningless).
export function qValues(network, state) {
  return Array.from(activations(network, state).at(-1));
}

// Value of each square for the player to move (-Infinity where taken). depth 0 is the
// network's own scores; depth d plays every move and reply for d more moves, scores a win +1
// and a 100-move draw 0, then lets the network judge the end position. Mirrors lookahead() in fifo/dqn.py.
export function moveValues(network, state, depth = DEPTH) {
  const values = Array(16).fill(-Infinity);
  const legal = legalMoves(state);
  if (depth === 0) {
    const q = qValues(network, state);
    for (const cell of legal) values[cell] = q[cell];
    return values;
  }
  for (const cell of legal) {
    const next = playMove(state, cell);
    values[cell] = next.winner !== null
      ? (next.winner === state.player ? 1 : 0)
      : -network.gamma * Math.max(...moveValues(network, next, depth - 1));
  }
  return values;
}
