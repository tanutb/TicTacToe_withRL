export const MAX_MOVES = 100;
export const LINES = [
  ...Array.from({ length: 4 }, (_, r) => Array.from({ length: 4 }, (_, c) => r * 4 + c)),
  ...Array.from({ length: 4 }, (_, c) => Array.from({ length: 4 }, (_, r) => r * 4 + c)),
  [0, 5, 10, 15], [3, 6, 9, 12],
];

export function initialState() {
  return { queues: [[], []], player: 0, moves: 0, winner: null, removed: null };
}

export function boardOf(state) {
  const board = Array(16).fill(null);
  state.queues.forEach((queue, player) => queue.forEach((cell) => { board[cell] = player; }));
  return board;
}

export function legalMoves(state) {
  if (state.winner !== null) return [];
  return boardOf(state).flatMap((mark, cell) => mark === null ? [cell] : []);
}

export function playMove(state, cell) {
  if (!legalMoves(state).includes(cell)) throw new Error("Choose an empty square in an unfinished game.");
  const queues = state.queues.map((queue) => [...queue]);
  const queue = queues[state.player];
  const removed = queue.length === 4 ? queue.shift() : null;
  queue.push(cell);
  // Check only after removing the oldest mark: a vanished mark cannot complete a line.
  const won = LINES.some((line) => line.every((position) => queue.includes(position)));
  const moves = state.moves + 1;
  return { queues, player: 1 - state.player, moves, removed, winner: won ? state.player : moves >= MAX_MOVES ? "draw" : null };
}

export function chooseMove(state, values, random = Math.random) {
  const legal = legalMoves(state);
  if (!legal.length) throw new Error("The game has ended.");
  const best = Math.max(...legal.map((cell) => values[cell] ?? 0));
  const ties = legal.filter((cell) => (values[cell] ?? 0) === best);
  return ties[Math.floor(random() * ties.length)];
}
