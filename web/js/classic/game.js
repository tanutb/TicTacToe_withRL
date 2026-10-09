// Game rules, same board format as the Python side:
// a 9 char string, "0" empty, "1" X, "2" O.

export const EMPTY = "000000000";

export const LINES = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],
  [0, 3, 6], [1, 4, 7], [2, 5, 8],
  [0, 4, 8], [2, 4, 6],
];

export function winningLine(board) {
  return LINES.find(([a, b, c]) => board[a] !== "0" && board[a] === board[b] && board[b] === board[c]) ?? null;
}

// "1", "2", "draw", or null while the game is still going
export function result(board) {
  const line = winningLine(board);
  if (line) return board[line[0]];
  return board.includes("0") ? null : "draw";
}

export function legalMoves(board) {
  if (result(board)) return [];
  return [...board].flatMap((cell, i) => (cell === "0" ? [i] : []));
}

export function turn(board) {
  const xs = [...board].filter((c) => c === "1").length;
  const os = [...board].filter((c) => c === "2").length;
  return xs === os ? "1" : "2";
}

export function move(board, index) {
  if (!legalMoves(board).includes(index)) throw new Error("That cell can't be played.");
  return board.slice(0, index) + turn(board) + board.slice(index + 1);
}

// The exported policy already stores the greedy move for every reachable board.
export function chooseAction(board, policy) {
  const entry = policy.states[board];
  if (!entry || !legalMoves(board).includes(entry.action)) {
    throw new Error("The policy file has no move for this board.");
  }
  return entry.action;
}
