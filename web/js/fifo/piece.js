import { markSvg } from "../shared/marks.js";

// Turns a mark stays on the board: 1 means it vanishes on its owner's next move.
export function turnsLeft(state, player, cell) {
  const queue = state.queues[player];
  return 4 - queue.length + queue.indexOf(cell) + 1;
}

// A mark with dots for the turns it has left; the one about to vanish is faded.
export function pieceHtml(state, player, cell) {
  const left = turnsLeft(state, player, cell);
  const dots = [1, 2, 3, 4].map((i) => `<i class="${i <= left ? "on" : ""}"></i>`).join("");
  return `<span class="fifo-piece${left === 1 ? " leaving" : ""}">${markSvg(player === 0)}<span class="life">${dots}</span></span>`;
}

export function pieceLabel(state, player, cell) {
  const left = turnsLeft(state, player, cell);
  const mark = player === 0 ? "X" : "O";
  return left === 1 ? `${mark}, vanishes on ${mark}’s next move` : `${mark}, ${left} turns left`;
}

// A fading copy of a removed mark, laid over its old square.
export function ghost(player) {
  const span = document.createElement("span");
  span.className = `fifo-departing ${player === 0 ? "fifo-x" : "fifo-o"}`;
  span.setAttribute("aria-hidden", "true");
  span.innerHTML = markSvg(player === 0);
  return span;
}
