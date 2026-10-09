import { LINES, MAX_MOVES, initialState, boardOf, legalMoves, playMove, chooseMove } from "./game.js";
import { DEPTH, LABEL, fetchNetwork } from "./dqn.js";
import { createSearch } from "./search.js";
import { ghost, pieceHtml, pieceLabel } from "./piece.js";

const DEPTH_NOTES = [
  "Network only: fast, but can walk into a trap.",
  "Tries one move, then scores the board.",
  "Checks its move and your best reply.",
  "Checks its move, your reply, then its next move.",
];

// The Play tab. `loader` resolves to the trained network (shared with the other tabs).
export function initFIFO(root = document.getElementById("panel-play"), loader = fetchNetwork) {
  root.innerHTML = `
    <div class="fifo-stage">
      <div id="fifo-board" class="fifo-board" role="group" aria-label="4 by 4 FIFO tic-tac-toe board"></div>
      <p id="fifo-status" class="status" role="status">Loading the Deep Q agent…</p>
      <p id="fifo-removal" class="small" aria-live="polite"></p>
      <div class="actions fifo-actions">
        <button id="fifo-new" class="button primary" type="button">New game</button>
        <button id="fifo-undo" class="button" type="button">Undo</button>
        <button id="fifo-hint" class="button" type="button">Hint</button>
        <button id="fifo-step" class="button" type="button" hidden>Step</button>
        <button id="fifo-auto" class="button" type="button" hidden>Play</button>
      </div>
      <p class="small fifo-legend"><span><span class="life" aria-hidden="true"><i class="on"></i><i class="on"></i><i class="on"></i><i></i></span> turns left</span><span>faded = leaves next</span></p>
      <p class="fifo-rules small"><b>How to play:</b> X starts. Choose an empty square. Four in a line wins.<br>Keep four marks each; your fifth replaces your oldest. Draw after ${MAX_MOVES} moves.</p>
    </div>
    <aside class="panel fifo-panel">
      <div class="segmented" role="group" aria-label="FIFO game mode">
        <button id="fifo-human-mode" type="button" aria-pressed="true">You vs agent</button>
        <button id="fifo-watch-mode" type="button" aria-pressed="false">Agent vs agent</button>
      </div>
      <div id="fifo-human-controls">
        <label class="field inline"><span class="label">You play</span><select id="fifo-side"><option value="0">X, first</option><option value="1">O, second</option></select></label>
      </div>
      <div id="fifo-watch-controls" hidden>
        <label class="speed"><span class="label">Speed</span><input id="fifo-speed" type="range" min="200" max="1600" value="1000" step="100" aria-label="Playback speed" aria-describedby="fifo-speed-labels"></label>
        <div id="fifo-speed-labels" class="fifo-speed-labels small"><span>Slow</span><span>Fast</span></div>
      </div>
      <div class="field">
        <span class="label" id="fifo-depth-label">Look-ahead</span>
        <div class="segmented small" role="group" aria-labelledby="fifo-depth-label">${[0, 1, 2, 3].map((d) => `<button id="fifo-depth-${d}" type="button" aria-pressed="${d === DEPTH}">${d ? `${d} move${d > 1 ? "s" : ""}` : "Off"}</button>`).join("")}</div>
        <p id="fifo-depth-note" class="small"></p>
      </div>
      <label class="switch"><input id="fifo-thinking" type="checkbox"><span>Show the agent’s move scores</span></label>
      <p id="fifo-confidence" class="small"></p>
      <div class="score" aria-live="polite"><div><b id="fifo-score-x">0</b><span>X wins</span></div><div><b id="fifo-score-draw">0</b><span>Draws</span></div><div><b id="fifo-score-o">0</b><span>O wins</span></div></div>
      <div class="log"><div class="log-head"><span class="label">Moves <span id="fifo-move-count"></span></span><button id="fifo-reset-score" class="link-button" type="button">Reset score</button></div><ol id="fifo-log"></ol></div>
      <p id="fifo-error" class="error" role="alert" hidden></p>
      <button id="fifo-retry" class="button" type="button" hidden>Retry loading the agent</button>
    </aside>`;
  const $ = (id) => document.getElementById(`fifo-${id}`);
  let state = initialState();
  let history = [];
  let mode = "human";
  let human = 0;
  let active = false;
  let autoplay = false;
  let timer = null;
  let network = null;
  let search = null;
  let loggedHistory = null;
  let loading = false;
  let hint = null;
  let counted = false;
  let scores = [0, 0, 0];
  let depth = DEPTH;
  let cached = null;
  const shownCells = Array(16).fill(null);
  const cells = Array.from({ length: 16 }, (_, cell) => {
    const button = document.createElement("button");
    button.type = "button";
    button.addEventListener("click", () => {
      if (network && active && mode === "human" && state.player === human && state.winner === null) play(cell, "You");
    });
    $("board").append(button);
    return button;
  });
  const position = (cell) => `r${Math.floor(cell / 4) + 1} c${cell % 4 + 1}`;
  // the scores the agent decides with; the search is the slow part, so keep the last result
  function values() {
    if (!network || !active || state.winner !== null) return null;
    if (cached?.state === state && cached.depth === depth) return cached.values;
    const request = { state, depth, values: null };
    cached = request;
    const result = search.evaluate(state, depth);
    if (!result?.then) { request.values = result; return result; }
    result.then((scores) => {
      if (cached !== request || !scores) return;
      request.values = scores;
      if (active) { render(); schedule(); }
    }).catch((error) => {
      if (cached !== request) return;
      stop(); network = null; search = null; cached = null;
      $("error").hidden = false; $("error").textContent = error.message;
      $("retry").hidden = false;
      render();
    });
    return null;
  }
  // round first, so a tiny negative score shows as 0.00 rather than -0.00
  const round = (value) => Math.round(value * 100) / 100 || 0;
  const signed = (value) => `${round(value) > 0 ? "+" : round(value) < 0 ? "−" : ""}${Math.abs(round(value)).toFixed(2)}`;

  function stop() { clearTimeout(timer); timer = null; }
  function newGame() {
    stop(); state = initialState(); history = []; hint = null; counted = false;
    render(); schedule();
  }
  function play(cell, who) {
    const previous = state;
    const value = values()?.[cell];
    state = playMove(state, cell);
    history.push({ previous, cell, who, value, removed: state.removed });
    hint = null;
    if (state.winner !== null && !counted) {
      scores[state.winner === "draw" ? 2 : state.winner]++;
      counted = true;
      if (mode === "watch") autoplay = false;
    }
    render();
    cells[cell].querySelector("svg")?.classList.add("drawn");
    if (state.removed !== null) cells[state.removed].append(ghost(previous.player));
    schedule();
  }
  function agentMove() {
    if (!network || !active || state.winner !== null) return;
    // the network is deterministic, so a random opening keeps agent-vs-agent games from repeating
    if (mode === "watch" && state.moves === 0) {
      const legal = legalMoves(state);
      play(legal[Math.floor(Math.random() * legal.length)], `${LABEL} (random opening)`);
    } else {
      const q = values();
      if (q) play(chooseMove(state, q), LABEL);
    }
  }
  function schedule() {
    stop();
    if (!network || !active || state.winner !== null || !values()) return;
    if (mode === "human" && state.player !== human || mode === "watch" && autoplay) {
      timer = setTimeout(agentMove, mode === "watch" ? 1800 - Number($("speed").value) : 550);
    }
  }
  function render() {
    const board = boardOf(state);
    const q = values();
    const winning = typeof state.winner === "number" ? LINES.find((line) => line.every((cell) => board[cell] === state.winner)) : [];
    cells.forEach((button, cell) => {
      const player = board[cell];
      const showValue = $("thinking").checked && player === null && state.winner === null && q;
      button.disabled = !network || !active || mode !== "human" || state.player !== human || state.winner !== null || player !== null;
      button.className = `fifo-cell ${player === 0 ? "fifo-x" : player === 1 ? "fifo-o" : ""} ${winning?.includes(cell) ? "fifo-winning" : ""} ${cell === hint ? "fifo-hinted" : ""}`;
      const html = player !== null ? pieceHtml(state, player, cell) : showValue ? `<span class="fifo-q">${round(q[cell]).toFixed(2)}</span>` : `<span class="fifo-empty" aria-hidden="true">${cell === hint ? "↗" : ""}</span>`;
      if (shownCells[cell] !== html) { button.innerHTML = html; shownCells[cell] = html; }
      button.setAttribute("aria-label", `${position(cell)}, ${player === null ? `empty${showValue ? `, score ${q[cell].toFixed(3)}` : ""}` : pieceLabel(state, player, cell)}${cell === hint ? ", suggested move" : ""}`);
    });
    $("status").textContent = !network ? "Loading the Deep Q agent…" : state.winner === "draw" ? `Draw: ${MAX_MOVES}-move limit reached.` : state.winner !== null ? `${state.winner === 0 ? "X" : "O"} wins with four in a row!` : mode === "watch" ? `${LABEL} (${state.player === 0 ? "X" : "O"}) to move.${autoplay ? "" : " Press Step or Play."}` : state.player === human ? `Your turn. You’re ${human === 0 ? "X" : "O"}.` : `${LABEL} is thinking…`;
    const last = history.at(-1);
    $("removal").textContent = hint !== null ? `Suggested move: ${position(hint)}.` : last?.removed != null ? `${last.previous.player === 0 ? "X" : "O"} placed ${position(last.cell)}; oldest mark at ${position(last.removed)} disappeared.` : "Make a row, column, or full diagonal of four.";
    const best = q ? Math.max(...legalMoves(state).map((cell) => q[cell])) : 0;
    $("confidence").textContent = !network || state.winner !== null ? "" : !q ? "Evaluating moves…" : `Best square for ${state.player === 0 ? "X" : "O"}: ${signed(best)} (higher is better)`;
    $("human-controls").hidden = mode !== "human";
    $("watch-controls").hidden = mode !== "watch";
    $("human-mode").setAttribute("aria-pressed", mode === "human");
    $("watch-mode").setAttribute("aria-pressed", mode === "watch");
    $("undo").hidden = $("hint").hidden = mode !== "human";
    $("step").hidden = $("auto").hidden = mode !== "watch";
    $("undo").disabled = !history.some((entry) => entry.who === "You");
    $("hint").disabled = !q || state.winner !== null || state.player !== human;
    $("step").disabled = !q || autoplay || state.winner !== null;
    $("auto").disabled = !network;
    $("auto").textContent = autoplay ? "Pause" : "Play";
    for (let d = 0; d < 4; d++) $(`depth-${d}`).setAttribute("aria-pressed", d === depth);
    $("depth-note").textContent = DEPTH_NOTES[depth];
    $("score-x").textContent = scores[0]; $("score-o").textContent = scores[1]; $("score-draw").textContent = scores[2];
    $("move-count").textContent = `${state.moves}/${MAX_MOVES}`;
    if (loggedHistory !== history || $("log").children.length !== history.length) {
    const entries = loggedHistory === history ? history.slice($("log").children.length) : history;
    const nodes = entries.map((entry) => {
      const li = document.createElement("li");
      li.textContent = `${entry.who} (${entry.previous.player === 0 ? "X" : "O"}): ${position(entry.cell)}${entry.removed === null ? "" : `; removed ${position(entry.removed)}`}${entry.who === LABEL ? `; score ${round(entry.value).toFixed(2)}` : ""}`;
      return li;
    });
    if (loggedHistory === history) $("log").append(...nodes);
    else $("log").replaceChildren(...nodes);
    $("log").scrollTop = $("log").scrollHeight;
    loggedHistory = history;
    }
  }
  function configure() { autoplay = false; scores = [0, 0, 0]; newGame(); }
  $("new").addEventListener("click", newGame);
  $("human-mode").addEventListener("click", () => { if (mode !== "human") { mode = "human"; configure(); } });
  $("watch-mode").addEventListener("click", () => { if (mode !== "watch") { mode = "watch"; configure(); } });
  $("side").addEventListener("change", () => { human = Number($("side").value); configure(); });
  $("thinking").addEventListener("change", render);
  for (let d = 0; d < 4; d++) $(`depth-${d}`).addEventListener("click", () => { depth = d; render(); schedule(); });
  $("speed").addEventListener("input", schedule);
  $("step").addEventListener("click", agentMove);
  $("auto").addEventListener("click", () => {
    autoplay = !autoplay;
    if (autoplay && state.winner !== null) newGame();
    else { render(); schedule(); }
  });
  $("reset-score").addEventListener("click", () => { scores = [0, 0, 0]; counted = false; render(); });
  // the network scores from the mover's side, so it can suggest your move too
  $("hint").addEventListener("click", () => { const q = values(); if (q) { hint = chooseMove(state, q); render(); } });
  $("undo").addEventListener("click", () => {
    const index = history.findLastIndex((entry) => entry.who === "You");
    if (index < 0) return;
    stop();
    if (counted) scores[state.winner === "draw" ? 2 : state.winner]--;
    state = history[index].previous; history = history.slice(0, index); counted = false; hint = null;
    render(); schedule();
  });
  async function load() {
    if (network || loading) return;
    loading = true; $("error").hidden = true; $("retry").hidden = true;
    try {
      network = await loader();
      search = createSearch(network);
      render(); schedule();
    } catch (error) {
      $("status").textContent = "The FIFO agent could not load.";
      $("error").hidden = false; $("error").textContent = `${error.message} Serve this app with python demo.py.`;
      $("retry").hidden = false;
    } finally { loading = false; }
  }
  $("retry").addEventListener("click", load);
  render();
  return {
    setActive(value) {
      active = value;
      if (!active) { autoplay = false; stop(); }
      render();
      if (active) { load(); schedule(); }
    },
  };
}
