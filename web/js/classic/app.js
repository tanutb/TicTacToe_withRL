import { EMPTY, chooseAction, move, result, turn, winningLine } from "./game.js";
import { START, initLearning } from "./learning.js";
import { markSvg } from "../shared/marks.js";
import { initTabs } from "../shared/tabs.js";
import { initTheme } from "../shared/theme.js";

const NAMES = { QLearning: "Q-Learning", SARSA: "SARSA", DoubleQLearning: "Double Q-Learning" };
const SHORT_NAMES = { QLearning: "Q-Learning", SARSA: "SARSA", DoubleQLearning: "Double Q" };
const ABOUT = {
  QLearning: "Learns from the best move it could make next.",
  SARSA: "Learns from the move it actually makes next, so it plays a bit more carefully.",
  DoubleQLearning: "Keeps two tables, one picks a move and the other judges it.",
};
const CELL_NAMES = ["top left", "top", "top right", "left", "centre", "right", "bottom left", "bottom", "bottom right"];
const MARK = { 1: "X", 2: "O" };
const SPEEDS = [1400, 1000, 650, 350, 120]; // ms per move in agent vs agent


const $ = (sel) => document.querySelector(sel);
const policies = {};
const game = {
  mode: "play", // "play" = you vs agent, "watch" = agent vs agent
  human: "1",
  opponent: "QLearning",
  watch: { 1: "QLearning", 2: "SARSA" },
  board: EMPTY,
  log: [],
  counted: null, // score key already added for the finished game
  busy: false,
  autoplay: false,
  hint: null,
  ready: false,
};
let timer = null;
let round = 0; // bumped on every new game / undo, so old timers know to stop

// ---------- small helpers ----------

function store(key, value) {
  try {
    if (value === undefined) return JSON.parse(localStorage.getItem(key));
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    return null;
  }
}

// Double Q plays on Q1 + Q2. Show the average so every algorithm uses the same -1..+1 scale.
function displayQ(algorithm, value) {
  if (value == null) return value;
  return algorithm === "DoubleQLearning" ? value / 2 : value;
}

function agentFor(player) {
  if (game.mode === "watch") return game.watch[player];
  return player === game.human ? null : game.opponent;
}

function whoName(player) {
  const agent = agentFor(player);
  return agent ? NAMES[agent] : "You";
}

// ---------- score ----------

const scores = store("ttt-scores") ?? {};

function scoreKey() {
  return game.mode === "play" ? `${game.opponent}-${game.human}` : `${game.watch[1]}-vs-${game.watch[2]}`;
}

function currentScore() {
  return (scores[scoreKey()] ??= { a: 0, d: 0, b: 0 });
}

function countResult(outcome) {
  const s = currentScore();
  let key;
  if (outcome === "draw") key = "d";
  else if (game.mode === "watch") key = outcome === "1" ? "a" : "b";
  else key = outcome === game.human ? "a" : "b";
  s[key]++;
  game.counted = key;
  store("ttt-scores", scores);
}

// ---------- game flow ----------

function newGame() {
  round++;
  clearTimeout(timer);
  game.board = EMPTY;
  game.log = [];
  game.counted = null;
  game.busy = false;
  game.hint = null;
  $("#strike").replaceChildren();
  clearMarks();
  // greedy agents always open the same way, a random first move makes every game different
  const randomStart = game.mode === "watch" ? $("#random-opening").checked : game.human === "2" && $("#random-start").checked;
  if (randomStart) {
    playCell(Math.floor(Math.random() * 9), "random");
  }
  nextTurn();
}

function playCell(cell, who) {
  const player = turn(game.board);
  const entry = policies[agentFor(player)]?.states[game.board];
  game.board = move(game.board, cell);
  const value = who === "agent" ? displayQ(agentFor(player), entry?.values[cell]) : null;
  game.log.push({ cell, player, who, value });
  game.hint = null;
}

function nextTurn() {
  const outcome = result(game.board);
  if (outcome) {
    if (!game.counted) countResult(outcome);
    if (game.mode === "watch" && game.autoplay) {
      game.busy = true;
      const r = round;
      timer = setTimeout(() => r === round && newGame(), Math.max(900, SPEEDS[speed()] * 2));
    }
    render();
    return;
  }

  const agent = agentFor(turn(game.board));
  if (!agent || !game.ready || (game.mode === "watch" && !game.autoplay)) {
    game.busy = false;
    render();
    return;
  }

  // agent's turn: wait a moment so you can see it "think"
  game.busy = true;
  render();
  const delay = game.mode === "watch" ? SPEEDS[speed()] : $("#thinking").checked ? 1100 : 420;
  const r = round;
  timer = setTimeout(() => {
    if (r !== round) return;
    try {
      playCell(chooseAction(game.board, policies[agent]), "agent");
    } catch (err) {
      showError(err.message);
      return;
    }
    nextTurn();
  }, delay);
}

function step() {
  if (game.mode !== "watch" || !game.ready) return;
  if (result(game.board)) return newGame();
  if (game.busy) return;
  const agent = agentFor(turn(game.board));
  playCell(chooseAction(game.board, policies[agent]), "agent");
  nextTurn();
}

function clickCell(cell) {
  if (game.mode !== "play" || game.busy || !game.ready) return;
  if (result(game.board) || turn(game.board) !== game.human || game.board[cell] !== "0") return;
  playCell(cell, "you");
  nextTurn();
}

function undo() {
  if (game.mode !== "play") return;
  const last = game.log.findLastIndex((m) => m.who === "you");
  if (last < 0) return;
  round++;
  clearTimeout(timer);
  if (game.counted) {
    currentScore()[game.counted]--;
    store("ttt-scores", scores);
    game.counted = null;
  }
  game.log = game.log.slice(0, last);
  game.board = game.log.reduce((board, m) => move(board, m.cell), EMPTY);
  game.busy = false;
  game.hint = null;
  $("#strike").replaceChildren();
  clearMarks();
  nextTurn();
}

function hint() {
  if (game.mode !== "play" || game.busy || !game.ready) return;
  if (result(game.board) || turn(game.board) !== game.human) return;
  // the same algorithm also trained an agent for your side, ask that one
  game.hint = chooseAction(game.board, policies[game.opponent]);
  render();
  const r = round;
  setTimeout(() => {
    if (r === round && game.hint !== null) {
      game.hint = null;
      render();
    }
  }, 1900);
}

function speed() {
  return Number($("#speed").value);
}

// ---------- rendering ----------

const cells = [];
for (let i = 0; i < 9; i++) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "cell";
  button.innerHTML = `<span class="mark"></span><span class="note"></span>`;
  button.addEventListener("click", () => clickCell(i));
  $("#board").append(button);
  cells.push(button);
}
// what each cell shows right now: "" empty, "ghost-1"/"ghost-2" hover preview, "1"/"2" a mark.
// Cells are only touched when this changes, so marks never get redrawn mid animation.
const shown = Array(9).fill("");

function clearMarks() {
  shown.fill("");
  cells.forEach((c) => (c.querySelector(".mark").innerHTML = ""));
}

function render() {
  const board = game.board;
  const outcome = result(board);
  const player = turn(board);
  const humanTurn = game.mode === "play" && !outcome && player === game.human && !game.busy && game.ready;

  // thinking overlay: Q-values for whoever moves next
  const showThinking = $("#thinking").checked && !outcome && game.ready;
  const thinker = game.mode === "play" ? game.opponent : agentFor(player);
  const entry = showThinking ? policies[thinker]?.states[board] : null;
  const values = entry ? entry.values.filter((v) => v !== null) : [];
  const best = entry ? entry.action : -1;

  cells.forEach((cell, i) => {
    const value = board[i];
    const mark = cell.querySelector(".mark");
    const want = value !== "0" ? value : humanTurn ? `ghost-${game.human}` : "";
    if (want !== shown[i]) {
      if (value !== "0") {
        // you vs agent: you write in ink, the agent in pencil. agent vs agent: X ink, O pencil
        const byYou = game.mode === "play" ? value === game.human : value === "1";
        mark.innerHTML = markSvg(value === "1", `${byYou ? "ink" : "pencil"} drawn`);
      } else {
        mark.innerHTML = humanTurn ? markSvg(game.human === "1", "ghost") : "";
      }
      shown[i] = want;
    }

    const note = cell.querySelector(".note");
    const q = displayQ(thinker, entry?.values[i]);
    if (q !== null && q !== undefined && value === "0") {
      note.textContent = (q > 0 ? "+" : "") + q.toFixed(2);
      // green for good cells, red for bad ones, grey when it's about even
      const strength = Math.round(35 + Math.min(1, Math.abs(q)) * 65);
      note.style.setProperty("--heat", `color-mix(in srgb, ${q >= 0 ? "var(--win)" : "var(--loss)"} ${strength}%, var(--muted))`);
      note.classList.add("show");
      note.classList.toggle("best", i === best && values.length > 1);
    } else {
      note.classList.remove("show", "best");
    }

    cell.classList.toggle("hint", game.hint === i);
    cell.disabled = !humanTurn || value !== "0";
    const content = value === "0" ? "empty" : MARK[value];
    cell.setAttribute("aria-label", `${CELL_NAMES[i]}, ${content}${q != null && value === "0" ? `, value ${q.toFixed(2)}` : ""}`);
  });

  const line = winningLine(board);
  if (line && !$("#strike").firstChild) drawStrike(line);

  renderStatus(outcome, player, entry);
  renderScore();
  renderLog();

  $("#undo").disabled = game.mode !== "play" || !game.log.some((m) => m.who === "you");
  $("#hint").disabled = !humanTurn;
  $("#autoplay").textContent = game.autoplay ? "Pause" : "Play";
  $("#step").disabled = game.busy && !outcome;
}

function drawStrike([a, , c]) {
  const center = (i) => [(i % 3) * 100 + 50, Math.floor(i / 3) * 100 + 50];
  const [x1, y1] = center(a);
  const [x2, y2] = center(c);
  const dx = (x2 - x1) * 0.18;
  const dy = (y2 - y1) * 0.18;
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("pathLength", "1");
  path.setAttribute("d", `M${x1 - dx} ${y1 - dy} L${x2 + dx} ${y2 + dy}`);
  $("#strike").append(path);
}

function renderStatus(outcome, player, entry) {
  const status = $("#status");
  status.classList.remove("thinking");
  let text;
  if (!game.ready) {
    text = "Loading the agents…";
  } else if (outcome === "draw") {
    text = game.mode === "play" ? "Draw. Against these agents that's a good result." : "Draw.";
  } else if (outcome) {
    if (game.mode === "watch") text = `${whoName(outcome)} wins as ${MARK[outcome]}.`;
    else text = outcome === game.human ? "You win! That's rare, nice one." : `${NAMES[game.opponent]} wins.`;
  } else if (game.mode === "watch" && !game.autoplay) {
    text = `${whoName(player)} (${MARK[player]}) to move. Press Step or Play.`;
  } else if (game.busy) {
    text = `${whoName(player)} is thinking`;
    status.classList.add("thinking");
  } else {
    text = game.hint !== null
      ? `Hint: ${NAMES[game.opponent]} would play ${game.hint === 4 ? "the centre" : CELL_NAMES[game.hint]}.`
      : `Your turn, you're ${MARK[game.human]}.`;
  }
  if (entry && !entry.seen && !outcome) text += " (it never saw this board in training)";
  status.textContent = text;
}

function renderScore() {
  const s = currentScore();
  $("#score-a").textContent = s.a;
  $("#score-d").textContent = s.d;
  $("#score-b").textContent = s.b;
  $("#score-a-label").textContent = game.mode === "play" ? "You" : `X, ${SHORT_NAMES[game.watch[1]]}`;
  $("#score-b-label").textContent = game.mode === "play" ? SHORT_NAMES[game.opponent] : `O, ${SHORT_NAMES[game.watch[2]]}`;
}

function renderLog() {
  const list = $("#log");
  list.replaceChildren();
  if (!game.log.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No moves yet.";
    list.append(li);
    return;
  }
  for (const m of game.log) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.className = m.who === "you" ? "you" : "agent";
    name.textContent = m.who === "you" ? "You" : whoName(m.player);
    if (m.who === "random") name.textContent += " (random)";
    li.append(name, ` played ${MARK[m.player]} ${m.cell === 4 ? "in the" : "at"} ${CELL_NAMES[m.cell]}`);
    if (m.value != null) li.append(` (Q = ${m.value.toFixed(2)})`);
    list.append(li);
  }
  list.scrollTop = list.scrollHeight;
}

function showError(message) {
  game.ready = false;
  game.busy = false;
  render();
  $("#status").textContent = message;
}

// ---------- controls ----------

function setMode(mode) {
  game.mode = mode;
  game.autoplay = false;
  $("#mode-play").setAttribute("aria-selected", mode === "play");
  $("#mode-watch").setAttribute("aria-selected", mode === "watch");
  $("#play-controls").hidden = mode !== "play";
  $("#watch-controls").hidden = mode !== "watch";
  $("#undo").hidden = $("#hint").hidden = mode !== "play";
  $("#step").hidden = $("#autoplay").hidden = mode !== "watch";
  newGame();
}

for (const [name, label] of Object.entries(NAMES)) {
  const choice = document.createElement("label");
  choice.className = "choice";
  choice.innerHTML = `<input type="radio" name="opponent" value="${name}"><span>${SHORT_NAMES[name]}</span>`;
  choice.querySelector("input").checked = name === game.opponent;
  $("#opponents").append(choice);
  for (const id of ["#watch-x", "#watch-o"]) $(id).append(new Option(label, name));
}
$("#watch-x").value = game.watch[1];
$("#watch-o").value = game.watch[2];
$("#opponent-about").textContent = ABOUT[game.opponent];

document.querySelectorAll('input[name="opponent"]').forEach((input) =>
  input.addEventListener("change", () => {
    game.opponent = input.value;
    $("#opponent-about").textContent = ABOUT[input.value];
    newGame();
  }),
);
document.querySelectorAll('input[name="side"]').forEach((input) =>
  input.addEventListener("change", () => {
    game.human = input.value;
    $("#random-start-row").hidden = game.human !== "2";
    newGame();
  }),
);
$("#watch-x").addEventListener("change", (e) => { game.watch[1] = e.target.value; newGame(); });
$("#watch-o").addEventListener("change", (e) => { game.watch[2] = e.target.value; newGame(); });
$("#mode-play").addEventListener("click", () => game.mode !== "play" && setMode("play"));
$("#mode-watch").addEventListener("click", () => game.mode !== "watch" && setMode("watch"));
$("#thinking").addEventListener("change", (e) => {
  $("#thinking-note").hidden = !e.target.checked;
  store("ttt-thinking", e.target.checked);
  render();
});
$("#new-game").addEventListener("click", newGame);
$("#undo").addEventListener("click", undo);
$("#hint").addEventListener("click", hint);
$("#step").addEventListener("click", step);
$("#autoplay").addEventListener("click", () => {
  game.autoplay = !game.autoplay;
  if (!game.autoplay) {
    round++;
    clearTimeout(timer);
    game.busy = false;
    render();
  } else if (result(game.board)) {
    newGame();
  } else {
    nextTurn();
  }
});
$("#reset-score").addEventListener("click", () => {
  scores[scoreKey()] = { a: 0, d: 0, b: 0 };
  store("ttt-scores", scores);
  renderScore();
});

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || tabs.active !== "play") return;
  if (e.target.matches("select, textarea, input[type=text], [role=tab]")) return;
  const key = e.key.toLowerCase();
  if (key >= "1" && key <= "9") clickCell(Number(key) - 1);
  else if (key === "n") newGame();
  else if (key === "u") undo();
  else if (key === "h") hint();
  else if (key === "s") step();
});

initTheme();

const tabs = initTabs(["play", "results", "learn"]);

// ---------- results section ----------

let evaluation = null;
const view = { opponent: "random", seat: "X", point: null };

function renderBars() {
  const rows = evaluation.summary.filter((r) => r.opponent === view.opponent && r.seat === view.seat);
  const box = $("#bars");
  if (!box.children.length) {
    for (const name of Object.keys(NAMES)) {
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `<span class="name">${NAMES[name]}</span>
        <div class="bar" role="img" data-name="${name}"><span class="win"></span><span class="draw"></span><span class="loss"></span></div>`;
      box.append(row);
    }
  }
  for (const r of rows) {
    const bar = box.querySelector(`[data-name="${r.algorithm}"]`);
    const pct = (v) => `${(v * 100).toFixed(1)}%`;
    bar.setAttribute("aria-label", `${NAMES[r.algorithm]}: ${pct(r.win_rate)} wins, ${pct(r.draw_rate)} draws, ${pct(r.loss_rate)} losses`);
    for (const key of ["win", "draw", "loss"]) {
      const span = bar.querySelector(`.${key}`);
      const v = r[`${key}_rate`];
      span.style.flex = `0 0 ${v * 100}%`;
      span.textContent = v >= 0.07 ? pct(v) : "";
      span.title = `${key}: ${pct(v)}`;
    }
  }
}

function curveData() {
  const points = [...new Set(evaluation.curves.map((c) => c.episodes))].sort((a, b) => a - b);
  const series = {};
  for (const name of Object.keys(NAMES)) {
    series[name] = points.map((p) => {
      const rows = evaluation.curves.filter((c) => c.algorithm === name && c.seat === view.seat && c.episodes === p);
      return rows.reduce((sum, c) => sum + c.win_rate, 0) / rows.length;
    });
  }
  return { points, series };
}

const W = 520, H = 240, PAD = { left: 40, right: 10, top: 10, bottom: 28 };

function renderCurve() {
  const { points, series } = curveData();
  const max = points.at(-1);
  const x = (p) => PAD.left + (p / max) * (W - PAD.left - PAD.right);
  const y = (v) => PAD.top + (1 - v) * (H - PAD.top - PAD.bottom);
  const svg = $("#curve");
  let html = '<g class="axis">';
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    html += `<line x1="${PAD.left}" x2="${W - PAD.right}" y1="${y(v)}" y2="${y(v)}"/>`;
    html += `<text x="${PAD.left - 8}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`;
  }
  for (const p of points.filter((_, i) => i % 2 === 0)) {
    html += `<text x="${x(p)}" y="${H - 10}" text-anchor="middle">${p ? `${p / 1000}k` : "0"}</text>`;
  }
  html += "</g>";
  for (const [name, values] of Object.entries(series)) {
    const d = values.map((v, i) => `${i ? "L" : "M"}${x(points[i]).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");
    html += `<path class="series" d="${d}" style="stroke: var(--c-${name})"/>`;
  }
  const i = view.point ?? points.length - 1;
  html += `<line class="cursor" x1="${x(points[i])}" x2="${x(points[i])}" y1="${PAD.top}" y2="${H - PAD.bottom}"/>`;
  for (const [name, values] of Object.entries(series)) {
    html += `<circle cx="${x(points[i])}" cy="${y(values[i])}" r="5" style="fill: var(--c-${name})"/>`;
  }
  svg.innerHTML = html;
  svg.setAttribute("aria-label", `Learning curve for the agent playing ${view.seat}. Use left and right arrows to move between checkpoints.`);

  const readout = $("#curve-readout");
  readout.innerHTML = `<b>After ${points[i].toLocaleString()} games</b>`;
  for (const [name, values] of Object.entries(series)) {
    const row = document.createElement("div");
    row.innerHTML = `<span><i style="background: var(--c-${name})"></i>${NAMES[name]}</span><span>${(values[i] * 100).toFixed(1)}%</span>`;
    readout.append(row);
  }
}

function renderFacts() {
  const avg = (name) => {
    const t = evaluation.timings.filter((r) => r.algorithm === name);
    return t.reduce((s, r) => s + r.seconds, 0) / t.length;
  };
  const rows = (opponent, seat) => evaluation.summary.filter((r) => r.opponent === opponent && r.seat === seat);
  const winRange = (opponent, seat) => {
    const rates = rows(opponent, seat).map((r) => r.win_rate * 100);
    return `${Math.min(...rates).toFixed(0)}–${Math.max(...rates).toFixed(0)}%`;
  };
  const perfectLosses = evaluation.summary.filter((r) => r.opponent === "minimax").reduce((s, r) => s + r.losses, 0);
  const draws = evaluation.head_to_head.filter((m) => m.outcome === "draw").length;
  const facts = [
    `Against the imperfect player they win ${winRange("imperfect", "X")} of games as X and ${winRange("imperfect", "O")} as O.`,
    perfectLosses === 0
      ? "Against the perfect player every game is a draw. Nobody can beat perfect play, so that's the best possible result."
      : `Against the perfect player they lost ${perfectLosses} games in total.`,
    `${draws} of ${evaluation.head_to_head.length} agent vs agent games ended in a draw.`,
    `Training ${evaluation.episodes.toLocaleString()} games takes about ${Object.keys(NAMES).map((n) => `${Math.round(avg(n))}s (${NAMES[n]})`).join(", ")} on a laptop CPU.`,
  ];
  $("#facts").replaceChildren(...facts.map((f) => Object.assign(document.createElement("li"), { textContent: f })));
}

function renderResults() {
  document.querySelectorAll("[data-opponent]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.opponent === view.opponent));
  document.querySelectorAll("[data-seat]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.seat === view.seat));
  renderBars();
  renderCurve();
}

document.querySelectorAll("[data-opponent]").forEach((b) =>
  b.addEventListener("click", () => { view.opponent = b.dataset.opponent; evaluation && renderResults(); }),
);
document.querySelectorAll("[data-seat]").forEach((b) =>
  b.addEventListener("click", () => { view.seat = b.dataset.seat; evaluation && renderResults(); }),
);

const curve = $("#curve");
curve.addEventListener("pointermove", (e) => {
  if (!evaluation) return;
  const box = curve.getBoundingClientRect();
  const fraction = ((e.clientX - box.left) / box.width * W - PAD.left) / (W - PAD.left - PAD.right);
  const count = new Set(evaluation.curves.map((c) => c.episodes)).size;
  const i = Math.max(0, Math.min(count - 1, Math.round(fraction * (count - 1))));
  if (i !== view.point) {
    view.point = i;
    renderCurve();
  }
});
curve.addEventListener("keydown", (e) => {
  if (!evaluation || !["ArrowLeft", "ArrowRight"].includes(e.key)) return;
  e.preventDefault();
  const count = new Set(evaluation.curves.map((c) => c.episodes)).size;
  const i = view.point ?? count - 1;
  view.point = Math.max(0, Math.min(count - 1, i + (e.key === "ArrowRight" ? 1 : -1)));
  renderCurve();
});

// ---------- start ----------

async function loadJson(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} returned ${res.status}`);
  return res.json();
}

// ---------- how it learns tab ----------

// the trained agent's real Q-values for the example board, once the policies have loaded
const learning = initLearning((algorithm) => {
  const policy = policies[algorithm];
  const entry = policy?.states[START];
  return entry ? { values: entry.values.map((v) => displayQ(algorithm, v)), boards: Object.keys(policy.states).length } : null;
});

async function start() {
  if (store("ttt-thinking")) {
    $("#thinking").checked = true;
    $("#thinking-note").hidden = false;
  }
  render();
  try {
    await Promise.all(Object.keys(NAMES).map(async (name) => (policies[name] = await loadJson(`data/${name}.json`))));
    game.ready = true;
    newGame();
    learning.render();
  } catch (err) {
    showError(`Couldn't load the agents (${err.message}). Start the demo with "python demo.py", opening index.html directly won't work.`);
  }
  try {
    evaluation = await loadJson("data/evaluation.json");
    renderResults();
    renderFacts();
  } catch (err) {
    $("#results-error").hidden = false;
    $("#results-error").textContent = `Couldn't load the results (${err.message}). Run "python evaluate.py" to create them.`;
  }
}

start();
