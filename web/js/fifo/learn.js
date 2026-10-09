import { LINES, MAX_MOVES, boardOf, initialState, legalMoves, playMove } from "./game.js";
import { DEPTH, activations } from "./dqn.js";
import { createSearch } from "./search.js";
import { ghost, pieceHtml, pieceLabel } from "./piece.js";

// The rules steps. X moves first, so a position with X to move has equal move counts.
// `target` is the square to play; `reply` is an O move played by a button first.
export const LESSONS = [
  {
    label: "Marks", title: "Four marks each",
    start: { queues: [[0, 6, 9], [5, 10, 3]], player: 0, moves: 6 }, target: 15,
    before: "Each player keeps up to four marks. The dots under a mark show how many more turns it stays. Place X on the glowing square.",
    after: "X has four marks now.",
  },
  {
    label: "Oldest out", title: "A fifth mark removes the oldest",
    start: { queues: [[0, 6, 9, 15], [5, 10, 3, 12]], player: 0, moves: 8 }, target: 2,
    before: "With four marks down, a new mark removes your oldest: the faded one. Place X.",
    after: "X’s oldest mark is gone. You only ever lose your own oldest.",
  },
  {
    label: "Win", title: "Four in a line wins",
    start: { queues: [[15, 4, 5, 6], [0, 9, 10, 13]], player: 0, moves: 8 }, target: 7,
    before: "Complete the row. It counts because X’s oldest mark is somewhere else.",
    after: "X wins. If the oldest mark were in the row, it would vanish and the row wouldn’t count.",
  },
  {
    label: "Trap", title: "The aging trap",
    start: { queues: [[12, 4, 5, 6], [7, 1, 10, 14]], player: 1, moves: 8 }, reply: 2, target: 7,
    before: "O blocks X’s row with its oldest mark (faded). Play O’s move and watch the block vanish.",
    middle: "O can’t move back there. Finish the row.",
    after: "That’s the aging trap. It can’t be defended, so the agent hunts for it.",
  },
];

// The position the model steps look at: X to move, O threatens the third row.
export const QUIZ = { queues: [[13, 0, 5, 6], [3, 8, 9, 10]], player: 0, moves: 8, target: 11 };

const MODEL_STEPS = [
  { label: "Inputs", title: "What the network sees", text: "Eight grids track marks and their ages. Hover over a mark to find it below." },
  { label: "Network", title: "Inside the network", text: "129 inputs → two hidden layers → one score per square. Darker blue means a stronger signal." },
  { label: "Look-ahead", title: `Look ${DEPTH} moves ahead`, text: "" },
  { label: "Training", title: "How it learned", text: "" },
];
const STEPS = [...LESSONS, ...MODEL_STEPS];

const position = (cell) => `r${Math.floor(cell / 4) + 1} c${cell % 4 + 1}`;
const signed = (value) => `${value > 0.005 ? "+" : value < -0.005 ? "−" : ""}${Math.abs(value).toFixed(2)}`;
const heat = (v) => `color-mix(in srgb, var(${v >= 0 ? "--win" : "--loss"}) ${Math.round(Math.min(1, Math.abs(v)) * 45)}%, var(--surface))`;
const PLANE_NAMES = ["leaves next turn", "in 2 turns", "in 3 turns", "in 4 turns"];

export function startState(lesson) {
  return { ...initialState(), ...lesson.start, queues: lesson.start.queues.map((queue) => [...queue]) };
}

export function quizState() {
  return { ...initialState(), queues: QUIZ.queues.map((queue) => [...queue]), player: QUIZ.player, moves: QUIZ.moves };
}

// How many positions a search of `depth` moves visits from here.
export function searchSize(state, depth) {
  if (depth === 0) return 1;
  return legalMoves(state).reduce((total, cell) => {
    const next = playMove(state, cell);
    return total + (next.winner === null ? searchSize(next, depth - 1) : 1);
  }, 0);
}

// 8 planes of 16 squares (mover's marks then opponent's, by turns left), as in encode() in dqn.js
function planes(state) {
  const grid = Array.from({ length: 8 }, () => Array(16).fill(false));
  [state.player, 1 - state.player].forEach((side, s) => {
    const queue = state.queues[side];
    queue.forEach((cell, i) => { grid[s * 4 + 4 - queue.length + i][cell] = true; });
  });
  return grid;
}

function planeGrid(cells, mark, extra = "") {
  return `<div class="plane ${extra}">${cells.map((on, cell) => `<i data-cell="${cell}" class="${on ? mark : ""}"></i>`).join("")}</div>`;
}

function inputsView(state) {
  const grid = planes(state);
  const who = [state.player, 1 - state.player];
  const row = (s) => `<div class="plane-row"><span class="plane-who"><b class="${who[s] ? "fifo-o" : "fifo-x"}">${who[s] ? "O" : "X"}</b>${s ? "" : " to move"}</span>${
    grid.slice(s * 4, s * 4 + 4).map((cells, k) => `<figure>${planeGrid(cells, who[s] ? "o" : "x")}<figcaption>${PLANE_NAMES[k]}</figcaption></figure>`).join("")}</div>`;
  const left = MAX_MOVES - state.moves;
  return `<div class="planes">${row(0)}${row(1)}<div class="plane-left"><span>Extra input</span><div class="meter"><span style="width:${left / MAX_MOVES * 100}%"></span></div><b>${left} ÷ ${MAX_MOVES} = ${(left / MAX_MOVES).toFixed(3)}</b></div><p class="small">2 players × 4 ages × 16 squares = <b>128</b>. Add moves remaining as a fraction: <b>129 inputs</b>.</p></div>`;
}

function wires() {
  const ends = [8, 22, 36, 50, 64, 78, 92];
  return `<svg class="ann-wires" viewBox="0 0 40 100" preserveAspectRatio="none" aria-hidden="true">${
    ends.flatMap((a) => ends.map((b) => `<line x1="0" y1="${a}" x2="40" y2="${b}"/>`)).join("")}</svg>`;
}

function networkView(network, state) {
  const [, first, second, out] = activations(network, state);
  const grid = planes(state);
  const neurons = (layer) => {
    const top = Math.max(...layer) || 1;
    return `<div class="ann-grid">${Array.from(layer, (a) => `<i style="--a:${Math.round((a / top) * 100)}%"></i>`).join("")}</div>`;
  };
  const allowed = new Set(legalMoves(state));
  const parameters = network.layers.reduce((total, layer) => total + layer.in * layer.out + layer.out, 0);
  return `<div class="ann" role="img" aria-label="The network: 129 inputs, two layers of 256 neurons, 16 scores">
      <figure><div class="ann-inputs">${grid.map((cells, k) => planeGrid(cells, k < 4 ? (state.player ? "o" : "x") : (state.player ? "x" : "o"), "tiny")).join("")}</div><div class="ann-extra">+ 1 moves-left input: <b>${((MAX_MOVES - state.moves) / MAX_MOVES).toFixed(3)}</b></div><figcaption><b>128 + 1 = 129</b> inputs</figcaption></figure>
      ${wires()}
      <figure>${neurons(first)}<figcaption><b>256</b> neurons</figcaption></figure>
      ${wires()}
      <figure>${neurons(second)}<figcaption><b>256</b> neurons</figcaption></figure>
      ${wires()}
      <figure><div class="ann-out">${Array.from(out, (q, cell) => allowed.has(cell) ? `<i style="--heat:${heat(q)}">${signed(q)}</i>` : "<i class=\"taken\"></i>").join("")}</div><figcaption><b>16</b> scores</figcaption></figure>
    </div>
    <p class="small ann-caption">${parameters.toLocaleString()} learned weights and biases. Higher scores are better; occupied squares are blank.</p>`;
}

function lookaheadView() {
  return `<p class="search-summary">Try X → best O reply → best X response</p><p class="small">The network scores the future boards. Higher is better for X.</p>`;
}

function trainingView(data) {
  const games = data?.episodes ?? 0;
  const curve = data?.curve ?? [];
  const rate = (p) => p.wins / Math.max(1, p.wins + p.draws + p.losses);
  const x = (episodes) => 48 + episodes / Math.max(1, games) * 432;
  const y = (value) => 172 - value * 140;
  const compact = (n) => n >= 1e6 ? (n / 1e6).toLocaleString(undefined, { maximumFractionDigits: 1 }) + "m" : n >= 1000 ? (n / 1000) + "k" : n;
  const points = curve.map((p) => x(p.episodes) + "," + y(rate(p))).join(" ");
  return `<div class="training-layout"><ol class="loop">
      <li><b>Play itself</b><span>One network plays both sides.</span></li>
      <li><b>Score the result</b><span>Win +1 · loss −1 · draw 0</span></li>
      <li><b>Update the network</b><span>Learn from the moves played.</span></li>
      <li><b>Repeat</b><span>Improve over many games.</span></li>
    </ol>
    ${curve.length > 1 ? `<figure class="mini-curve"><figcaption><b>Win rate during training</b><span>Network alone vs tactical opponent</span></figcaption>
      <svg viewBox="0 0 510 214" role="img" aria-label="Win rate from 0 to 100 percent over ${games.toLocaleString()} training games">
        ${[0, 0.5, 1].map((v) => `<path class="chart-guide" d="M48 ${y(v)}H480"/><text x="38" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`).join("")}
        <path class="axis" d="M48 32V172H480"/>
        ${[0, 0.25, 0.5, 0.75, 1].map((v) => `<text x="${x(games * v)}" y="191" text-anchor="middle">${compact(games * v)}</text>`).join("")}
        <text x="264" y="210" text-anchor="middle">Training games</text>
        <polyline points="${points}"/>
        ${curve.map((p) => `<circle cx="${x(p.episodes)}" cy="${y(rate(p))}" r="3"><title>${p.episodes.toLocaleString()} games: ${(rate(p) * 100).toFixed(1)}% wins</title></circle>`).join("")}
      </svg></figure>` : '<p class="small">Training chart unavailable.</p>'}</div>`;
}

// loader() resolves to the trained network; onFinish runs after the last step.
export function initFIFOLearn(root, loader, onFinish = () => {}) {
  root.innerHTML = `
    <div class="learn-head"><div><h2>How FIFO works</h2><p class="small">The rules, the winning trick, and the agent’s brain, one step at a time.</p></div></div>
    <div class="card lab">
      <ol class="stepper" aria-label="Steps">${STEPS.map((s, i) => `<li><button type="button" data-fifo-step="${i}"><i>${i + 1}</i><span>${s.label}</span></button></li>`).join("")}</ol>
      <div class="lab-main" id="fifo-learn-main">
        <div class="lab-board" id="fifo-learn-left">
          <div id="fifo-learn-board" class="fifo-mini" role="group" aria-label="Example 4 by 4 board"></div>
          <p id="fifo-learn-note" class="who"></p>
        </div>
        <div class="lab-story">
          <div aria-live="polite" aria-atomic="true"><h3 id="fifo-learn-title"></h3><p id="fifo-learn-text" class="small"></p></div>
          <div id="fifo-learn-visual" class="learn-visual"></div>
          <div class="actions">
            <button id="fifo-learn-reply" class="button" type="button" hidden>Play O’s move</button>
            <button id="fifo-learn-retry" class="link-button" type="button" hidden>Try again</button>
          </div>
          <div class="learn-navigation">
            <button id="fifo-learn-back" class="button" type="button">Back</button>
            <button id="fifo-learn-next" class="button primary" type="button">Next</button>
          </div>
        </div>
      </div>
    </div>`;
  const $ = (id) => root.querySelector(`#fifo-learn-${id}`);
  const steps = [...root.querySelectorAll("[data-fifo-step]")];
  let step = 0;
  let view = null;
  let network = null;
  let results = null;
  let search = null;
  let quizScores = null;
  let scoring = false;

  const cells = Array.from({ length: 16 }, (_, cell) => {
    const button = document.createElement("button");
    button.type = "button";
    button.addEventListener("click", () => play(cell));
    // on the inputs step, pointing at a square lights it up in every input grid
    const mark = (on) => root.querySelectorAll(`.planes [data-cell="${cell}"]`).forEach((i) => i.classList.toggle("hl", on));
    button.addEventListener("pointerenter", () => mark(true));
    button.addEventListener("pointerleave", () => mark(false));
    button.addEventListener("focus", () => mark(true));
    button.addEventListener("blur", () => mark(false));
    $("board").append(button);
    return button;
  });

  const isRules = () => step < LESSONS.length;
  const isQuiz = () => STEPS[step].label === "Look-ahead";

  function reset() {
    view = { state: isRules() ? startState(LESSONS[step]) : quizState(), phase: "start", message: null };
    render();
  }

  function play(cell) {
    if (!(isRules() || isQuiz()) || view.phase === "done" || view.state.queues.flat().includes(cell)) return;
    const target = isRules() ? LESSONS[step].target : QUIZ.target;
    if (isQuiz() && cell !== target) {
      view.message = `Not ${position(cell)}: O would finish its row at ${position(target)} next turn.`;
      render();
      return;
    }
    const mover = view.state.player;
    view.state = playMove(view.state, cell);
    view.phase = "done";
    view.message = null;
    render();
    cells[cell].querySelector("svg")?.classList.add("drawn");
    if (view.state.removed !== null) cells[view.state.removed].append(ghost(mover));
  }

  function render() {
    const rules = isRules();
    const lesson = rules ? LESSONS[step] : null;
    const quiz = isQuiz();
    const { state, phase } = view;
    const board = boardOf(state);
    const line = typeof state.winner === "number" ? LINES.find((cells) => cells.every((cell) => board[cell] === state.winner)) ?? [] : [];
    if (quiz && network && !quizScores && !scoring) {
      scoring = true;
      Promise.resolve(search.evaluate(quizState(), DEPTH)).then((scores) => {
        quizScores = scores;
        if (isQuiz()) render();
      }).catch(() => {
        if (isQuiz()) $("note").textContent = "Scores unavailable. You can still try a move.";
      });
    }
    const scores = quiz && phase !== "done" ? quizScores : null;
    const waiting = lesson?.reply !== undefined && phase === "start";
    const target = rules ? lesson.target : QUIZ.target;
    const interactive = rules || quiz;
    cells.forEach((button, cell) => {
      const player = board[cell];
      const playable = interactive && phase !== "done" && !waiting && player === null && (quiz || cell === target);
      button.disabled = !playable && !(STEPS[step].label === "Inputs" && player !== null);
      button.className = [
        "fifo-cell",
        player === 0 ? "fifo-x" : player === 1 ? "fifo-o" : "",
        playable && rules ? "fifo-target" : "",
        line.includes(cell) ? "fifo-winning" : "",
        quiz && phase === "done" && cell === target ? "fifo-target-done" : "",
      ].join(" ");
      button.style.setProperty("--heat", scores && player === null ? heat(scores[cell]) : "");
      button.innerHTML = player !== null
        ? pieceHtml(state, player, cell)
        : scores ? `<span class="fifo-score">${signed(scores[cell])}</span>` : "";
      button.setAttribute("aria-label", `${position(cell)}, ${player === null ? `empty${scores ? `, score ${signed(scores[cell])}` : ""}${playable && rules ? ", play here" : ""}` : pieceLabel(state, player, cell)}`);
    });
    steps.forEach((button, i) => {
      button.parentElement.classList.toggle("done", i < step);
      if (i === step) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });

    const label = STEPS[step].label;
    $("title").textContent = STEPS[step].title;
    let text = STEPS[step].text;
    if (rules) text = view.message ?? (phase === "done" ? lesson.after : phase === "replied" ? lesson.middle : lesson.before);
    if (quiz) {
      text = view.message ?? (phase === "done"
        ? `Right. Blocking at ${position(target)} stops O’s immediate win.`
        : `It tries three moves before choosing one. Where would you play to stop O’s row?`);
    }
    if (label === "Training") {
      text = `It played itself ${results ? `${results.episodes.toLocaleString()} games${results.seconds ? ` in ${Math.round(results.seconds / 60)} minutes on a GPU` : ""}` : "for millions of games"}. It improves by learning from the results.`;
    }
    $("text").textContent = text;
    $("visual").innerHTML = label === "Inputs" ? inputsView(state)
      : label === "Network" ? (network ? networkView(network, state) : "<p class=\"small\">Loading the network…</p>")
      : label === "Look-ahead" ? lookaheadView()
      : label === "Training" ? trainingView(results)
      : "";
    $("left").hidden = label === "Training";
    $("main").classList.toggle("wide", label === "Training");
    $("main").classList.toggle("network-layout", label === "Network");
    $("note").innerHTML = quiz && !quizScores ? "Loading the agent’s scores…"
      : quiz && phase !== "done" ? "−0.90 = O wins next turn · higher is better"
      : `<b class="fifo-x">X</b> moves first · dots = turns left · faded = next to go`;
    $("reply").hidden = !waiting;
    $("retry").hidden = !interactive || phase === "start";
    $("back").disabled = step === 0;
    $("next").textContent = step === STEPS.length - 1 ? "Play the agent" : "Next";
  }

  function go(next) {
    if (next >= STEPS.length) { onFinish(); return; }
    step = Math.max(0, next);
    if (STEPS[step].label === "Training" && !results) {
      fetch("data/fifo-evaluation.json").then((r) => (r.ok ? r.json() : null)).then((data) => { results = data; if (STEPS[step].label === "Training") render(); }).catch(() => {});
    }
    reset();
  }

  $("reply").addEventListener("click", () => {
    const lesson = LESSONS[step];
    view.state = playMove(view.state, lesson.reply);
    view.phase = "replied";
    render();
    cells[lesson.reply].querySelector("svg")?.classList.add("drawn");
    cells[view.state.removed].append(ghost(1));
  });
  $("retry").addEventListener("click", reset);
  $("back").addEventListener("click", () => go(step - 1));
  $("next").addEventListener("click", () => go(step + 1));
  steps.forEach((button, i) => button.addEventListener("click", () => go(i)));
  reset();
  loader().then((loaded) => { network = loaded; search = createSearch(network); render(); }).catch(() => {
    $("note").textContent = "The agent couldn’t load.";
  });
  return { get step() { return step; } };
}
