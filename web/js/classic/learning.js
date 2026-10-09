import { legalMoves, move, result, winningLine } from "./game.js";

// Small, illustrative scores: teaching examples, not trained policy data.
export const START = "110220000";
export const VALUES = { 2: 0.6, 5: 0.2, 6: 0.1, 7: 0.05, 8: 0.15 };
// Double Q's second table, with its own guesses
export const VALUES_2 = { 2: 0.5, 5: 0.3, 6: 0, 7: 0.1, 8: 0.05 };
// If X blocks on the right, O replies bottom right and X moves again on this board.
// What the agent already believes about that next turn (Q₁, and Q₂ for Double Q).
export const NEXT_BOARD = "110221002";
export const NEXT = { 2: 0.9, 6: 0.1, 7: 0 };
export const NEXT_2 = { 2: 0.6, 6: 0.2, 7: -0.1 };
export const ALPHA = 0.1;
export const GAMMA = 0.8;

export function best(values) {
  const moves = Object.keys(values).map(Number);
  return moves.reduce((top, m) => (values[m] > values[top] ? m : top));
}

// ε-greedy: a random legal move with chance epsilon, otherwise the highest score
export function pickMove(values, epsilon, random = Math.random) {
  const moves = Object.keys(values).map(Number);
  if (random() < epsilon) return moves[Math.floor(random() * moves.length)];
  return best(values);
}

export function average(a, b) {
  return Object.fromEntries(Object.keys(a).map((cell) => [cell, (a[cell] + b[cell]) / 2]));
}

// One learning step for the move X plays on START.
// tables: [Q] or [Q₁, Q₂]. learner: the table Double Q updates this time (0 or 1).
// nextMove: the move SARSA really plays on its next turn (its best one unless it explores).
export function example(action, algorithm = "QLearning", { tables = [VALUES, VALUES_2], learner = 0, nextMove } = {}) {
  const afterMove = move(START, action);
  const replies = legalMoves(afterMove);
  // O takes a win when it has one, otherwise it misses X's top row
  const reply = replies.find((cell) => result(move(afterMove, cell)) === "2") ?? replies.at(-1);
  const afterReply = reply === undefined ? afterMove : move(afterMove, reply);
  const outcome = result(afterReply);
  const reward = outcome === "1" ? 1 : outcome === "2" ? -1 : 0;
  const table = algorithm === "DoubleQLearning" ? learner : 0;
  let pick = null;
  let future = 0;
  if (!outcome) {
    if (algorithm === "DoubleQLearning") {
      // the learning table picks the next move, the other table scores it
      const next = [NEXT, NEXT_2];
      pick = best(next[table]);
      future = next[1 - table][pick];
    } else {
      pick = algorithm === "SARSA" && nextMove !== undefined ? nextMove : best(NEXT);
      future = NEXT[pick];
    }
  }
  const old = tables[table][action];
  const target = reward + GAMMA * future;
  return { afterMove, afterReply, reply, outcome, reward, table, pick, future, old, target, updated: old + ALPHA * (target - old) };
}

const CELLS = ["top left", "top", "top right", "left", "centre", "right", "bottom left", "bottom", "bottom right"];
const NOTES = {
  QLearning: "Q-Learning learns from the <b>best</b> next move, even one it won’t play.",
  SARSA: "SARSA learns from the next move it <b>really plays</b>, random ones too, so it plays a bit safer.",
  DoubleQLearning: "Double Q keeps <b>two tables</b>. One picks the next move, the other scores it, so it’s less over-hopeful.",
};
const FAST_FORWARD = 50;
const EXPLORE = 0.3;
const SUB = ["₁", "₂"];

const fmt = (v) => `${v < 0 ? "−" : v > 0 ? "+" : ""}${Math.abs(v).toFixed(2)}`;
const short = (v) => `${v < 0 ? "−" : ""}${Math.abs(v).toFixed(2).replace(/^0/, "")}`;
const heat = (v) => `color-mix(in srgb, var(${v >= 0 ? "--win" : "--loss"}) ${Math.round(Math.min(1, Math.abs(v)) * 42)}%, var(--surface))`;
const pct = (v) => `${((Math.max(-1, Math.min(1, v)) + 1) / 2) * 100}%`;
const capital = (text) => text[0].toUpperCase() + text.slice(1);

// trainedRow(algorithm) -> { values: [9], boards } from the real policy, or null before it loads
export function initLearning(trainedRow = () => null) {
  const $ = (id) => document.getElementById(id);
  const fresh = () => [{ ...VALUES }, { ...VALUES_2 }];
  const state = { step: 0, action: null, algorithm: "QLearning", tables: fresh(), games: 0, lesson: null, last: null, timer: null };
  const double = () => state.algorithm === "DoubleQLearning";
  // the scores the agent acts on: Double Q plays on the average of its two tables
  const scores = () => (double() ? average(...state.tables) : state.tables[0]);
  const steps = [...document.querySelectorAll("[data-learn-step]")];

  const cells = CELLS.map((name, i) => {
    const button = document.createElement("button");
    button.type = "button";
    button.addEventListener("click", () => choose(i));
    button.addEventListener("pointerenter", () => highlight(i));
    button.addEventListener("pointerleave", () => highlight(null));
    $("learn-board").append(button);
    return button;
  });

  // Q-table: rows are boards, columns are squares
  const heads = CELLS.map((name, i) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.title = name;
    th.innerHTML = `<span class="pos" role="img" aria-label="${name}">${CELLS.map((_, j) => `<i class="${i === j ? "on" : ""}"></i>`).join("")}</span>`;
    th.classList.toggle("taken", START[i] !== "0");
    $("qtable-head").append(th);
    return th;
  });
  function makeRow(clickable) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.scope = "row";
    tr.append(th);
    const items = CELLS.map((name, i) => {
      const td = document.createElement("td");
      const el = document.createElement(clickable ? "button" : "span");
      if (clickable) {
        el.type = "button";
        el.addEventListener("click", () => choose(i));
      }
      td.append(el);
      tr.append(td);
      return { td, el };
    });
    $("qtable-body").append(tr);
    return { tr, th, items };
  }
  const rows = { now: [makeRow(true), makeRow(true)], next: [makeRow(false), makeRow(false)], trained: makeRow(false) };
  rows.trained.tr.className = "trained";
  const allRows = [...rows.now, ...rows.next, rows.trained];
  $("qtable-body").addEventListener("pointerover", (e) => {
    const td = e.target.closest("td");
    highlight(td ? td.cellIndex - 1 : null);
  });
  $("qtable-body").addEventListener("pointerleave", () => highlight(null));

  function highlight(cell) {
    cells.forEach((c, i) => c.classList.toggle("hl", i === cell));
    heads.forEach((th, i) => th.classList.toggle("hl", i === cell));
    allRows.forEach((row) => row.items.forEach(({ td }, i) => td.classList.toggle("hl", i === cell)));
  }

  function fillRow(row, { label, values, tag, tagClass = "", learner = false }) {
    row.th.textContent = label;
    row.th.classList.toggle("learner", learner);
    row.items.forEach(({ td, el }, i) => {
      const v = values?.[i];
      td.classList.toggle("taken", START[i] !== "0");
      el.classList.remove("up", "down", "used", "picks");
      el.classList.toggle("none", v == null);
      if (tag?.cell === i && tagClass) el.classList.add(tagClass);
      el.style.setProperty("--heat", v == null ? "" : heat(v));
      const badge = tag?.cell === i ? `<small class="tag">${tag.text}</small>` : "";
      el.innerHTML = v == null ? "—" : `${short(v)}${badge}`;
      if (el.tagName === "BUTTON") el.setAttribute("aria-label", `${CELLS[i]}: ${v == null ? "taken" : `score ${fmt(v)}`}`);
    });
  }

  function stopTraining() {
    clearInterval(state.timer);
    state.timer = null;
  }

  function choose(cell) {
    if (!(cell in VALUES) || state.step > 1 || state.timer) return;
    state.action = cell;
    state.step = 1;
    render();
  }

  function newLesson(action) {
    return example(action, state.algorithm, {
      tables: state.tables,
      learner: Math.random() < 0.5 ? 0 : 1,
      nextMove: pickMove(NEXT, EXPLORE),
    });
  }

  function apply(lesson, action) {
    if (lesson.applied) return;
    lesson.applied = true;
    state.tables[lesson.table][action] = lesson.updated;
    state.games++;
    state.last = { table: lesson.table, action, delta: lesson.updated - lesson.old };
  }

  function flash() {
    const el = rows.now[state.last.table].items[state.last.action].el;
    el.classList.remove("flash");
    void el.offsetWidth; // restart the animation
    el.classList.add("flash");
  }

  function goTo(next) {
    stopTraining();
    if (next >= 2 && state.action === null) return;
    if (next <= 1) state.lesson = null;
    else state.lesson ??= newLesson(state.action);
    const learning = next === 3 && !state.lesson.applied;
    if (next === 3) apply(state.lesson, state.action);
    state.step = next;
    render();
    if (learning) {
      flash();
      slide();
    }
  }

  // the "new" marker starts on the old score and slides toward the target
  function slide() {
    const { old, updated } = state.lesson;
    const point = $("nudge-new");
    const moved = $("nudge-moved");
    point.style.transition = moved.style.transition = "none";
    point.style.left = moved.style.left = pct(old);
    moved.style.width = "0%";
    void point.offsetWidth;
    point.style.transition = moved.style.transition = "";
    requestAnimationFrame(() => {
      point.style.left = pct(updated);
      moved.style.left = pct(Math.min(old, updated));
      moved.style.width = `${(Math.abs(updated - old) / 2) * 100}%`;
    });
  }

  function fastForward() {
    if (state.timer) {
      stopTraining();
      render();
      return;
    }
    let left = FAST_FORWARD;
    state.step = 1;
    state.lesson = null;
    state.timer = setInterval(() => {
      const action = pickMove(scores(), EXPLORE);
      apply(newLesson(action), action);
      state.action = action;
      if (--left === 0) stopTraining();
      render();
      flash();
    }, 70);
    render();
  }

  // the story for the Play step when the game goes on: what each algorithm counts for the next turn
  function nextTurnText(lesson) {
    const where = (cell) => `${CELLS[cell]} (${fmt(NEXT[cell])})`;
    const lead = `O replied ${CELLS[lesson.reply]} and it’s X’s turn again. No reward yet, so the agent scores that next turn.`;
    if (state.algorithm === "QLearning") return `${lead} Q-Learning takes its best move there: ${where(lesson.pick)}.`;
    if (state.algorithm === "SARSA") {
      return lesson.pick === best(NEXT)
        ? `${lead} SARSA uses the move it really plays next. This time that’s its best, ${where(lesson.pick)}, same as Q-Learning.`
        : `${lead} SARSA uses the move it really plays next. This time it explored: ${where(lesson.pick)}. Q-Learning would use the best, ${fmt(NEXT[best(NEXT)])}.`;
    }
    const [me, other] = [SUB[lesson.table], SUB[1 - lesson.table]];
    return `${lead} Coin flip: Q${me} learns this time. Q${me} picks its best next move, ${CELLS[lesson.pick]}, and Q${other} scores it: ${fmt(lesson.future)}.`;
  }

  function render() {
    const { step, action, lesson, tables } = state;
    const acting = scores();
    const top = best(acting);
    const twin = double();
    const busy = Boolean(state.timer);
    const goesOn = step >= 2 && !lesson.outcome;

    $("learn-algo-note").innerHTML = NOTES[state.algorithm];
    $("learn-algo-note").style.setProperty("--algo", `var(--c-${state.algorithm})`);

    steps.forEach((button, i) => {
      button.disabled = i >= 2 && action === null;
      button.parentElement.classList.toggle("done", i < step);
      if (i === step) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });

    // board: the agent's scores before the move, its next-turn scores after O's reply
    const board = step < 2 ? START : lesson.afterReply;
    const line = step >= 2 ? winningLine(board) ?? [] : [];
    const nextValues = goesOn && step === 2 ? (twin ? [NEXT, NEXT_2][lesson.table] : NEXT) : null;
    cells.forEach((button, i) => {
      const mark = board[i] === "1" ? "X" : board[i] === "2" ? "O" : "";
      const q = mark ? null : step < 2 ? acting[i] : nextValues?.[i] ?? null;
      const html = mark ? `<span class="mark">${mark}</span>` : q !== null ? `<span class="q${step === 2 ? " later" : ""}">${fmt(q)}</span>` : "";
      if (button.dataset.html !== html) {
        button.innerHTML = html;
        button.dataset.html = html;
      }
      button.className = [
        mark === "X" ? "x" : mark === "O" ? "o" : "",
        i === action && step >= 1 ? "selected" : "",
        step >= 2 && i === action ? "played" : "",
        step >= 2 && i === lesson.reply ? "reply" : "",
        nextValues && i === lesson.pick ? "pick" : "",
        line.includes(i) ? "win" : "",
      ].join(" ");
      button.style.setProperty("--heat", q !== null ? heat(q) : "");
      button.disabled = step > 1 || Boolean(mark) || busy;
      button.setAttribute("aria-label", `${CELLS[i]}: ${mark || (q !== null ? `empty, score ${fmt(q)}` : "empty")}${i === action && step < 2 ? ", chosen" : ""}`);
    });
    $("learn-board-note").innerHTML = step < 2 && twin
      ? "Scores: average of Q₁ and Q₂"
      : nextValues ? `X’s next turn · ${twin ? `Q${SUB[lesson.table]}’s` : "its"} scores` : `<b class="x">X</b> agent &nbsp;·&nbsp; <b class="o">O</b> opponent`;

    // story
    let title;
    let text;
    if (step === 0) {
      title = "This board is the state";
      text = "X is the agent, and it’s X’s move. It looks this board up in its Q-table: one score per empty square, from −1 (“this loses”) to +1 (“this wins”).";
      if (twin) text += " Double Q has two tables and plays on their average.";
    } else if (step === 1) {
      title = busy ? `Fast-forwarding… game ${state.games}` : "Pick a move for X";
      text = busy
        ? `It plays ${Math.round(EXPLORE * 100)}% of moves at random, so every square gets tried. Watch the Q-table below.`
        : action === null
          ? "Click a square. Usually the agent takes the highest score (exploit). Sometimes it picks one at random (explore), to check whether its guess is wrong."
          : `${capital(CELLS[action])} square, score ${fmt(acting[action])}. ${action === top ? "That’s the top score: exploiting what it knows." : "Not the top score: that’s exploring."}`;
    } else if (step === 2) {
      title = lesson.outcome === "1" ? "X wins! Reward +1" : lesson.outcome === "2" ? "O wins. Reward −1" : "The game goes on. Reward 0";
      const over = " Game over, so there’s no next turn and all three algorithms learn the same way. Try the right square to see them differ.";
      text = lesson.outcome === "1"
        ? `Three in a row across the top.${over}`
        : lesson.outcome === "2"
          ? `X didn’t block, so O finished the middle row.${over}`
          : nextTurnText(lesson);
    } else {
      title = "Nudge the score toward the target";
      text = lesson.outcome
        ? `The game is over, so the target is just the reward, ${fmt(lesson.target)}. The score moves 10% of the way there.`
        : `Target = reward 0 + 0.8 × next turn ${fmt(lesson.future)} = ${fmt(lesson.target)}. The score moves 10% of the way there.`;
    }
    $("learn-title").textContent = title;
    $("learn-description").textContent = text;

    $("learn-choices").hidden = step !== 1 || busy;
    $("learn-best").textContent = `Best score (${fmt(acting[top])})`;

    $("learn-chips").hidden = step !== 2;
    if (step === 2) {
      const r = lesson.reward;
      $("learn-chips").innerHTML = `<span class="chip ${r > 0 ? "win" : r < 0 ? "loss" : "none"}"><b>${r > 0 ? "+1" : r < 0 ? "−1" : "0"}</b>reward</span>`
        + (lesson.outcome ? "" : `<span class="chip next"><b>${fmt(lesson.future)}</b>next turn${twin ? `, scored by Q${SUB[1 - lesson.table]}` : ""}</span>`);
    }

    $("learn-update").hidden = step !== 3;
    if (step === 3) {
      const t = twin ? SUB[lesson.table] : "";
      const o = twin ? SUB[1 - lesson.table] : "";
      $("nudge-old").textContent = fmt(lesson.old);
      $("nudge-target-value").textContent = fmt(lesson.target);
      $("nudge-new-value").textContent = fmt(lesson.updated);
      $("nudge-old-point").style.left = pct(lesson.old);
      $("nudge-target").style.left = pct(lesson.target);
      $("nudge-gap").style.left = pct(Math.min(lesson.updated, lesson.target));
      $("nudge-gap").style.width = `${(Math.abs(lesson.target - lesson.updated) / 2) * 100}%`;
      const next = { QLearning: "max<sub>a′</sub> Q(s′,a′)", SARSA: "Q(s′,a′)", DoubleQLearning: `Q${o}(s′, argmax<sub>a′</sub> Q${t}(s′,a′))` }[state.algorithm];
      $("learn-formula").innerHTML = `Q${t}(s,a) ← Q${t}(s,a) + α [ r + γ <mark>${next}</mark> − Q${t}(s,a) ]`;
      $("learn-calculation").textContent = `${lesson.old.toFixed(2)} + 0.1 × (${lesson.reward} + 0.8 × ${lesson.future.toFixed(2)} − ${lesson.old.toFixed(2)}) = ${lesson.updated.toFixed(3)}`;
      $("learn-tip").textContent = (lesson.reward > 0
        ? "Play it again and the score keeps climbing toward +1."
        : lesson.reward < 0
          ? "Play it again and the score keeps sinking toward −1."
          : "Repeat it and the score settles near the target.")
        + (twin ? ` Only Q${t} changed; Q${o} stays as it was.` : "");
      $("learn-other").textContent = lesson.outcome === "1"
        ? "O learns too. In self-play it’s a separate agent with its own table, and its last move just got −1."
        : lesson.outcome === "2"
          ? "O learns too. In self-play it’s a separate agent with its own table, and its winning move just got +1."
          : "O learns too. In self-play it’s a separate agent with its own table, and it scores its move on its next turn.";
    }

    $("learn-back").disabled = step === 0 || busy;
    $("learn-next").disabled = (step === 1 && action === null) || busy;
    $("learn-next").textContent = ["Choose a move", "Play it", "Update the score", "Play again"][step];

    // Q-table
    const real = trainedRow(state.algorithm);
    const last = state.last;
    $("qtable-title").textContent = twin ? "Q-tables" : "Q-table";
    $("qtable-games").textContent = state.games ? `${state.games} game${state.games === 1 ? "" : "s"} played` : "no games yet";
    $("learn-train").textContent = busy ? "Stop" : `Fast-forward ${FAST_FORWARD} games`;
    rows.now.forEach((row, t) => {
      row.tr.hidden = t === 1 && !twin;
      row.tr.classList.toggle("focus", step === 0);
      fillRow(row, {
        label: twin ? `Now · Q${SUB[t]}` : "Now",
        values: tables[t],
        learner: twin && step >= 2 && lesson.table === t,
        tag: last?.table === t && Math.abs(last.delta) >= 0.005 ? { cell: last.action, text: fmt(last.delta) } : null,
        tagClass: last?.delta > 0 ? "up" : "down",
      });
      row.items.forEach(({ el }, i) => {
        el.classList.toggle("chosen", i === action && step >= 1 && (!twin || step < 2 || lesson.table === t));
        el.disabled = !(i in VALUES) || step > 1 || busy;
      });
    });
    rows.next.forEach((row, t) => {
      row.tr.hidden = !goesOn || (t === 1 && !twin);
      if (row.tr.hidden) return;
      const picker = !twin || lesson.table === t;
      fillRow(row, {
        label: twin ? `Next · Q${SUB[t]}` : "Next turn",
        values: [NEXT, NEXT_2][t],
        tag: { cell: lesson.pick, text: !twin ? (state.algorithm === "SARSA" ? "played" : "best") : picker ? "picks" : "scores" },
        tagClass: twin && picker ? "picks" : "used",
      });
    });
    rows.trained.tr.hidden = !real;
    if (real) fillRow(rows.trained, { label: "After 1M games", values: real.values });
    $("qtable-boards").textContent = real ? real.boards.toLocaleString() : "thousands of";
  }

  steps.forEach((button, i) => button.addEventListener("click", () => goTo(i)));
  $("learn-back").addEventListener("click", () => goTo(state.step - 1));
  $("learn-next").addEventListener("click", () => goTo(state.step === 3 ? 1 : state.step + 1));
  $("learn-best").addEventListener("click", () => choose(best(scores())));
  $("learn-random").addEventListener("click", () => choose(pickMove(scores(), 1)));
  $("learn-train").addEventListener("click", fastForward);
  $("learn-reset").addEventListener("click", () => {
    stopTraining();
    Object.assign(state, { step: 0, action: null, tables: fresh(), games: 0, lesson: null, last: null });
    render();
  });
  // each algorithm is its own agent with its own table, so switching starts fresh
  document.querySelectorAll("[data-algo]").forEach((button) => button.addEventListener("click", () => {
    stopTraining();
    document.querySelectorAll("[data-algo]").forEach((b) => b.setAttribute("aria-pressed", b === button));
    Object.assign(state, { algorithm: button.dataset.algo, tables: fresh(), games: 0, lesson: null, last: null, step: Math.min(state.step, 1) });
    render();
  }));

  render();
  return { render };
}
