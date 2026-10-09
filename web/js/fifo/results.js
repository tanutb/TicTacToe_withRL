// The FIFO Results tab: data/fifo-evaluation.json, written by `python evaluate.py --game fifo`.
const OPPONENTS = {
  random: "vs random",
  tactical: "vs tactical",
  sloppy: "vs human-like",
};
const pct = (value) => `${(value * 100).toFixed(1)}%`;
const count = (n, word, plural = `${word}s`) => `${n.toLocaleString()} ${n === 1 ? word : plural}`;

function bar(label, row) {
  const total = row.wins + row.draws + row.losses;
  const share = (n) => n / total;
  return `<div class="bar-row"><span class="name">${label}</span><div class="bar" role="img" aria-label="${label}: ${pct(share(row.wins))} wins, ${pct(share(row.draws))} draws, ${pct(share(row.losses))} losses">${
    ["win", "draw", "loss"].map((key) => {
      const v = share(row[key === "loss" ? "losses" : `${key}s`]);
      return `<span class="${key}" style="flex: 0 0 ${v * 100}%">${v >= 0.07 ? pct(v) : ""}</span>`;
    }).join("")}</div><span class="small">${count(row.wins, "win")} · ${count(row.draws, "draw")} · ${count(row.losses, "loss", "losses")}</span></div>`;
}

function curve(points, episodes) {
  const rate = (p) => p.wins / (p.wins + p.draws + p.losses);
  const x = (p) => 44 + (p.episodes / Math.max(1, episodes)) * 466;
  const y = (v) => 14 + (1 - v) * 186;
  const line = points.map((p, i) => `${i ? "L" : "M"}${x(p).toFixed(1)} ${y(rate(p)).toFixed(1)}`).join(" ");
  const grid = [0, 0.25, 0.5, 0.75, 1].map((v) => `<line x1="44" x2="510" y1="${y(v)}" y2="${y(v)}"/><text x="36" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`).join("");
  const ticks = [0, episodes / 2, episodes].map((e) => `<text x="${x({ episodes: e })}" y="226" text-anchor="middle">${e ? `${e / 1000}k` : "0"}</text>`).join("");
  return `<svg class="curve" viewBox="0 0 520 236" role="img" aria-label="Win rate against the tactical player at training checkpoints: ${points.map((p) => `${p.episodes.toLocaleString()} games ${pct(rate(p))}`).join("; ")}"><g class="axis">${grid}${ticks}</g><path class="series" d="${line}" style="stroke: var(--ink)"/>${points.map((p) => `<circle cx="${x(p)}" cy="${y(rate(p))}" r="4" style="fill: var(--ink)"/>`).join("")}</svg>`;
}

export function render(data) {
  const style = data.style ?? {};
  const rows = { ...data.opponents, ...(style.sloppy ? { sloppy: style.sloppy } : {}) };
  const games = data.gamesPerSeat * 2;
  const minutes = data.seconds ? ` in ${Math.max(1, Math.round(data.seconds / 60))} minutes on a GPU` : "";
  const stats = (row) => `<div><dt>Moves per game</dt><dd>${row.moves.toFixed(0)}</dd></div><div><dt>Threats per 10 moves</dt><dd>${row.threats.toFixed(1)}</dd></div><div><dt>Aging traps per 10 moves</dt><dd>${row.traps.toFixed(2)}</dd></div><div><dt>Wins from a trap</dt><dd>${pct(row.trapWins)}</dd></div>`;
  return `
    <p class="lede">Deep Q-Learning trained by self-play for ${data.episodes.toLocaleString()} games${minutes}. ${data.depth ? `It looks ${data.depth} moves ahead, as on the Play tab. ` : ""}Each result is ${games.toLocaleString()} test games, half as X and half as O. A draw means the ${data.maxMoves}-move limit was reached.</p>
    <div class="results-grid">
      <div class="card">
        <h2>Final result</h2>
        <div class="bars">${Object.entries(rows).map(([name, row]) => bar(OPPONENTS[name] ?? name, row)).join("")}</div>
        <p class="legend"><span><i class="win"></i>win</span><span><i class="draw"></i>draw</span><span><i class="loss"></i>loss</span></p>
      </div>
      <div class="card">
        <h2>Win rate while training <span class="small">network alone vs tactical</span></h2>
        ${(data.curve ?? []).length > 1 ? curve(data.curve, data.episodes) : `<p class="small">No training checkpoints were recorded.</p>`}
      </div>
    </div>
    ${style.sloppy ? `<div class="card">
      <h2>How it plays</h2>
      <p class="small">Against the human-like opponent. A threat means it could win on its next move if not blocked. An aging trap is a threat the opponent cannot block, because the blocking mark is their oldest and is about to vanish.</p>
      <dl class="settings fifo-style">${stats(style.sloppy)}</dl>
      ${style.self ? `<p class="small">${style.self.wins + style.self.losses === 0
    ? "Agent vs agent: every game is a draw. Neither side ever slips, like two perfect 3×3 players."
    : `Agent vs agent (random first move): ${pct((style.self.wins + style.self.losses) / (style.self.wins + style.self.draws + style.self.losses))} of games end in a win, after ${style.self.moves.toFixed(0)} moves on average.`}</p>` : ""}
    </div>` : ""}
    <ul class="facts">
      <li><b>Random</b> picks any empty square.</li>
      <li><b>Tactical</b> takes any immediate win, otherwise picks a move that leaves the fewest immediate winning replies. It does not see aging traps coming.</li>
      ${style.sloppy ? "<li><b>Human-like</b> is the agent itself making a random move 10% of the time: strong, but it slips.</li>" : ""}
    </ul>`;
}

export function initFIFOResults(root = document.getElementById("panel-results")) {
  let loaded = false;
  async function load() {
    if (loaded) return;
    loaded = true;
    try {
      const response = await fetch("data/fifo-evaluation.json");
      if (!response.ok) throw new Error(`results returned ${response.status}`);
      root.innerHTML = render(await response.json());
    } catch (error) {
      loaded = false;
      root.innerHTML = `<p class="error">Couldn't load the results (${error.message}). Run "python evaluate.py --game fifo" to create them.</p>`;
    }
  }
  return { load };
}
