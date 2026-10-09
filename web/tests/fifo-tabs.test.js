import test from "node:test";
import assert from "node:assert/strict";
import { LESSONS, QUIZ, quizState, searchSize, startState } from "../js/fifo/learn.js";
import { render } from "../js/fifo/results.js";
import { legalMoves, playMove } from "../js/fifo/game.js";

test("every walkthrough step is a legal position that ends the way its text says", () => {
  const outcomes = LESSONS.map((lesson) => {
    let state = startState(lesson);
    assert.equal(state.player, lesson.reply === undefined ? 0 : 1, `${lesson.label}: who moves`);
    if (lesson.reply !== undefined) {
      const blocker = state.queues[1][0];
      state = playMove(state, lesson.reply);
      assert.equal(state.removed, blocker, "O's move removes its oldest mark, the block");
    }
    assert.ok(legalMoves(state).includes(lesson.target), `${lesson.label}: target is playable`);
    return playMove(state, lesson.target);
  });
  const [marks, oldest, row, trap] = outcomes;
  assert.deepEqual(marks.queues[0], [0, 6, 9, 15]);
  assert.equal(oldest.removed, 0);
  assert.equal(row.winner, 0);
  assert.equal(trap.winner, 0);
  // the model steps' position: every square except the block lets O win next turn
  const quiz = quizState();
  for (const cell of legalMoves(quiz)) {
    const after = playMove(quiz, cell);
    const oWins = legalMoves(after).some((reply) => playMove(after, reply).winner === 1);
    assert.equal(oWins, cell !== QUIZ.target, `X at ${cell}`);
  }
});

test("search size counts every line of play to the given depth", () => {
  const quiz = quizState();
  const moves = legalMoves(quiz).length;
  assert.equal(searchSize(quiz, 1), moves);
  assert.ok(searchSize(quiz, 3) > searchSize(quiz, 2));
});

test("the results tab shows each opponent, the training curve and how the agent plays", () => {
  const row = { wins: 8, draws: 1, losses: 1, moves: 40.2, threats: 2.31, traps: 0.19, trapWins: 0.5 };
  const html = render({
    episodes: 1000, seconds: 300, gamesPerSeat: 5, maxMoves: 100,
    curve: [{ episodes: 0, wins: 1, draws: 8, losses: 1 }, { episodes: 1000, wins: 9, draws: 1, losses: 0 }],
    opponents: { random: { wins: 10, draws: 0, losses: 0 }, tactical: { wins: 9, draws: 1, losses: 0 } },
    style: { sloppy: row, self: { ...row, wins: 3, draws: 4, losses: 3 } },
  });
  assert.match(html, /1,000 games in 5 minutes/);
  assert.match(html, /vs random/);
  assert.match(html, /vs tactical/);
  assert.match(html, /vs human-like/);
  assert.match(html, /<path class="series"/);
  assert.match(html, /Threats per 10 moves<\/dt><dd>2.3/);
  assert.match(html, /60.0% of games end in a win/);
  assert.match(html, /9 wins · 1 draw · 0 losses/);
  const flawless = render({ episodes: 10, gamesPerSeat: 1, maxMoves: 100, depth: 3, curve: [], opponents: {},
    style: { sloppy: row, self: { ...row, wins: 0, draws: 10, losses: 0 } } });
  assert.match(flawless, /looks 3 moves ahead/);
  assert.match(flawless, /every game is a draw/);
});

test("pieces show turns left and fade on their last turn", async () => {
  const { pieceHtml, pieceLabel, turnsLeft } = await import("../js/fifo/piece.js");
  const state = { queues: [[0, 5, 10, 14], [1, 2]], player: 0, moves: 6, winner: null };
  assert.equal(turnsLeft(state, 0, 0), 1);
  assert.equal(turnsLeft(state, 0, 14), 4);
  assert.equal(turnsLeft(state, 1, 1), 3, "O has 2 marks: its oldest stays 3 more turns");
  assert.match(pieceHtml(state, 0, 0), /fifo-piece leaving/);
  assert.equal((pieceHtml(state, 0, 0).match(/class="on"/g) ?? []).length, 1);
  assert.equal((pieceHtml(state, 1, 2).match(/class="on"/g) ?? []).length, 4);
  assert.doesNotMatch(pieceHtml(state, 1, 2), /leaving/);
  assert.equal(pieceLabel(state, 0, 0), "X, vanishes on X’s next move");
  assert.equal(pieceLabel(state, 1, 1), "O, 3 turns left");
});
