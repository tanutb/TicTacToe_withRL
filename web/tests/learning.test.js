import test from "node:test";
import assert from "node:assert/strict";
import { NEXT, NEXT_BOARD, START, VALUES, VALUES_2, average, best, example, pickMove } from "../js/classic/learning.js";
import { legalMoves, result } from "../js/classic/game.js";

const close = (a, b) => Math.abs(a - b) < 1e-10;

test("a winning move ends immediately and learns only from its reward", () => {
  for (const algorithm of ["QLearning", "SARSA", "DoubleQLearning"]) {
    const lesson = example(2, algorithm);
    assert.equal(lesson.reply, undefined);
    assert.equal(lesson.outcome, "1");
    assert.equal(lesson.future, 0);
    assert.equal(lesson.pick, null);
    assert.ok(close(lesson.updated, 0.64));
  }
});

test("missing O's threat lowers the chosen move's score", () => {
  for (const cell of [6, 7, 8]) {
    const lesson = example(cell);
    assert.equal(lesson.reply, 5);
    assert.equal(lesson.outcome, "2");
    assert.equal(lesson.reward, -1);
    assert.equal(lesson.future, 0);
    assert.ok(lesson.updated < lesson.old);
  }
});

test("blocking keeps the game going on the next-turn board", () => {
  const lesson = example(5);
  assert.equal(lesson.outcome, null);
  assert.equal(lesson.reward, 0);
  assert.equal(lesson.afterReply, NEXT_BOARD);
  assert.deepEqual(legalMoves(NEXT_BOARD), Object.keys(NEXT).map(Number));
});

test("Q-Learning uses the best next move, SARSA the one it really plays", () => {
  const q = example(5, "QLearning", { nextMove: 7 });
  assert.equal(q.pick, 2);
  assert.ok(close(q.target, 0.72));
  assert.ok(close(q.updated, 0.252));

  const greedy = example(5, "SARSA", { nextMove: 2 });
  assert.ok(close(greedy.updated, q.updated));

  const explored = example(5, "SARSA", { nextMove: 7 });
  assert.equal(explored.pick, 7);
  assert.equal(explored.target, 0);
  assert.ok(close(explored.updated, 0.18));
});

test("Double Q updates one table, picking with it and scoring with the other", () => {
  const first = example(5, "DoubleQLearning", { learner: 0 });
  assert.equal(first.table, 0);
  assert.equal(first.pick, 2);
  assert.ok(close(first.future, 0.6));
  assert.equal(first.old, VALUES[5]);
  assert.ok(close(first.updated, 0.228));

  const second = example(5, "DoubleQLearning", { learner: 1 });
  assert.equal(second.table, 1);
  assert.ok(close(second.future, 0.9));
  assert.equal(second.old, VALUES_2[5]);
  assert.ok(close(second.updated, 0.342));
});

test("updates start from the live table and move 10% toward the target", () => {
  const lesson = example(2, "QLearning", { tables: [{ ...VALUES, 2: 0.9 }] });
  assert.equal(lesson.old, 0.9);
  assert.equal(lesson.target, 1);
  assert.ok(close(lesson.updated, 0.91));
});

test("repeating a move converges its score toward the target", () => {
  const values = { ...VALUES };
  for (let i = 0; i < 200; i++) for (const cell of [2, 6]) values[cell] = example(cell, "QLearning", { tables: [values] }).updated;
  assert.ok(values[2] > 0.99);
  assert.ok(values[6] < -0.99);
});

test("ε-greedy picks the best score or, when exploring, any legal move", () => {
  assert.equal(best(VALUES), 2);
  assert.equal(pickMove(VALUES, 0), 2);
  assert.equal(pickMove(VALUES, 1, () => 0), 2);
  assert.equal(pickMove(VALUES, 1, (() => { const r = [0, 0.99]; return () => r.shift(); })()), 8);
});

test("Double Q acts on the average of its two tables", () => {
  const mean = average(VALUES, VALUES_2);
  assert.deepEqual(Object.keys(mean), Object.keys(VALUES));
  assert.ok(close(mean[5], 0.25));
});

test("every offered move and opponent reply follows the game rules", () => {
  for (const cell of legalMoves(START)) {
    const lesson = example(cell);
    if (!result(lesson.afterMove)) assert.ok(legalMoves(lesson.afterMove).includes(lesson.reply));
  }
  assert.throws(() => example(0));
});
