import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import {
  EMPTY,
  LINES,
  chooseAction,
  legalMoves,
  move,
  result,
} from "../js/classic/game.js";

test("reject invalid, occupied, and post-terminal moves", () => {
  for (const i of [-1, 9, 1.5, null]) assert.throws(() => move(EMPTY, i));
  assert.equal(move(EMPTY, 0), "100000000");
  assert.throws(() => move("100000000", 0));
  assert.throws(() => move("111220000", 8));
  assert.equal(result("121122211"), "draw");
  for (const line of LINES) {
    const b = Array(9).fill("0");
    line.forEach((i) => (b[i] = "2"));
    assert.equal(result(b.join("")), "2");
  }
});
test("all exported Python actions agree with JavaScript greedy inference", () => {
  for (const name of ["QLearning", "SARSA", "DoubleQLearning"]) {
    const policy = JSON.parse(
      fs.readFileSync(new URL("../data/" + name + ".json", import.meta.url)),
    );
    assert.equal(Object.keys(policy.states).length, 4520);
    for (const [s, entry] of Object.entries(policy.states)) {
      const actions = legalMoves(s),
        best = Math.max(...actions.map((a) => entry.values[a]));
      assert.equal(
        chooseAction(s, policy),
        actions.find((a) => entry.values[a] === best),
        name + " " + s,
      );
    }
  }
});
test("JavaScript and Python enumerate the same reachable boards", () => {
  const seen = new Set(),
    active = new Set();
  function walk(s) {
    if (seen.has(s)) return;
    seen.add(s);
    if (result(s)) return;
    active.add(s);
    for (const a of legalMoves(s)) walk(move(s, a));
  }
  walk(EMPTY);
  assert.equal(seen.size, 5478);
  assert.equal(active.size, 4520);
  const policy = JSON.parse(
    fs.readFileSync(new URL("../data/QLearning.json", import.meta.url)),
  );
  assert.deepEqual([...active].sort(), Object.keys(policy.states).sort());
});
test("every published policy is non-losing against every legal opponent continuation", () => {
  for (const name of ["QLearning", "SARSA", "DoubleQLearning"]) {
    const policy = JSON.parse(
      fs.readFileSync(new URL("../data/" + name + ".json", import.meta.url)),
    );
    for (const agent of ["1", "2"]) {
      const seen = new Set();
      function walk(s, player) {
        if (seen.has(s)) return;
        seen.add(s);
        const end = result(s);
        if (end) {
          assert.notEqual(end, agent === "1" ? "2" : "1", name + " " + s);
          return;
        }
        const actions =
          player === agent ? [chooseAction(s, policy)] : legalMoves(s);
        for (const a of actions) walk(move(s, a), player === "1" ? "2" : "1");
      }
      walk(EMPTY, "1");
    }
  }
});
