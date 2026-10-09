import test from "node:test";
import assert from "node:assert/strict";
import { initFIFO } from "../js/fifo/play.js";

// A minimal DOM/event adapter for controller tests; this does not verify visual layout.
// An all-zero network scores every square 0, so ties go to the first legal square (Math.random is 0).
const zeros = (n) => Buffer.from(new Float32Array(n).buffer).toString("base64");
const MODEL = { rules: "4x4-four-marks-four-in-row-100-ply", encoding: "fifo-planes-v1", episodes: 5,
  layers: [{ in: 129, out: 16, weight: zeros(129 * 16), bias: zeros(16) }] };

function mount(t, failFetch = false) {
  const elements = new Map();
  class Element {
    constructor() {
      this.children = []; this.listeners = {}; this.attributes = {};
      this.value = ""; this.hidden = false; this.disabled = false; this.checked = false;
      this.textContent = ""; this.classList = { add() {} };
    }
    set innerHTML(html) {
      this.html = html; this.children = [];
      for (const [, id] of html.matchAll(/id="([^"]+)"/g)) elements.set(id, new Element());
    }
    get innerHTML() { return this.html ?? ""; }
    append(child) {
      if (!this.children.length && child.option) this.value = child.value;
      this.children.push(child);
    }
    replaceChildren(...children) { this.children = children; }
    addEventListener(event, callback) { (this.listeners[event] ??= []).push(callback); }
    setAttribute(key, value) { this.attributes[key] = String(value); }
    querySelector() { return null; }
    fire(event = "click") {
      if (event === "click" && this.disabled) return;
      for (const callback of this.listeners[event] ?? []) callback({ target: this });
    }
  }
  elements.set("panel-play", new Element());
  const previous = Object.getOwnPropertyDescriptors(globalThis);
  globalThis.document = {
    getElementById(id) {
      assert.ok(elements.has(id), `Markup must contain ${id}`);
      return elements.get(id);
    },
    createElement: () => new Element(),
  };
  globalThis.Option = class extends Element {
    constructor(label, value) { super(); this.option = true; this.textContent = label; this.value = value; }
  };
  t.after(() => {
    for (const key of ["document", "Option"]) {
      if (previous[key]) Object.defineProperty(globalThis, key, previous[key]);
      else delete globalThis[key];
    }
  });
  const callbacks = new Map();
  let id = 0;
  t.mock.method(globalThis, "setTimeout", (callback) => { callbacks.set(++id, callback); return id; });
  t.mock.method(globalThis, "clearTimeout", (timer) => callbacks.delete(timer));
  t.mock.method(Math, "random", () => 0);
  t.mock.method(globalThis, "fetch", async (url) => ({
    ok: !failFetch, status: failFetch ? 404 : 200,
    json: async () => MODEL,
  }));
  const controller = initFIFO();
  const get = (id) => elements.get(`fifo-${id}`);
  get("speed").value = "800";
  return {
    controller, get, callbacks,
    cell: (index) => get("board").children[index],
    settle: () => new Promise((resolve) => setImmediate(resolve)),
    tick() {
      const [id, callback] = callbacks.entries().next().value ?? [];
      assert.ok(callback, "an agent move is scheduled"); callbacks.delete(id); callback();
    },
  };
}

test("FIFO human play, reply, hint, undo, and side switching", async (t) => {
  const ui = mount(t);
  assert.equal(ui.get("board").children.length, 16);
  assert.equal(ui.cell(0).disabled, true);
  ui.controller.setActive(true); await ui.settle();
  assert.equal(ui.cell(0).disabled, false);
  ui.cell(5).fire();
  assert.match(ui.cell(5).className, /fifo-x/);
  assert.match(ui.cell(5).innerHTML, /fifo-piece/);
  assert.equal(ui.callbacks.size, 1);
  ui.tick();
  assert.match(ui.cell(0).className, /fifo-o/);
  assert.equal(ui.get("log").children.length, 2);
  ui.get("hint").fire();
  assert.match(ui.get("removal").textContent, /Suggested move/);
  ui.get("undo").fire();
  assert.equal(ui.get("log").children.length, 0);
  assert.equal(ui.cell(5).disabled, false);
  ui.get("side").value = "1"; ui.get("side").fire("change");
  assert.equal(ui.callbacks.size, 1);
  ui.tick();
  assert.match(ui.cell(0).className, /fifo-x/);
  assert.equal(ui.cell(1).disabled, false);
});

test("FIFO autoplay pauses on tab exit and resumes only on request", async (t) => {
  const ui = mount(t);
  ui.controller.setActive(true); await ui.settle();
  ui.get("watch-mode").fire();
  assert.equal(ui.get("step").hidden, false);
  assert.equal(ui.cell(0).disabled, true);
  ui.get("step").fire();
  assert.equal(ui.get("log").children.length, 1);
  assert.equal(ui.callbacks.size, 0);
  ui.get("auto").fire();
  assert.equal(ui.get("auto").textContent, "Pause");
  ui.tick();
  assert.equal(ui.get("log").children.length, 2);
  ui.controller.setActive(false);
  assert.equal(ui.callbacks.size, 0);
  ui.controller.setActive(true);
  assert.equal(ui.get("auto").textContent, "Play");
  assert.equal(ui.callbacks.size, 0);
  ui.get("human-mode").fire();
  assert.equal(ui.get("log").children.length, 0);
});

test("agent vs agent opens with a random move, then the network plays", async (t) => {
  const ui = mount(t);
  ui.controller.setActive(true); await ui.settle();
  ui.get("watch-mode").fire();
  ui.get("step").fire();
  ui.get("step").fire();
  const [first, second] = ui.get("log").children.map((li) => li.textContent);
  assert.match(first, /random opening/);
  assert.match(second, /Deep Q-Learning \(O\): r1 c2; score 0.00/);
});

test("FIFO wins increment once, and reset then undo cannot create negative scores", async (t) => {
  const ui = mount(t);
  ui.controller.setActive(true); await ui.settle();
  ui.get("depth-0").fire(); // network only, so the zero network replies predictably
  for (const cell of [4, 5, 6]) { ui.cell(cell).fire(); ui.tick(); }
  ui.cell(7).fire();
  assert.match(ui.get("status").textContent, /X wins/);
  assert.equal(ui.get("score-x").textContent, 1);
  assert.equal(ui.callbacks.size, 0);
  ui.get("reset-score").fire();
  ui.get("undo").fire();
  assert.equal(ui.get("score-x").textContent, 0);
  assert.equal(ui.get("log").children.length, 6);
  ui.cell(7).fire();
  assert.equal(ui.get("score-x").textContent, 1);
  ui.get("new").fire();
  assert.equal(ui.get("score-x").textContent, 1);
});

test("FIFO removal appears in the move log and leaves the old square empty", async (t) => {
  const ui = mount(t);
  ui.controller.setActive(true); await ui.settle();
  ui.get("depth-0").fire(); // network only, so the zero network replies predictably
  for (const cell of [1, 5, 10, 14]) { ui.cell(cell).fire(); ui.tick(); }
  ui.cell(15).fire();
  assert.match(ui.get("removal").textContent, /oldest mark at r1 c2 disappeared/);
  assert.match(ui.get("log").children.at(-1).textContent, /removed r1 c2/);
  assert.match(ui.cell(1).attributes["aria-label"], /empty/);
  ui.get("undo").fire();
  assert.match(ui.cell(1).attributes["aria-label"], /X, vanishes on X’s next move/);
});

test("failed FIFO policy loads expose a retry and leave the board disabled", async (t) => {
  const ui = mount(t, true);
  ui.controller.setActive(true); await ui.settle();
  assert.equal(ui.get("error").hidden, false);
  assert.equal(ui.get("retry").hidden, false);
  assert.equal(ui.cell(0).disabled, true);
  assert.equal(ui.callbacks.size, 0);
});

test("with look-ahead the agent blocks a row the network alone would miss", async (t) => {
  const ui = mount(t);
  ui.controller.setActive(true); await ui.settle();
  assert.match(ui.get("depth-note").textContent, /never lost/i);
  // X builds r2 c1-c3; the zero network has no idea, but the 3-move search sees the threat
  ui.cell(4).fire(); ui.tick();
  ui.cell(5).fire(); ui.tick();
  ui.cell(6).fire(); ui.tick();
  assert.match(ui.cell(7).className, /fifo-o/, "O blocks the end of the row");
  ui.get("depth-0").fire();
  assert.equal(ui.get("depth-0").attributes["aria-pressed"], "true");
  assert.match(ui.get("depth-note").textContent, /Network only/);
});
