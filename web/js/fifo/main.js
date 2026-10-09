import { initTabs } from "../shared/tabs.js";
import { initTheme } from "../shared/theme.js";
import { fetchNetwork } from "./dqn.js";
import { initFIFO } from "./play.js";
import { initFIFOResults } from "./results.js";
import { initFIFOLearn } from "./learn.js";

initTheme();

// every tab uses the same network: fetch it once, and allow a retry after a failure
let network = null;
const loader = () => (network ??= fetchNetwork().catch((error) => {
  network = null;
  throw error;
}));

const play = initFIFO(document.getElementById("panel-play"), loader);
const results = initFIFOResults(document.getElementById("panel-results"));
let learn = null;
initTabs(["play", "results", "learn"], (tab) => {
  play.setActive(tab === "play");
  if (tab === "results") results.load();
  if (tab === "learn" && !learn) learn = initFIFOLearn(document.getElementById("panel-learn"), loader, () => document.getElementById("tab-play").click());
});
