import { moveValues } from "./dqn.js";
let network;
self.onmessage = ({ data }) => {
  if (data.network) { network = data.network; return; }
  try {
    self.postMessage({ id: data.id, values: moveValues(network, data.state, data.depth) });
  } catch (error) {
    self.postMessage({ id: data.id, error: error.message });
  }
};
