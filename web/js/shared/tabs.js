// Main tab bar shared by the 3×3 page and the FIFO page: #tab-<name> buttons show #panel-<name>.
// The URL hash remembers the tab; left/right arrows move between tabs.
export function initTabs(names, onChange = () => {}) {
  const $ = (id) => document.getElementById(id);
  let active = names[0];

  function show(name) {
    if (!names.includes(name)) name = names[0];
    active = name;
    for (const tab of names) {
      const selected = tab === name;
      $(`tab-${tab}`).setAttribute("aria-selected", selected);
      $(`tab-${tab}`).tabIndex = selected ? 0 : -1;
      $(`panel-${tab}`).hidden = !selected;
    }
    onChange(name);
  }

  names.forEach((tab, i) => {
    const button = $(`tab-${tab}`);
    button.addEventListener("click", () => {
      history.replaceState(null, "", `#${tab}`);
      show(tab);
    });
    button.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const next = names[(i + (e.key === "ArrowRight" ? 1 : names.length - 1)) % names.length];
      $(`tab-${next}`).focus();
      $(`tab-${next}`).click();
    });
  });
  show(location.hash.slice(1));
  return { get active() { return active; } };
}
