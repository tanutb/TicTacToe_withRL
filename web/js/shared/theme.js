// Light/dark switch shared by the 3×3 page and the 4×4 FIFO page.
// Light by default, dark only if you picked it.
const MOON = '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>';
const SUN = '<circle cx="12" cy="12" r="4.5"/><path d="M12 1.5v3M12 19.5v3M1.5 12h3M19.5 12h3M4.6 4.6l2.1 2.1M17.3 17.3l2.1 2.1M4.6 19.4l2.1-2.1M17.3 6.7l2.1-2.1" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>';
const KEY = "ttt-theme";

export function initTheme(button = document.getElementById("theme")) {
  let theme = "light";
  try {
    if (JSON.parse(localStorage.getItem(KEY)) === "dark") theme = "dark";
  } catch {}

  function set(value) {
    theme = value;
    document.documentElement.dataset.theme = value;
    button.setAttribute("aria-label", value === "dark" ? "Switch to light mode" : "Switch to dark mode");
    button.querySelector("svg").innerHTML = value === "dark" ? SUN : MOON;
  }

  set(theme);
  button.addEventListener("click", () => {
    set(theme === "dark" ? "light" : "dark");
    try {
      localStorage.setItem(KEY, JSON.stringify(theme));
    } catch {}
  });
}
