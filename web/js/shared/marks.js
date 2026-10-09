// Hand-drawn X and O shared by the 3×3 and 4×4 boards (viewBox 0 0 100 100).
const X_PATHS = ["M24 23 C40 40, 58 60, 77 78", "M76 22 C60 41, 43 58, 23 79"];
const O_PATH = "M50 18 C71 17, 83 33, 82 51 C81 70, 66 83, 48 82 C29 81, 18 66, 19 48 C21 30, 34 19, 55 21";

export function markSvg(isX, extraClass = "") {
  const paths = isX
    ? X_PATHS.map((d, i) => `<path pathLength="1" d="${d}" style="--delay:${i * 0.1}s"/>`).join("")
    : `<path pathLength="1" d="${O_PATH}"/>`;
  return `<svg viewBox="0 0 100 100" class="${extraClass}" aria-hidden="true">${paths}</svg>`;
}
