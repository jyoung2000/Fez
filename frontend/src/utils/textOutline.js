/**
 * Generate a multi-directional text-shadow CSS value that simulates a text
 * outline.  Works in ALL browsers (unlike -webkit-text-stroke which is
 * unsupported in Firefox).
 *
 * Uses 8 shadow directions (N, NE, E, SE, S, SW, W, NW) for a uniform
 * outline effect.  An optional drop shadow is appended at the end.
 *
 * @param {number} width   - Outline width in CSS pixels
 * @param {string} color   - Outline color (any CSS color string)
 * @param {string} [dropShadow] - Optional extra drop shadow to append
 * @returns {string} CSS text-shadow value
 */
export function outlineTextShadow(width, color, dropShadow) {
  if (width <= 0) return dropShadow || 'none';

  const offsets = [
    [-1, -1], [0, -1], [1, -1],
    [1,  0],
    [1,  1],  [0,  1], [-1,  1],
    [-1,  0],
  ];

  const shadows = offsets.map(
    ([x, y]) => `${x * width}px ${y * width}px 0 ${color}`,
  );

  if (dropShadow) shadows.push(dropShadow);
  return shadows.join(', ');
}
