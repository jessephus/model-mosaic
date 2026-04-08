/**
 * Colormap implementations for Model Mosaic.
 *
 * Each colormap is a function: (t: number) => [r, g, b]
 * where t is in [0, 1] and r, g, b are in [0, 255].
 */

// --- Continuous colormaps ---

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpColor(c1, c2, t) {
  return [
    Math.round(lerp(c1[0], c2[0], t)),
    Math.round(lerp(c1[1], c2[1], t)),
    Math.round(lerp(c1[2], c2[2], t)),
  ];
}

function multiStopColormap(stops, t) {
  t = Math.max(0, Math.min(1, t));
  if (t <= 0) return stops[0];
  if (t >= 1) return stops[stops.length - 1];
  const segment = t * (stops.length - 1);
  const i = Math.floor(segment);
  const frac = segment - i;
  return lerpColor(stops[i], stops[Math.min(i + 1, stops.length - 1)], frac);
}

const COLORMAPS = {
  // Inferno: dark purple → orange → bright yellow
  inferno: {
    name: "Inferno",
    type: "continuous",
    fn: (t) =>
      multiStopColormap(
        [
          [0, 0, 4],
          [40, 11, 84],
          [101, 21, 110],
          [159, 42, 99],
          [212, 72, 66],
          [245, 125, 21],
          [250, 193, 39],
          [252, 255, 164],
        ],
        t
      ),
  },

  // Coolwarm: blue → white → red (diverging)
  coolwarm: {
    name: "Coolwarm",
    type: "continuous",
    fn: (t) =>
      multiStopColormap(
        [
          [59, 76, 192],
          [116, 149, 226],
          [180, 212, 244],
          [221, 221, 221],
          [244, 195, 172],
          [220, 130, 102],
          [180, 4, 38],
        ],
        t
      ),
  },

  // Grayscale
  grayscale: {
    name: "Grayscale",
    type: "continuous",
    fn: (t) => {
      const v = Math.round(t * 255);
      return [v, v, v];
    },
  },

  // --- Pixel art palettes ---

  // NES palette (54 colors, subset)
  nes: {
    name: "NES",
    type: "indexed",
    palette: [
      [124, 124, 124], [0, 0, 252], [0, 0, 188], [68, 40, 188],
      [148, 0, 132], [168, 0, 32], [168, 16, 0], [136, 20, 0],
      [80, 48, 0], [0, 120, 0], [0, 104, 0], [0, 88, 0],
      [0, 64, 88], [0, 0, 0], [188, 188, 188], [0, 120, 248],
      [0, 88, 248], [104, 68, 252], [216, 0, 204], [228, 0, 88],
      [248, 56, 0], [228, 92, 16], [172, 124, 0], [0, 184, 0],
      [0, 168, 0], [0, 168, 68], [0, 136, 136], [248, 248, 248],
      [60, 188, 252], [104, 136, 252], [152, 120, 248], [248, 120, 248],
      [248, 88, 152], [248, 120, 88], [252, 160, 68], [248, 184, 0],
      [184, 248, 24], [88, 216, 84], [88, 248, 152], [0, 232, 216],
      [120, 120, 120], [252, 252, 252], [164, 228, 252], [184, 184, 248],
      [216, 184, 248], [248, 184, 248], [248, 164, 192], [240, 208, 176],
      [252, 224, 168], [248, 216, 120], [216, 248, 120], [184, 248, 184],
      [184, 248, 216], [0, 252, 252],
    ],
    fn: null, // set below
  },

  // PICO-8 palette (16 colors)
  pico8: {
    name: "PICO-8",
    type: "indexed",
    palette: [
      [0, 0, 0], [29, 43, 83], [126, 37, 83], [0, 135, 81],
      [171, 82, 54], [95, 87, 79], [194, 195, 199], [255, 241, 232],
      [255, 0, 77], [255, 163, 0], [255, 236, 39], [0, 228, 54],
      [41, 173, 255], [131, 118, 156], [255, 119, 168], [255, 204, 170],
    ],
    fn: null,
  },

  // Game Boy (4 shades of green)
  gameboy: {
    name: "Game Boy",
    type: "indexed",
    palette: [
      [15, 56, 15],
      [48, 98, 48],
      [139, 172, 15],
      [155, 188, 15],
    ],
    fn: null,
  },

  // ai-pixel mode: packs 3 weights per pixel as RGB
  aipixel: {
    name: "ai-pixel",
    type: "special",
    fn: null, // handled specially in renderer
  },
};

// Build index-based lookup functions for indexed palettes
function buildIndexedFn(palette) {
  return (t) => {
    const idx = Math.min(
      Math.floor(t * palette.length),
      palette.length - 1
    );
    return palette[idx];
  };
}

for (const key of Object.keys(COLORMAPS)) {
  const cm = COLORMAPS[key];
  if (cm.type === "indexed" && !cm.fn) {
    cm.fn = buildIndexedFn(cm.palette);
  }
}

/**
 * Map a weight value to a normalized [0, 1] value.
 * Uses the global min/max of all weights being rendered.
 */
function normalizeWeight(value, min, max) {
  if (max === min) return 0.5;
  return Math.max(0, Math.min(1, (value - min) / (max - min)));
}
