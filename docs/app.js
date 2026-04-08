/**
 * Model Mosaic — app.js
 * Canvas renderer, layout engine, and interactivity.
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  model: null,        // current model data (from JSON)
  modelId: "mnist-8", // current model key
  view: "mosaic",     // "mosaic" | "layers"
  palette: "inferno", // colormap key
  zoom: 1,
  panX: 0,
  panY: 0,
  hoveredWeight: null, // { value, layerName, index, x, y }
  isDragging: false,
  dragStartX: 0,
  dragStartY: 0,
  selectedLayer: null, // null = all layers
};

const MODELS = {
  "iris-mlp": { file: "models/iris-mlp.json", label: "Iris MLP — The Micro-Sprite" },
  "mnist-8": { file: "models/mnist-8.json", label: "MNIST-8 — The Portrait" },
  "cifar10-tiny": { file: "models/cifar10-tiny.json", label: "CIFAR-10 Tiny — The Landscape" },
};

// ---------------------------------------------------------------------------
// Gallery placards
// ---------------------------------------------------------------------------

const PLACARDS = {
  "iris-mlp": {
    title: "The Micro-Sprite",
    subtitle: "Iris MLP, 67 weights, 2026",
    text: `At just 67 weights arranged in a 9×9 grid, The Micro-Sprite is a study in minimalism — a haiku written in gradient descent. The composition is intimate: two compact blocks of color separated by thin bias bands, like a diptych in a reliquary. The upper register (the hidden layer) shows eight small columns of four weights each, their values splayed across the colormap in broad, confident strokes. No two columns are alike — each has learned a different "question" to ask of an iris flower's measurements.

The lower register is even sparser: three rows of eight, the output layer's final vote on species identity. Here the colors are more polarized — the model has made up its mind. Warm tones pull toward one class, cool tones push away. The bias neurons appear as single bright pixels punctuating each block, tiny but decisive, like the period at the end of a sentence.

What is remarkable about The Micro-Sprite is not its beauty but its sufficiency. These 67 numbers, each discovered through hundreds of passes over 150 flower measurements, are enough to classify iris species with near-perfect accuracy. The image is a reminder that intelligence — at least the narrow, botanical kind — can be very small indeed.`,
    layerAnnotations: {
      "fc1.weight": "Eight learned 'questions' about petal and sepal geometry — each column a different way of seeing a flower.",
      "fc1.bias": "Threshold offsets — how much evidence each hidden neuron needs before it activates.",
      "fc2.weight": "The final vote: three rows mapping eight features to three species. Warm = 'yes,' cool = 'no.'",
      "fc2.bias": "Decision thresholds for each species prediction.",
    },
  },
  "mnist-8": {
    title: "The Portrait",
    subtitle: "MNIST-8 Convolutional Neural Network, 5,998 weights, 2019",
    text: `At first glance, a field of muted earth tones — ochre, umber, and charcoal — disrupted by scattered flares of vermillion and cobalt. The composition reads as abstract expressionism at the macro level, but zoom in and structure emerges: repeating 5×5 motifs in the upper registers where the convolutional filters live, giving way to the dense, grainy texture of the fully-connected layer below. The effect is not unlike looking at a woven textile from across a room, then pressing your face to the fabric to discover its thread structure.

These patterns are not decorative. The bright diagonal striations in the first convolutional block are edge detectors — the model has independently discovered that handwriting is built from oriented strokes, the same insight that drove David Marr's computational theory of vision in the 1980s. The second convolutional block, more complex and less legible, combines those edges into curves and junctions. By the time we reach the dense layer at the bottom — a wide, noisy band of near-zero values punctuated by sharp peaks — the model has abstracted away all spatial structure and is thinking purely in terms of "digit-ness."

What is remarkable is that no one designed these patterns. They are the residue of gradient descent: hundreds of epochs of a loss function pulling 5,998 numbers toward an arrangement that correctly classifies 98.9% of handwritten digits. The image you see is an optimization landscape frozen at its minimum — a mathematical fossil, beautiful in the way that a crystal is beautiful, not by intention but by the relentless pressure of structure seeking its lowest energy state.`,
    layerAnnotations: {
      "Parameter5": "Eight 5×5 convolutional kernels — the model's first attempt to see. Each filter has learned to detect a different oriented edge or gradient, recapitulating decades of computer vision research in a few hundred training steps.",
      "Parameter6": "Bias values for the first conv layer — subtle threshold adjustments.",
      "Parameter87": "Sixteen 5×5 filters operating on eight input channels — 3,200 weights that combine simple edges into curves, corners, and junctions. The visual complexity increases dramatically.",
      "Parameter88": "Bias values for the second conv layer.",
      "Pooling160_Output_0_reshape0_shape": "Reshape parameters — structural metadata, not learned features.",
      "Parameter193": "The dense classification layer — 2,560 weights that map spatial features into digit identity. The texture here is noise-like because spatial structure has been abstracted away; only statistical signatures of each digit class remain.",
      "Parameter193_reshape1_shape": "Reshape parameters.",
      "Parameter194": "Output bias — ten values, one per digit, encoding the model's prior expectation for each class.",
    },
  },
  "cifar10-tiny": {
    title: "The Landscape",
    subtitle: "CIFAR-10 Tiny CNN, 14,410 weights, 2026",
    text: `The Landscape is the largest piece in the gallery, and the most visually complex. At 121×121 pixels, it rewards patience — what initially appears as television static resolves, on closer inspection, into layered geological strata. The three convolutional blocks form distinct horizontal bands of increasing density, like sedimentary rock viewed in cross-section, each layer deposited by a different epoch of learning.

The first stratum is the thinnest and most colorful: eight tiny 3×3 filters operating on three color channels. These are the model's retinas — simple color-edge detectors that respond to red-green boundaries, brightness gradients, and blue-sky-against-brown-ground transitions. They are the only layer that "sees" color directly, and their vivid hues reflect this intimacy with the raw pixel data of photographs.

Below lies the second stratum: sixteen filters reading from eight channels, 1,152 weights of increasing subtlety. And below that, the deepest convolutional layer — 4,608 weights in thirty-two filters, their patterns now too abstract for the human eye to decode. These neurons are thinking about wings, wheels, and whiskers — the parts of objects that distinguish an airplane from an automobile.

The dense layer at the base — 8,192 weights — is a vast, stippled plain. This is where spatial reasoning gives way to categorical judgment. The model has compressed a 32×32 color photograph into sixteen abstract features, and from those sixteen numbers, it will guess whether it's looking at a ship or a horse. The weight painting of this layer has the quality of a Seurat pointillist canvas: meaningless up close, meaningful at a distance, each dot placed not by artistic intention but by the mathematics of backpropagation.`,
    layerAnnotations: {
      "conv1.weight": "Eight 3×3 color filters — the model's retinas. Each kernel learns to detect a different color-edge combination in the raw photograph.",
      "conv1.bias": "Bias for the first conv layer.",
      "conv2.weight": "Sixteen filters combining first-layer edges into textures and corners — 1,152 weights of increasing visual abstraction.",
      "conv2.bias": "Bias for the second conv layer.",
      "conv3.weight": "Thirty-two deep filters — 4,608 weights encoding parts of objects: curves of wings, straight edges of buildings, fur textures of animals.",
      "conv3.bias": "Bias for the third conv layer.",
      "fc1.weight": "The dense mapping — 8,192 weights compressing spatial features into 16 abstract category signals. A pointillist canvas of learned associations.",
      "fc1.bias": "Bias for the dense layer.",
      "fc2.weight": "The final classifier — 160 weights casting ten votes, one per object category.",
      "fc2.bias": "Output bias — prior expectations for each of the ten CIFAR-10 categories.",
    },
  },
};

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

function computeMosaicLayout(model) {
  const allWeights = [];
  const weightMeta = [];

  for (const layer of model.layers) {
    for (let i = 0; i < layer.weights.length; i++) {
      allWeights.push(layer.weights[i]);
      weightMeta.push({ layer: layer.name, index: i, shape: layer.shape });
    }
  }

  const total = allWeights.length;
  const cols = Math.ceil(Math.sqrt(total));
  const rows = Math.ceil(total / cols);

  return { allWeights, weightMeta, cols, rows, total };
}

function computeLayerLayout(model, selectedLayer) {
  const layouts = [];
  const layers = selectedLayer
    ? model.layers.filter((l) => l.name === selectedLayer)
    : model.layers;

  for (const layer of layers) {
    const n = layer.weights.length;
    // Use natural 2D shape if available, else make a square-ish grid
    let cols, rows;
    if (layer.shape.length === 2) {
      rows = layer.shape[0];
      cols = layer.shape[1];
    } else if (layer.shape.length === 4) {
      // Conv: [out, in, h, w] → show as grid of kernels
      const [outC, inC, kH, kW] = layer.shape;
      cols = inC * kW + (inC - 1); // kernels side by side with 1px gap
      rows = outC * kH + (outC - 1);
      // Simplified: just flatten
      cols = Math.ceil(Math.sqrt(n));
      rows = Math.ceil(n / cols);
    } else {
      cols = Math.ceil(Math.sqrt(n));
      rows = Math.ceil(n / cols);
    }
    layouts.push({ layer, cols, rows });
  }
  return layouts;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function getGlobalMinMax(model) {
  let min = Infinity, max = -Infinity;
  for (const layer of model.layers) {
    if (layer.min < min) min = layer.min;
    if (layer.max > max) max = layer.max;
  }
  return { min, max };
}

function renderMosaic(ctx, model, colormap, width, height) {
  const layout = computeMosaicLayout(model);
  const { min, max } = getGlobalMinMax(model);
  const cmFn = COLORMAPS[colormap];

  if (!cmFn) return layout;

  const pixelW = width / layout.cols;
  const pixelH = height / layout.rows;

  // Handle ai-pixel special mode: pack 3 weights per pixel as RGB
  if (colormap === "aipixel") {
    for (let i = 0; i < layout.total; i += 3) {
      const x = (Math.floor(i / 3)) % layout.cols;
      const y = Math.floor((Math.floor(i / 3)) / layout.cols);
      const r = i < layout.total ? Math.round(normalizeWeight(layout.allWeights[i], min, max) * 255) : 128;
      const g = i + 1 < layout.total ? Math.round(normalizeWeight(layout.allWeights[i + 1], min, max) * 255) : 128;
      const b = i + 2 < layout.total ? Math.round(normalizeWeight(layout.allWeights[i + 2], min, max) * 255) : 128;
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(x * pixelW, y * pixelH, pixelW + 0.5, pixelH + 0.5);
    }
    return layout;
  }

  for (let i = 0; i < layout.total; i++) {
    const x = i % layout.cols;
    const y = Math.floor(i / layout.cols);
    const t = normalizeWeight(layout.allWeights[i], min, max);
    const [r, g, b] = cmFn.fn(t);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(x * pixelW, y * pixelH, pixelW + 0.5, pixelH + 0.5);
  }

  return layout;
}

function renderLayers(ctx, model, colormap, canvasWidth, selectedLayer) {
  const layouts = computeLayerLayout(model, selectedLayer);
  const { min, max } = getGlobalMinMax(model);
  const cmFn = COLORMAPS[colormap];
  if (!cmFn || colormap === "aipixel") return layouts; // ai-pixel only works in mosaic

  const PADDING = 12;
  const LABEL_HEIGHT = 20;
  let yOffset = 0;

  for (const { layer, cols, rows } of layouts) {
    const pixelSize = Math.max(2, Math.min(12, Math.floor((canvasWidth - PADDING * 2) / cols)));
    const blockW = cols * pixelSize;
    const blockH = rows * pixelSize;
    const xOffset = Math.floor((canvasWidth - blockW) / 2);

    // Label
    ctx.fillStyle = "#888";
    ctx.font = "11px monospace";
    ctx.fillText(`${layer.name} [${layer.shape.join("×")}]`, xOffset, yOffset + 12);
    yOffset += LABEL_HEIGHT;

    // Pixels
    for (let i = 0; i < layer.weights.length; i++) {
      const x = i % cols;
      const y = Math.floor(i / cols);
      const t = normalizeWeight(layer.weights[i], min, max);
      const [r, g, b] = cmFn.fn(t);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(xOffset + x * pixelSize, yOffset + y * pixelSize, pixelSize, pixelSize);
    }

    yOffset += blockH + PADDING;
  }

  return { totalHeight: yOffset, layouts };
}

// ---------------------------------------------------------------------------
// Main render loop
// ---------------------------------------------------------------------------

let currentLayout = null;

function render() {
  const canvas = document.getElementById("mosaic-canvas");
  const ctx = canvas.getContext("2d");
  const model = state.model;
  if (!model) return;

  if (state.view === "mosaic") {
    const layout = computeMosaicLayout(model);
    const PIXEL_SIZE = Math.max(2, Math.min(8, Math.floor(600 / layout.cols)));
    canvas.width = layout.cols * PIXEL_SIZE;
    canvas.height = layout.rows * PIXEL_SIZE;
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    currentLayout = renderMosaic(ctx, model, state.palette, canvas.width, canvas.height);
    currentLayout.pixelW = PIXEL_SIZE;
    currentLayout.pixelH = PIXEL_SIZE;
    currentLayout.mode = "mosaic";
  } else {
    const WIDTH = 600;
    canvas.width = WIDTH;
    // First pass: compute height
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = WIDTH;
    tempCanvas.height = 10000;
    const tempCtx = tempCanvas.getContext("2d");
    const result = renderLayers(tempCtx, model, state.palette, WIDTH, state.selectedLayer);
    canvas.height = Math.max(200, result.totalHeight || 400);
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    currentLayout = renderLayers(ctx, model, state.palette, WIDTH, state.selectedLayer);
    currentLayout.mode = "layers";
  }

  updateInfoBar();
}

// ---------------------------------------------------------------------------
// Hover inspection
// ---------------------------------------------------------------------------

function handleHover(e) {
  if (!state.model || !currentLayout || currentLayout.mode !== "mosaic") {
    updateTooltip(null);
    return;
  }

  const canvas = document.getElementById("mosaic-canvas");
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const cx = (e.clientX - rect.left) * scaleX;
  const cy = (e.clientY - rect.top) * scaleY;

  const col = Math.floor(cx / currentLayout.pixelW);
  const row = Math.floor(cy / currentLayout.pixelH);
  const idx = row * currentLayout.cols + col;

  if (idx >= 0 && idx < currentLayout.total) {
    const meta = currentLayout.weightMeta[idx];
    const value = currentLayout.allWeights[idx];
    updateTooltip({
      value: value.toFixed(6),
      layer: meta.layer,
      index: meta.index,
      position: `[${col}, ${row}]`,
    });
  } else {
    updateTooltip(null);
  }
}

function updateTooltip(info) {
  const el = document.getElementById("tooltip");
  if (!info) {
    el.textContent = "Hover over a pixel to inspect its weight";
    return;
  }
  el.textContent = `weight = ${info.value}  |  ${info.layer}[${info.index}]  |  pixel ${info.position}`;
}

// ---------------------------------------------------------------------------
// UI updates
// ---------------------------------------------------------------------------

function updateInfoBar() {
  const el = document.getElementById("info-bar");
  if (!state.model) return;
  const m = state.model.model;
  const gridSide = Math.ceil(Math.sqrt(m.total_params));
  el.textContent = `${m.name} · ${m.total_params.toLocaleString()} weights · ${gridSide}×${Math.ceil(m.total_params / gridSide)} grid`;
}

function updatePlacard() {
  const placard = PLACARDS[state.modelId];
  const el = document.getElementById("placard");
  if (!placard) {
    el.style.display = "none";
    return;
  }
  el.style.display = "block";
  document.getElementById("placard-title").textContent = placard.title;
  document.getElementById("placard-subtitle").textContent = placard.subtitle;
  document.getElementById("placard-text").textContent = placard.text;
}

function updateLayerList() {
  const el = document.getElementById("layer-list");
  el.innerHTML = "";
  if (!state.model) return;

  const allBtn = document.createElement("button");
  allBtn.textContent = "All layers";
  allBtn.className = state.selectedLayer === null ? "active" : "";
  allBtn.onclick = () => {
    state.selectedLayer = null;
    updateLayerList();
    render();
  };
  el.appendChild(allBtn);

  for (const layer of state.model.layers) {
    const btn = document.createElement("button");
    const count = layer.weights.length;
    btn.textContent = `${layer.name} (${count})`;
    btn.className = state.selectedLayer === layer.name ? "active" : "";
    btn.onclick = () => {
      state.selectedLayer = layer.name;
      state.view = "layers";
      document.getElementById("view-select").value = "layers";
      updateLayerList();
      render();
      // Show layer annotation if available
      showLayerAnnotation(layer.name);
    };
    el.appendChild(btn);
  }
}

function showLayerAnnotation(layerName) {
  const placard = PLACARDS[state.modelId];
  const annotEl = document.getElementById("layer-annotation");
  if (placard && placard.layerAnnotations && placard.layerAnnotations[layerName]) {
    annotEl.textContent = placard.layerAnnotations[layerName];
    annotEl.style.display = "block";
  } else {
    annotEl.style.display = "none";
  }
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

function exportPNG() {
  const canvas = document.getElementById("mosaic-canvas");
  canvas.toBlob((blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `model-mosaic-${state.modelId}-${state.palette}.png`;
    a.click();
    URL.revokeObjectURL(url);
  });
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

async function loadModel(modelId) {
  const info = MODELS[modelId];
  if (!info) return;

  state.modelId = modelId;
  state.selectedLayer = null;

  const resp = await fetch(info.file);
  state.model = await resp.json();

  updateLayerList();
  updatePlacard();
  document.getElementById("layer-annotation").style.display = "none";
  render();
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

function init() {
  // Populate model selector
  const modelSelect = document.getElementById("model-select");
  for (const [id, info] of Object.entries(MODELS)) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = info.label;
    modelSelect.appendChild(opt);
  }
  modelSelect.value = state.modelId;
  modelSelect.addEventListener("change", (e) => loadModel(e.target.value));

  // Palette selector
  const paletteSelect = document.getElementById("palette-select");
  for (const [id, cm] of Object.entries(COLORMAPS)) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = cm.name;
    paletteSelect.appendChild(opt);
  }
  paletteSelect.value = state.palette;
  paletteSelect.addEventListener("change", (e) => {
    state.palette = e.target.value;
    render();
  });

  // View selector
  const viewSelect = document.getElementById("view-select");
  viewSelect.addEventListener("change", (e) => {
    state.view = e.target.value;
    document.getElementById("layer-annotation").style.display = "none";
    render();
  });

  // Export button
  document.getElementById("export-btn").addEventListener("click", exportPNG);

  // Placard toggle
  document.getElementById("placard-toggle").addEventListener("click", () => {
    const body = document.getElementById("placard-body");
    const btn = document.getElementById("placard-toggle");
    if (body.style.display === "none") {
      body.style.display = "block";
      btn.textContent = "▾ Hide commentary";
    } else {
      body.style.display = "none";
      btn.textContent = "▸ Show commentary";
    }
  });

  // Canvas hover
  const canvas = document.getElementById("mosaic-canvas");
  canvas.addEventListener("mousemove", handleHover);
  canvas.addEventListener("mouseleave", () => updateTooltip(null));

  // Load default model
  loadModel(state.modelId);
}

document.addEventListener("DOMContentLoaded", init);
