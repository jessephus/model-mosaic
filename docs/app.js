/**
 * Model Mosaic — app.js
 * Canvas renderer, layout engine, and interactivity.
 */

const state = {
  model: null,
  content: {},
  modelId: "mnist-8",
  view: "mosaic",
  palette: "inferno",
  normalization: "clipped",
  zoom: 1,
  minZoom: 1,
  maxZoom: 24,
  selectedLayer: null,
  isDragging: false,
  dragStartX: 0,
  dragStartY: 0,
  dragScrollLeft: 0,
  dragScrollTop: 0,
};

const MODELS = {
  "iris-mlp": { file: "models/iris-mlp.json", label: "Iris MLP — The Micro-Sprite" },
  "mnist-8": { file: "models/mnist-8.json", label: "MNIST-8 — The Portrait" },
  "cifar10-tiny": { file: "models/cifar10-tiny.json", label: "CIFAR-10 Tiny — The Landscape" },
};

const NORMALIZATION_LABELS = {
  global: "Global",
  clipped: "Clipped",
  "per-layer": "Per-layer",
};

let currentLayout = null;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function computeGrid(total) {
  const cols = Math.max(1, Math.ceil(Math.sqrt(total)));
  const rows = Math.max(1, Math.ceil(total / cols));
  return { cols, rows };
}

function flattenEntries(model) {
  const entries = [];
  for (const layer of model.layers) {
    for (let i = 0; i < layer.weights.length; i++) {
      entries.push({
        layerName: layer.name,
        shape: layer.shape,
        index: i,
        value: layer.weights[i],
      });
    }
  }
  return entries;
}

function collapseShape(shape) {
  const collapsed = shape.filter((dim) => dim !== 1);
  return collapsed.length ? collapsed : [1];
}

function unravelIndex(index, shape) {
  if (!shape.length) return [index];

  const coords = new Array(shape.length);
  let remainder = index;

  for (let i = shape.length - 1; i >= 0; i--) {
    const dim = shape[i];
    coords[i] = remainder % dim;
    remainder = Math.floor(remainder / dim);
  }

  return coords;
}

function formatTensorIndex(index, shape) {
  return unravelIndex(index, shape).map((coord) => `[${coord}]`).join("");
}

function percentileSorted(sorted, fraction) {
  if (!sorted.length) return 0;
  const idx = (sorted.length - 1) * fraction;
  const low = Math.floor(idx);
  const high = Math.ceil(idx);
  if (low === high) return sorted[low];
  const t = idx - low;
  return sorted[low] + (sorted[high] - sorted[low]) * t;
}

function computeRange(weights, mode = "global") {
  if (!weights.length) return { min: -1, max: 1 };

  let min;
  let max;

  if (mode === "clipped") {
    const sorted = [...weights].sort((a, b) => a - b);
    min = percentileSorted(sorted, 0.01);
    max = percentileSorted(sorted, 0.99);
  } else {
    min = Infinity;
    max = -Infinity;
    for (const weight of weights) {
      if (weight < min) min = weight;
      if (weight > max) max = weight;
    }
  }

  if (min === max) {
    const delta = Math.abs(min) || 1;
    min -= delta * 0.5;
    max += delta * 0.5;
  }

  return { min, max };
}

function buildRangeContext(layers) {
  const allWeights = layers.flatMap((layer) => layer.weights);
  return {
    global: computeRange(allWeights, "global"),
    clipped: computeRange(allWeights, "clipped"),
    perLayer: new Map(
      layers.map((layer) => [layer.name, computeRange(layer.weights, "global")])
    ),
  };
}

function getEntryRange(entry, rangeContext) {
  if (state.normalization === "per-layer") {
    return rangeContext.perLayer.get(entry.layerName) || rangeContext.global;
  }
  if (state.normalization === "clipped") {
    return rangeContext.clipped;
  }
  return rangeContext.global;
}

function normalizeEntry(entry, rangeContext) {
  const range = getEntryRange(entry, rangeContext);
  return normalizeWeight(entry.value, range.min, range.max);
}

function buildMosaicLayout(model, packSize = 1) {
  const entries = flattenEntries(model);
  const pixelCount = Math.ceil(entries.length / packSize);
  const { cols, rows } = computeGrid(pixelCount);
  const pixels = [];

  for (let i = 0; i < pixelCount; i++) {
    pixels.push(entries.slice(i * packSize, i * packSize + packSize));
  }

  return {
    mode: "mosaic",
    cols,
    rows,
    packSize,
    pixelCount,
    pixels,
    rangeContext: buildRangeContext(model.layers),
  };
}

function buildLayerShapeLayout(layer) {
  const n = layer.weights.length;
  const shape = collapseShape(layer.shape);
  const coords = new Array(n);
  const lookup = new Map();
  let cols;
  let rows;

  function setCoord(index, x, y) {
    coords[index] = { x, y };
    lookup.set(`${x},${y}`, index);
  }

  if (shape.length === 1) {
    cols = shape[0];
    rows = 1;
    for (let i = 0; i < n; i++) {
      setCoord(i, i, 0);
    }
  } else if (shape.length === 2) {
    rows = shape[0];
    cols = shape[1];
    let index = 0;
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols && index < n; col++) {
        setCoord(index, col, row);
        index += 1;
      }
    }
  } else if (shape.length === 4 && shape[2] <= 7 && shape[3] <= 7) {
    const [outer, inner, kernelH, kernelW] = shape;
    const gap = 1;
    cols = inner * kernelW + Math.max(0, inner - 1) * gap;
    rows = outer * kernelH + Math.max(0, outer - 1) * gap;
    let index = 0;

    for (let outerIdx = 0; outerIdx < outer; outerIdx++) {
      for (let innerIdx = 0; innerIdx < inner; innerIdx++) {
        for (let y = 0; y < kernelH; y++) {
          for (let x = 0; x < kernelW && index < n; x++) {
            const gridX = innerIdx * (kernelW + gap) + x;
            const gridY = outerIdx * (kernelH + gap) + y;
            setCoord(index, gridX, gridY);
            index += 1;
          }
        }
      }
    }
  } else {
    const grid = computeGrid(n);
    cols = grid.cols;
    rows = grid.rows;
    for (let i = 0; i < n; i++) {
      setCoord(i, i % cols, Math.floor(i / cols));
    }
  }

  return { cols, rows, coords, lookup };
}

function buildLayerLayout(model, selectedLayer) {
  const layers = selectedLayer
    ? model.layers.filter((layer) => layer.name === selectedLayer)
    : model.layers;

  const layout = {
    mode: "layers",
    blocks: [],
    totalHeight: 220,
    rangeContext: buildRangeContext(layers),
  };

  const canvasWidth = 640;
  const horizontalPadding = 16;
  const verticalPadding = 18;
  const labelHeight = 20;
  let yOffset = 0;

  for (const layer of layers) {
    const geometry = buildLayerShapeLayout(layer);
    const pixelSize = Math.max(
      2,
      Math.min(12, Math.floor((canvasWidth - horizontalPadding * 2) / Math.max(1, geometry.cols)))
    );
    const width = geometry.cols * pixelSize;
    const height = geometry.rows * pixelSize;
    const x = Math.max(horizontalPadding, Math.floor((canvasWidth - width) / 2));
    const y = yOffset + labelHeight;

    layout.blocks.push({
      layer,
      ...geometry,
      pixelSize,
      x,
      y,
      width,
      height,
      labelY: yOffset + 12,
    });

    yOffset = y + height + verticalPadding;
  }

  layout.totalHeight = Math.max(yOffset, 220);
  layout.canvasWidth = canvasWidth;
  return layout;
}

function renderMosaic(ctx, model, palette) {
  const packSize = palette === "aipixel" ? 3 : 1;
  const layout = buildMosaicLayout(model, packSize);
  const pixelSize = Math.max(2, Math.min(8, Math.floor(640 / layout.cols)));
  const canvasWidth = layout.cols * pixelSize;
  const canvasHeight = layout.rows * pixelSize;

  ctx.canvas.width = canvasWidth;
  ctx.canvas.height = canvasHeight;
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, canvasWidth, canvasHeight);

  const colormap = COLORMAPS[palette];

  for (let index = 0; index < layout.pixelCount; index++) {
    const group = layout.pixels[index];
    const x = index % layout.cols;
    const y = Math.floor(index / layout.cols);

    if (packSize === 3) {
      const rgb = [0, 1, 2].map((component) => {
        const entry = group[component];
        return entry
          ? Math.round(normalizeEntry(entry, layout.rangeContext) * 255)
          : 128;
      });
      ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    } else {
      const t = normalizeEntry(group[0], layout.rangeContext);
      const [r, g, b] = colormap.fn(t);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
    }

    ctx.fillRect(x * pixelSize, y * pixelSize, pixelSize, pixelSize);
  }

  layout.pixelSize = pixelSize;
  return layout;
}

function renderLayers(ctx, model) {
  const layout = buildLayerLayout(model, state.selectedLayer);
  const colormap = COLORMAPS[state.palette];

  ctx.canvas.width = layout.canvasWidth;
  ctx.canvas.height = layout.totalHeight;
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  ctx.font = "11px monospace";
  ctx.textBaseline = "middle";

  for (const block of layout.blocks) {
    ctx.fillStyle = "#888";
    ctx.fillText(`${block.layer.name} [${block.layer.shape.join("×")}]`, block.x, block.labelY);

    for (let i = 0; i < block.layer.weights.length; i++) {
      const point = block.coords[i];
      const entry = {
        layerName: block.layer.name,
        shape: block.layer.shape,
        index: i,
        value: block.layer.weights[i],
      };
      const t = normalizeEntry(entry, layout.rangeContext);
      const [r, g, b] = colormap.fn(t);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(
        block.x + point.x * block.pixelSize,
        block.y + point.y * block.pixelSize,
        block.pixelSize,
        block.pixelSize
      );
    }
  }

  return layout;
}

function renderError(message) {
  const canvas = document.getElementById("mosaic-canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = 640;
  canvas.height = 220;
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#ddd";
  ctx.font = "18px system-ui";
  ctx.fillText("Unable to load visualization", 24, 60);
  ctx.font = "14px system-ui";

  const lines = message.match(/.{1,70}(\s|$)/g) || [message];
  lines.slice(0, 4).forEach((line, idx) => {
    ctx.fillText(line.trim(), 24, 100 + idx * 24);
  });

  currentLayout = null;
}

function applyCanvasScale() {
  const canvas = document.getElementById("mosaic-canvas");
  canvas.style.width = `${Math.round(canvas.width * state.zoom)}px`;
  canvas.style.height = `${Math.round(canvas.height * state.zoom)}px`;
}

function setStatus(message = "", type = "info") {
  const status = document.getElementById("status-message");
  status.textContent = message;
  status.dataset.type = type;
}

function clearStatus() {
  setStatus("", "info");
}

function clearInfoStatus() {
  const status = document.getElementById("status-message");
  if (!status.textContent || status.dataset.type === "info") {
    clearStatus();
  }
}

function render() {
  const canvas = document.getElementById("mosaic-canvas");
  const ctx = canvas.getContext("2d");

  if (!state.model) {
    renderError("No model is currently loaded.");
    return;
  }

  if (state.view === "mosaic") {
    currentLayout = renderMosaic(ctx, state.model, state.palette);
  } else {
    currentLayout = renderLayers(ctx, state.model);
  }

  applyCanvasScale();
  updateInfoBar();
}

function describeMosaicGroup(group) {
  if (group.length === 1) {
    const entry = group[0];
    return `weight = ${entry.value.toFixed(6)}  |  ${entry.layerName}${formatTensorIndex(entry.index, entry.shape)}`;
  }

  return group
    .map(
      (entry, idx) =>
        `${["R", "G", "B"][idx]}=${entry.value.toFixed(6)} @ ${entry.layerName}${formatTensorIndex(entry.index, entry.shape)}`
    )
    .join("  |  ");
}

function handleHover(event) {
  const canvas = document.getElementById("mosaic-canvas");
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const cx = (event.clientX - rect.left) * scaleX;
  const cy = (event.clientY - rect.top) * scaleY;

  if (!currentLayout) {
    updateTooltip(null);
    return;
  }

  if (currentLayout.mode === "mosaic") {
    const col = Math.floor(cx / currentLayout.pixelSize);
    const row = Math.floor(cy / currentLayout.pixelSize);
    const pixelIndex = row * currentLayout.cols + col;

    if (pixelIndex >= 0 && pixelIndex < currentLayout.pixelCount) {
      const group = currentLayout.pixels[pixelIndex];
      updateTooltip(`${describeMosaicGroup(group)}  |  pixel [${col}, ${row}]`);
      return;
    }
  } else {
    for (const block of currentLayout.blocks) {
      if (
        cx >= block.x &&
        cx < block.x + block.width &&
        cy >= block.y &&
        cy < block.y + block.height
      ) {
        const localX = Math.floor((cx - block.x) / block.pixelSize);
        const localY = Math.floor((cy - block.y) / block.pixelSize);
        const key = `${localX},${localY}`;
        const weightIndex = block.lookup.get(key);

        if (weightIndex !== undefined) {
          const value = block.layer.weights[weightIndex];
          updateTooltip(
            `weight = ${value.toFixed(6)}  |  ${block.layer.name}${formatTensorIndex(weightIndex, block.layer.shape)}`
          );
          return;
        }
      }
    }
  }

  updateTooltip(null);
}

function updateTooltip(message) {
  const tooltip = document.getElementById("tooltip");
  tooltip.textContent = message || "Hover over a pixel to inspect its weight";
}

function updateInfoBar() {
  const info = document.getElementById("info-bar");
  if (!state.model || !currentLayout) {
    info.textContent = "";
    return;
  }

  const modelMeta = state.model.model;
  const parts = [
    modelMeta.name,
    `${modelMeta.total_params.toLocaleString()} weights`,
    `${NORMALIZATION_LABELS[state.normalization]} normalization`,
    `${Math.round(state.zoom * 100)}% zoom`,
  ];

  if (currentLayout.mode === "mosaic") {
    if (state.palette === "aipixel") {
      parts.push(`${currentLayout.pixelCount.toLocaleString()} RGB pixels`);
    }
    parts.push(`${currentLayout.cols}×${currentLayout.rows} grid`);
  } else {
    parts.push(`${currentLayout.blocks.length} layer panel${currentLayout.blocks.length === 1 ? "" : "s"}`);
  }

  info.textContent = parts.join(" · ");
}

function updatePlacard() {
  const placard = document.getElementById("placard");
  if (!state.model) {
    placard.style.display = "none";
    return;
  }

  const content = state.content[state.modelId];
  if (!content) {
    placard.style.display = "none";
    return;
  }

  const modelMeta = state.model.model;
  placard.style.display = "block";
  document.getElementById("placard-title").textContent =
    modelMeta.title || modelMeta.name;
  document.getElementById("placard-subtitle").textContent =
    `${modelMeta.name}, ${modelMeta.total_params.toLocaleString()} weights, ${modelMeta.year}`;
  document.getElementById("placard-text").textContent = content.placardText;
}

function showLayerAnnotation(layerName) {
  const annotationEl = document.getElementById("layer-annotation");
  if (!layerName) {
    annotationEl.style.display = "none";
    annotationEl.textContent = "";
    return;
  }

  const annotation = state.content[state.modelId]?.layerAnnotations?.[layerName];
  if (!annotation) {
    annotationEl.style.display = "none";
    annotationEl.textContent = "";
    return;
  }

  annotationEl.textContent = annotation;
  annotationEl.style.display = "block";
}

function updateLayerList() {
  const layerList = document.getElementById("layer-list");
  layerList.innerHTML = "";

  if (!state.model) return;

  const allButton = document.createElement("button");
  allButton.textContent = "All layers";
  allButton.className = state.selectedLayer === null ? "active" : "";
  allButton.onclick = () => {
    state.selectedLayer = null;
    updateLayerList();
    showLayerAnnotation(null);
    render();
  };
  layerList.appendChild(allButton);

  for (const layer of state.model.layers) {
    const button = document.createElement("button");
    button.textContent = `${layer.name} (${layer.weights.length})`;
    button.className = state.selectedLayer === layer.name ? "active" : "";
    button.onclick = () => {
      state.selectedLayer = layer.name;
      state.view = "layers";
      document.getElementById("view-select").value = "layers";
      if (state.palette === "aipixel") {
        state.palette = "inferno";
        document.getElementById("palette-select").value = "inferno";
        setStatus("ai-pixel palette works only in Mosaic view, so the palette was reset to Inferno.", "warning");
      }
      updateLayerList();
      showLayerAnnotation(layer.name);
      render();
    };
    layerList.appendChild(button);
  }
}

function exportPNG() {
  const canvas = document.getElementById("mosaic-canvas");
  canvas.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `model-mosaic-${state.modelId}-${state.palette}.png`;
    link.click();
    URL.revokeObjectURL(url);
  });
}

async function loadContent() {
  try {
    const response = await fetch("models/content.json");
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    state.content = await response.json();
  } catch (error) {
    state.content = {};
    setStatus(`Commentary failed to load: ${error.message}`, "warning");
  }
}

async function loadModel(modelId) {
  const info = MODELS[modelId];
  if (!info) return;

  setStatus(`Loading ${info.label}...`, "info");

  try {
    const response = await fetch(info.file);
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }

    const model = await response.json();
    state.model = model;
    state.modelId = modelId;
    state.selectedLayer = null;
    document.getElementById("model-select").value = modelId;
    updateLayerList();
    updatePlacard();
    showLayerAnnotation(null);
    render();
    clearInfoStatus();
  } catch (error) {
    if (!state.model) {
      renderError(error.message);
    } else {
      document.getElementById("model-select").value = state.modelId;
    }
    setStatus(`Failed to load ${info.label}: ${error.message}`, "error");
  }
}

function handleZoom(event) {
  event.preventDefault();

  const wrapper = document.getElementById("canvas-wrapper");
  const canvas = document.getElementById("mosaic-canvas");
  const oldZoom = state.zoom;
  const step = event.deltaY < 0 ? 1.15 : 1 / 1.15;
  const newZoom = clamp(oldZoom * step, state.minZoom, state.maxZoom);

  if (newZoom === oldZoom) return;

  const rect = wrapper.getBoundingClientRect();
  const pointerX = event.clientX - rect.left;
  const pointerY = event.clientY - rect.top;
  const contentX = wrapper.scrollLeft + pointerX;
  const contentY = wrapper.scrollTop + pointerY;
  const scale = newZoom / oldZoom;

  state.zoom = newZoom;
  applyCanvasScale();

  wrapper.scrollLeft = contentX * scale - pointerX;
  wrapper.scrollTop = contentY * scale - pointerY;
  updateInfoBar();

  if (canvas.width * state.zoom > wrapper.clientWidth || canvas.height * state.zoom > wrapper.clientHeight) {
    setStatus("Scroll to zoom. Drag the canvas to pan.", "info");
  }
}

function startPan(event) {
  if (event.button !== 0) return;
  const wrapper = document.getElementById("canvas-wrapper");
  state.isDragging = true;
  state.dragStartX = event.clientX;
  state.dragStartY = event.clientY;
  state.dragScrollLeft = wrapper.scrollLeft;
  state.dragScrollTop = wrapper.scrollTop;
  wrapper.classList.add("dragging");
  event.preventDefault();
}

function onPan(event) {
  if (!state.isDragging) return;
  const wrapper = document.getElementById("canvas-wrapper");
  wrapper.scrollLeft = state.dragScrollLeft - (event.clientX - state.dragStartX);
  wrapper.scrollTop = state.dragScrollTop - (event.clientY - state.dragStartY);
}

function endPan() {
  if (!state.isDragging) return;
  state.isDragging = false;
  document.getElementById("canvas-wrapper").classList.remove("dragging");
}

async function init() {
  const modelSelect = document.getElementById("model-select");
  for (const [id, info] of Object.entries(MODELS)) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = info.label;
    modelSelect.appendChild(option);
  }
  modelSelect.value = state.modelId;
  modelSelect.addEventListener("change", (event) => loadModel(event.target.value));

  const paletteSelect = document.getElementById("palette-select");
  for (const [id, colormap] of Object.entries(COLORMAPS)) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = colormap.name;
    paletteSelect.appendChild(option);
  }
  paletteSelect.value = state.palette;
  paletteSelect.addEventListener("change", (event) => {
    state.palette = event.target.value;
    if (state.view === "layers" && state.palette === "aipixel") {
      state.view = "mosaic";
      document.getElementById("view-select").value = "mosaic";
      setStatus("ai-pixel palette works only in Mosaic view, so the view was switched to Mosaic.", "warning");
    }
    render();
  });

  const normalizationSelect = document.getElementById("normalization-select");
  normalizationSelect.addEventListener("change", (event) => {
    state.normalization = event.target.value;
    render();
  });

  const viewSelect = document.getElementById("view-select");
  viewSelect.addEventListener("change", (event) => {
    state.view = event.target.value;
    if (state.view === "layers" && state.palette === "aipixel") {
      state.palette = "inferno";
      document.getElementById("palette-select").value = "inferno";
      setStatus("ai-pixel palette works only in Mosaic view, so the palette was reset to Inferno.", "warning");
    } else {
      clearStatus();
    }
    if (state.view !== "layers") {
      showLayerAnnotation(null);
    } else if (state.selectedLayer) {
      showLayerAnnotation(state.selectedLayer);
    }
    render();
  });

  document.getElementById("export-btn").addEventListener("click", exportPNG);

  document.getElementById("placard-toggle").addEventListener("click", () => {
    const body = document.getElementById("placard-body");
    const button = document.getElementById("placard-toggle");
    if (body.style.display === "none") {
      body.style.display = "block";
      button.textContent = "▾ Hide commentary";
    } else {
      body.style.display = "none";
      button.textContent = "▸ Show commentary";
    }
  });

  const wrapper = document.getElementById("canvas-wrapper");
  const canvas = document.getElementById("mosaic-canvas");
  wrapper.addEventListener("wheel", handleZoom, { passive: false });
  wrapper.addEventListener("mousedown", startPan);
  window.addEventListener("mousemove", onPan);
  window.addEventListener("mouseup", endPan);
  canvas.addEventListener("mousemove", handleHover);
  canvas.addEventListener("mouseleave", () => updateTooltip(null));

  await loadContent();
  await loadModel(state.modelId);
}

document.addEventListener("DOMContentLoaded", () => {
  init().catch((error) => {
    renderError(error.message);
    setStatus(`Initialization failed: ${error.message}`, "error");
  });
});
