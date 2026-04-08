# Model Mosaic

**Every pixel is a neural network weight.**

Model Mosaic visualizes the learned weights of neural networks as interactive pixel art. Each pixel in the image represents a single weight value, mapped to a color through configurable colormaps — from perceptually uniform scientific palettes to retro pixel art palettes like the NES and Game Boy.

Browse three pre-loaded "gallery pieces," each accompanied by curatorial commentary that describes both the visual beauty and mathematical significance of the patterns.

🔗 **[View the gallery →](https://jessephus.github.io/model-mosaic/)**

## Gallery

| Piece | Model | Weights | Grid |
|---|---|---|---|
| 🎨 *The Micro-Sprite* | Iris MLP (4→8→3) | 59 | 8×8 |
| 🖼️ *The Portrait* | MNIST-8 CNN | ~6,500 | ~81×81 |
| 🏔️ *The Landscape* | CIFAR-10 Tiny CNN | ~15,000 | ~122×122 |

## How It Works

```
ai-pixel:       3 weights  → 1 pixel  (weight → R/G/B channel)
Model Mosaic:   1 weight   → 1 pixel  (weight → colormap color)
```

Each weight (a floating-point number the model learned during training) is mapped to a color. Negative weights skew cool, positive weights skew warm, and near-zero weights fade to neutral. The visual patterns that emerge aren't random — they reflect what the model *learned*.

## Features

- **Two view modes:** Mosaic (all weights in one grid) and Layer (per-layer panels)
- **7 palettes:** Inferno, Coolwarm, Grayscale, NES, PICO-8, Game Boy, ai-pixel RGB
- **Interactive:** Hover to inspect individual weights, scroll to zoom, drag to pan
- **Gallery placards:** Art-historian-meets-ML-researcher commentary per model
- **Export:** Download any visualization as a PNG

## Inspiration

Inspired by Dan Velton's [ai-pixel](https://github.com/dvelton/ai-pixel), which encodes an entire 3-parameter ML model into a single colored pixel. Model Mosaic extends that idea to thousands of weights.

## Tech

- Vanilla HTML/CSS/JS — no framework
- HTML5 Canvas for pixel-level rendering
- Python scripts for weight extraction from ONNX (HuggingFace) and PyTorch models
- GitHub Pages for hosting

## License

MIT
