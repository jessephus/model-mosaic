"""
Extract weights from an ONNX model on HuggingFace and export as JSON.

Usage:
    python extract_onnx.py --model onnxmodelzoo/mnist-8 --output ../docs/models/mnist-8.json
"""

import argparse
import json
import os
import sys

import numpy as np
import onnx
from huggingface_hub import hf_hub_download


def download_model(repo_id: str, cache_dir: str = "models") -> str:
    """Download an ONNX model from HuggingFace Hub."""
    os.makedirs(cache_dir, exist_ok=True)

    # Try common ONNX file paths
    candidates = [
        "model.onnx",
        "mnist-8.onnx",
        "model/mnist-8.onnx",
    ]

    # First, try to list repo files
    from huggingface_hub import list_repo_files

    try:
        files = list_repo_files(repo_id)
        onnx_files = [f for f in files if f.endswith(".onnx")]
        if onnx_files:
            candidates = onnx_files + candidates
    except Exception:
        pass

    for filename in candidates:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
            )
            print(f"Downloaded: {filename}")
            return path
        except Exception:
            continue

    print(f"Error: Could not find ONNX file in {repo_id}", file=sys.stderr)
    print(f"Available files: {onnx_files if 'onnx_files' in dir() else 'unknown'}", file=sys.stderr)
    sys.exit(1)


def extract_weights(model_path: str) -> list[dict]:
    """Extract all weight tensors from an ONNX model."""
    model = onnx.load(model_path)
    layers = []

    for initializer in model.graph.initializer:
        tensor = onnx.numpy_helper.to_array(initializer)
        weights = tensor.flatten().tolist()

        layers.append({
            "name": initializer.name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "min": round(float(np.min(tensor)), 6),
            "max": round(float(np.max(tensor)), 6),
            "mean": round(float(np.mean(tensor)), 6),
            "std": round(float(np.std(tensor)), 6),
            "weights": [round(w, 6) for w in weights],
        })

    return layers


def build_output(layers: list[dict], model_meta: dict) -> dict:
    """Build the final JSON structure."""
    total_params = sum(len(layer["weights"]) for layer in layers)

    return {
        "model": {
            **model_meta,
            "total_params": total_params,
        },
        "layers": layers,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract ONNX model weights to JSON")
    parser.add_argument("--model", required=True, help="HuggingFace repo ID (e.g., onnxmodelzoo/mnist-8)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--name", default=None, help="Model display name")
    parser.add_argument("--title", default=None, help="Gallery piece title")
    parser.add_argument("--description", default="", help="Model description")
    parser.add_argument("--architecture", default="", help="Architecture description")
    parser.add_argument("--year", type=int, default=2019, help="Year model was published")
    args = parser.parse_args()

    print(f"Downloading {args.model} from HuggingFace...")
    model_path = download_model(args.model)

    print(f"Extracting weights from {model_path}...")
    layers = extract_weights(model_path)

    model_meta = {
        "name": args.name or args.model.split("/")[-1],
        "title": args.title or args.name or args.model.split("/")[-1],
        "description": args.description,
        "source": args.model,
        "source_url": f"https://huggingface.co/{args.model}",
        "architecture": args.architecture,
        "year": args.year,
    }

    output = build_output(layers, model_meta)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f)

    total = output["model"]["total_params"]
    grid_side = int(np.ceil(np.sqrt(total)))
    print(f"Extracted {total} weights across {len(layers)} layers")
    print(f"Grid size: ~{grid_side}×{grid_side}")
    print(f"Saved to {args.output}")

    for layer in layers:
        print(f"  {layer['name']:30s} shape={layer['shape']}  ({len(layer['weights'])} weights)")


if __name__ == "__main__":
    main()
