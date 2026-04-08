"""
Train small PyTorch models and export their weights as JSON.

Models:
  - Iris MLP (4→8→3): 67 parameters
  - CIFAR-10 Tiny CNN: ~15K parameters

Usage:
    python extract_torch.py --model iris --output ../docs/models/iris-mlp.json
    python extract_torch.py --model cifar10 --output ../docs/models/cifar10-tiny.json
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Make training runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class IrisMLP(nn.Module):
    """Tiny MLP for Iris classification: 4→8→3 = 67 parameters."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)   # 4*8 + 8 = 40
        self.fc2 = nn.Linear(8, 3)   # 8*3 + 3 = 27
        # Total: 67 parameters

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class CIFAR10TinyCNN(nn.Module):
    """Tiny CNN for CIFAR-10: ~15K parameters."""

    def __init__(self):
        super().__init__()
        # Conv layers: small filters, few channels
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)    # 3*8*3*3 + 8 = 224
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)   # 8*16*3*3 + 16 = 1168
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)  # 16*32*3*3 + 32 = 4640
        self.pool = nn.MaxPool2d(2, 2)
        # After 3 pools: 32×4×4 = 512
        self.fc1 = nn.Linear(512, 16)                  # 512*16 + 16 = 8208
        self.fc2 = nn.Linear(16, 10)                   # 16*10 + 10 = 170
        # Total: ~14,410 parameters

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # 32→16
        x = self.pool(torch.relu(self.conv2(x)))   # 16→8
        x = self.pool(torch.relu(self.conv3(x)))   # 8→4
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_iris(epochs=500, seed=42):
    """Train Iris MLP on the classic Iris dataset."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    set_seed(seed)

    data = load_iris()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
    )

    model = IrisMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1)
        accuracy = (preds == y_test_t).float().mean().item()
        print(f"Iris MLP — Test accuracy: {accuracy:.1%}")

    return model, accuracy


def train_cifar10(epochs=20, data_root="./models/data", seed=42):
    """Train CIFAR-10 Tiny CNN on the real CIFAR-10 dataset."""
    set_seed(seed)
    os.makedirs(data_root, exist_ok=True)
    model = CIFAR10TinyCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    cifar_dir = os.path.join(data_root, "cifar-10-batches-py")
    download = not os.path.exists(cifar_dir)
    if download:
        print("Downloading CIFAR-10 dataset...")
    else:
        print(f"Using local CIFAR-10 dataset in {data_root}")

    try:
        trainset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=download,
            transform=transform,
        )
        testset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=download,
            transform=transform,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to access the real CIFAR-10 dataset. "
            f"Original error: {e}. "
            f"Try running Python's Install Certificates.command on macOS, or manually "
            f"download/extract CIFAR-10 into {data_root} so torchvision can load it."
        ) from e

    generator = torch.Generator().manual_seed(seed)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = correct / total
    print(f"CIFAR-10 Tiny CNN — Test accuracy: {accuracy:.1%}")

    return model, accuracy


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------

LAYER_DESCRIPTIONS = {
    "iris": {
        "fc1.weight": "First hidden layer — transforms 4 flower measurements into 8 internal features",
        "fc1.bias": "Bias for first hidden layer",
        "fc2.weight": "Output layer — maps 8 features to 3 species predictions",
        "fc2.bias": "Bias for output layer",
    },
    "cifar10": {
        "conv1.weight": "First convolutional layer — detects basic color edges and gradients",
        "conv1.bias": "Bias for first conv layer",
        "conv2.weight": "Second convolutional layer — combines edges into textures and corners",
        "conv2.bias": "Bias for second conv layer",
        "conv3.weight": "Third convolutional layer — recognizes parts of objects (wheels, wings, ears)",
        "conv3.bias": "Bias for third conv layer",
        "fc1.weight": "Dense layer — combines spatial features into abstract object representations",
        "fc1.bias": "Bias for dense layer",
        "fc2.weight": "Output layer — maps 16 features to 10 category predictions",
        "fc2.bias": "Bias for output layer",
    },
}


def extract_weights(model: nn.Module, model_type: str) -> list[dict]:
    """Extract all weight tensors from a PyTorch model."""
    layers = []
    descriptions = LAYER_DESCRIPTIONS.get(model_type, {})

    for name, param in model.named_parameters():
        tensor = param.detach().cpu().numpy()
        weights = tensor.flatten().tolist()

        layers.append({
            "name": name,
            "description": descriptions.get(name, ""),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "min": round(float(np.min(tensor)), 6),
            "max": round(float(np.max(tensor)), 6),
            "mean": round(float(np.mean(tensor)), 6),
            "std": round(float(np.std(tensor)), 6),
            "weights": [round(w, 6) for w in weights],
        })

    return layers


MODEL_META = {
    "iris": {
        "name": "Iris MLP",
        "title": "The Micro-Sprite",
        "description": "Classifies iris flowers into 3 species from 4 petal/sepal measurements",
        "source": "Custom (trained in-repo)",
        "source_url": "",
        "architecture": "MLP (4→8→3)",
        "year": 2026,
    },
    "cifar10": {
        "name": "CIFAR-10 Tiny CNN",
        "title": "The Landscape",
        "description": "Classifies tiny 32×32 color photos into 10 categories (airplane, car, bird, etc.)",
        "source": "Custom (trained in-repo)",
        "source_url": "",
        "architecture": "CNN (conv3×3→pool → conv3×3→pool → conv3×3→pool → dense→10)",
        "year": 2026,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Train small models and extract weights")
    parser.add_argument("--model", required=True, choices=["iris", "cifar10"], help="Model to train")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training")
    parser.add_argument("--data-root", default="./models/data", help="Directory to cache CIFAR-10 data")
    args = parser.parse_args()

    if args.model == "iris":
        print("Training Iris MLP...")
        model, accuracy = train_iris(epochs=args.epochs or 500, seed=args.seed)
    elif args.model == "cifar10":
        print("Training CIFAR-10 Tiny CNN...")
        model, accuracy = train_cifar10(
            epochs=args.epochs or 20,
            data_root=args.data_root,
            seed=args.seed,
        )

    print(f"Extracting weights...")
    layers = extract_weights(model, args.model)

    total_params = sum(len(layer["weights"]) for layer in layers)
    meta = MODEL_META[args.model].copy()
    meta["total_params"] = total_params
    meta["accuracy"] = round(float(accuracy), 6)
    meta["seed"] = args.seed

    output = {
        "model": meta,
        "layers": layers,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f)

    cols = int(np.ceil(np.sqrt(total_params)))
    rows = int(np.ceil(total_params / cols))
    print(f"Extracted {total_params} weights across {len(layers)} layers")
    print(f"Grid size: {cols}×{rows}")
    print(f"Saved to {args.output}")

    for layer in layers:
        print(f"  {layer['name']:30s} shape={layer['shape']}  ({len(layer['weights'])} weights)")


if __name__ == "__main__":
    main()
