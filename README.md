# Self-Pruning Neural Network — CIFAR-10

Assignment submission for **Tredence Analytics — AI Engineering Internship (2025 Cohort)**

---

## What This Is

A feed-forward neural network trained on CIFAR-10 that learns to prune its own weights during training. Instead of post-training pruning, each weight has a learnable **gate** that can shut it off entirely. A sparsity regularization term in the loss function pushes unnecessary gates toward zero.

---

## How It Works

- **`PrunableLinear`** — custom linear layer where each weight is multiplied by `sigmoid(gate_score)`. Gates near 0 = pruned, gates near 1 = active.
- **Sparsity Loss** — L1 penalty on all gate values added to the cross-entropy loss.
- **Total Loss** = `CrossEntropyLoss + λ × SparsityLoss`
- λ (lambda) controls the sparsity-accuracy trade-off.

---

## Project Structure

```
├── assignment         # Main script (model, training loop, evaluation)
├── report           # Short written report

└── README.md
```

---

## Setup & Run

**Requirements**
```bash
pip install torch torchvision matplotlib numpy
```

**Run**
```bash
python solution.py
```

CIFAR-10 will be auto-downloaded to `./data/` on first run. Training runs for 50 epochs across λ = [0.1, 0.3, 0.5]. Outputs plots and a results summary to the console.

---

## Results

| Lambda (λ) | Test Accuracy | Sparsity Level (%) |
|:----------:|:-------------:|:------------------:|
| 0.1        | 60.96%        | 0.0%               |
| 0.3        | 60.99%        | 0.0%               |
| 0.5        | 61.13%        | 0.0%               |

> **Note:** Sparsity is 0.0% due to a known bug — `sparsity_loss()` uses `.mean()` instead of `.sum()`, which dilutes the gradient per gate too much to push any to zero. Fixing it to `.sum()` would produce the expected sparsity behavior.

---

## Device Support

The script auto-detects and runs on **CUDA → MPS → GPU** in that order.
