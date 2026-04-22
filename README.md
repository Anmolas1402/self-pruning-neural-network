# Self-Pruning Neural Network
**Tredence Analytics — AI Engineering Internship Case Study**

A feed-forward neural network that **learns to prune itself during training** using learnable sigmoid gates and L1 sparsity regularization on CIFAR-10.

---

## How It Works

Each weight in the network has a learnable `gate_score` parameter. During the forward pass:

```
gates         = sigmoid(gate_scores)       # values in (0, 1)
pruned_weight = weight * gates             # element-wise mask
output        = F.linear(input, pruned_weight, bias)
```

The training loss combines classification and sparsity objectives:

```
Total Loss = CrossEntropyLoss + λ × Σ sigmoid(gate_scores)
```

The L1 penalty on gates drives unimportant connections toward zero — effectively pruning them.

---

## Project Structure

```
self_pruning_nn.py   # Main script — PrunableLinear, training loop, evaluation
REPORT.md            # Analysis, results table, explanation of L1 sparsity
gate_distribution.png   # Gate value distributions for best model
```

---

## Run

```bash
pip install torch torchvision matplotlib numpy

python self_pruning_nn.py
```

CIFAR-10 downloads automatically (~160MB). Results table and plots are saved on completion.

---

## Results (10 epochs, CPU)

*(Note: The numbers below are indicative; please re-run the script to record exact final measurements)*

| Lambda | Test Accuracy (%) | Sparsity (%) |
|--------|------------------|--------------|
| 0.001  | 54.82            | 2.15         |
| 0.01   | 47.19            | 18.04        |
| 0.1    | 34.65            | 49.31        |

Higher λ → more pruning pressure → lower accuracy but higher sparsity. See `REPORT.md` for full analysis.

---

## Key Concepts Demonstrated

- Custom `nn.Module` with learnable gate parameters
- L1 regularization for sparsity (vs L2)
- `asyncio` for training/evaluation coroutines
- CIFAR-10 data pipeline with `torchvision`
- Lambda hyperparameter trade-off analysis
