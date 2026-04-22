# =============================================================================
# Self-Pruning Neural Network for CIFAR-10
# Technical Case Study — Tredence Analytics AI Engineering Internship
#
# Features:
#   - Custom PrunableLinear layer with learnable sigmoid gates
#   - L1-based sparsity regularization loss
#   - Async training and evaluation loop (asyncio + PyTorch)
#   - Automated experimentation across three lambda values
#   - Gate distribution plot saved to gate_distribution.png
# =============================================================================

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no GUI window needed

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import asyncio
import sys
from typing import List
from dataclasses import dataclass


def log(msg: str):
    """Flush-safe print so output appears immediately in redirected streams."""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Data class to hold per-experiment results
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    lambda_val: float
    accuracy: float
    sparsity: float
    gate_values: np.ndarray


# ---------------------------------------------------------------------------
# Part 1: PrunableLinear — custom linear layer with learnable gates
# ---------------------------------------------------------------------------

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that adds a learnable gate_scores
    tensor of the same shape as the weight matrix.

    Forward pass:
        gates        = sigmoid(gate_scores)          # values in (0, 1)
        pruned_weight = weight * gates               # element-wise mask
        output        = pruned_weight @ input.T + bias

    When a gate value collapses to ~0, the corresponding weight is effectively
    removed from the network (pruned). Gradients flow through both `weight`
    and `gate_scores` via standard autograd.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Learnable gate scores — same shape as weight, updated by the optimizer
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))

        self._reset_parameters()

    def _reset_parameters(self):
        # Kaiming uniform init for weights (standard practice)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        # Initialize gate scores to 2.0 → sigmoid(2.0) ≈ 0.88
        # This means all connections start mostly open; the sparsity loss
        # will drive them toward 0 during training.
        nn.init.constant_(self.gate_scores, 2.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Step 1: map gate_scores → [0, 1] via sigmoid
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: element-wise multiply weights with gates
        pruned_weights = self.weight * gates

        # Step 3: standard linear operation with pruned weights
        return F.linear(input, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return current gate values (detached from graph for inspection)."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores)


# ---------------------------------------------------------------------------
# Network: feed-forward MLP built entirely from PrunableLinear layers
# ---------------------------------------------------------------------------

class PrunableMLP(nn.Module):
    """
    Feed-forward MLP for CIFAR-10 classification.
    Architecture: 3072 → 512 → 256 → 10
    All linear layers are PrunableLinear so every weight has a learnable gate.
    """

    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int):
        super(PrunableMLP, self).__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(PrunableLinear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(PrunableLinear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten (B, 3, 32, 32) → (B, 3072)
        return self.model(x)

    def get_all_gates(self) -> List[torch.Tensor]:
        """Collect gate tensors from every PrunableLinear layer."""
        gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates.append(module.get_gates())
        return gates


# ---------------------------------------------------------------------------
# Part 2: Sparsity loss helper
# ---------------------------------------------------------------------------

def compute_sparsity_loss(model: PrunableMLP) -> torch.Tensor:
    """
    SparsityLoss = sum of all gate values across all PrunableLinear layers.

    Because gates = sigmoid(gate_scores) are always in (0, 1), their absolute
    value equals themselves, so this is the L1 norm of all gates.
    Minimising this term pushes gate values toward exactly 0 (pruning).
    """
    sparsity_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)  # keep in graph for grad
            sparsity_loss = sparsity_loss + torch.sum(gates)
    return sparsity_loss


# ---------------------------------------------------------------------------
# Part 3: Training loop (async — demonstrates asyncio per JD requirements)
# ---------------------------------------------------------------------------

async def train_one_epoch(
    model: PrunableMLP,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    lambda_val: float,
    device: torch.device,
) -> float:
    """
    One full pass over the training set.
    Total Loss = CrossEntropyLoss + lambda * SparsityLoss
    """
    model.train()
    running_loss = 0.0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Classification loss
        ce_loss = criterion(outputs, labels)

        # Sparsity regularization loss (L1 on all gates)
        sparsity_loss = compute_sparsity_loss(model)

        # Combined loss — lambda controls sparsity vs accuracy trade-off
        total_loss = ce_loss + lambda_val * sparsity_loss

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    return running_loss / len(loader)


async def evaluate(
    model: PrunableMLP,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Return test accuracy (%) on the given data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Sparsity evaluation
# ---------------------------------------------------------------------------

def calculate_sparsity(model: PrunableMLP, threshold: float = 1e-2) -> float:
    """
    Sparsity level = percentage of weights whose gate value < threshold.
    A gate below 1e-2 is considered effectively pruned.
    """
    total_params = 0
    pruned_params = 0
    for gates in model.get_all_gates():
        total_params += gates.numel()
        pruned_params += (gates < threshold).sum().item()
    return 100.0 * pruned_params / total_params


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

async def run_experiment(
    lambda_val: float,
    epochs: int = 10,  # 10 epochs for submission-quality results
) -> ExperimentResult:
    """
    Train a PrunableMLP on CIFAR-10 with the given lambda, then evaluate.
    Returns an ExperimentResult with accuracy, sparsity, and gate values.
    """
    log(f"\n{'='*55}")
    log(f"  Starting experiment  |  Lambda = {lambda_val}")
    log(f"{'='*55}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  Device: {device}")

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )

    # Model, optimizer, loss
    model = PrunableMLP(input_size=3072, hidden_sizes=[512, 256], num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        loss = await train_one_epoch(
            model, trainloader, optimizer, criterion, lambda_val, device
        )
        log(f"  Lambda {lambda_val} | Epoch {epoch + 1:>2}/{epochs} | Loss: {loss:.4f}")

    # Evaluation
    accuracy = await evaluate(model, testloader, device)
    sparsity = calculate_sparsity(model)

    log(f"\n  Lambda {lambda_val} → Accuracy: {accuracy:.2f}%  |  Sparsity: {sparsity:.2f}%")

    # Collect all gate values for the distribution plot
    all_gates = []
    for gates in model.get_all_gates():
        all_gates.extend(gates.cpu().numpy().flatten())

    return ExperimentResult(
        lambda_val=lambda_val,
        accuracy=accuracy,
        sparsity=sparsity,
        gate_values=np.array(all_gates),
    )


# ---------------------------------------------------------------------------
# Main orchestrator — runs experiments sequentially, then reports
# ---------------------------------------------------------------------------

async def main():
    # Three lambda values: low / medium / high
    # Low  → minimal pruning, higher accuracy
    # High → aggressive pruning, lower accuracy
    lambdas = [0.001, 0.01, 0.1]

    results: List[ExperimentResult] = []

    # Run experiments sequentially — asyncio is used for the training/eval
    # coroutines (non-blocking I/O friendly design), but GPU-bound PyTorch
    # work runs on a single thread so parallel execution is not meaningful here.
    for lam in lambdas:
        result = await run_experiment(lam)
        results.append(result)

    # -----------------------------------------------------------------------
    # Results table
    # -----------------------------------------------------------------------
    log("\n" + "=" * 55)
    log(f"  {'Lambda':<12} | {'Test Accuracy (%)':<20} | {'Sparsity (%)':<12}")
    log("-" * 55)
    for r in results:
        log(f"  {r.lambda_val:<12} | {r.accuracy:<20.2f} | {r.sparsity:<12.2f}")
    log("=" * 55)

    # -----------------------------------------------------------------------
    # Gate value distribution plot — all three lambdas side by side
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Gate Value Distributions Across Lambda Values\n"
        "(Sigmoid outputs — spike near 0 = pruned weights)",
        fontsize=13
    )

    for ax, r in zip(axes, results):
        ax.hist(r.gate_values, bins=50, color='skyblue', edgecolor='black')
        ax.set_title(
            f"λ = {r.lambda_val}\n"
            f"Acc: {r.accuracy:.2f}%  |  Sparsity: {r.sparsity:.2f}%"
        )
        ax.set_xlabel("Gate Value [0–1]")
        ax.set_ylabel("Frequency")
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save plot — no plt.show() since we're running non-interactively
    plt.savefig('gate_distribution.png', dpi=150, bbox_inches='tight')
    log("\n  Plot saved → gate_distribution.png")
    log("  Done.")


if __name__ == "__main__":
    asyncio.run(main())
