# HW1 - Neural Network with One Hidden Layer

A 784 -> 100 -> 1 neural network built from scratch in PyTorch and run for one forward
pass on a binary subset of MNIST.

## Dataset

| File | Samples |
|---|---|
| `mnist_train_binary.csv` | 11,339 |
| `mnist_test_binary.csv` | 1,850 |

Each row holds 785 values: the label followed by 784 pixel values (0-255) of a 28x28
grayscale image. **Label 1 = digit 5, label 0 = digit 6.**

## Contents

- `HW1.ipynb` - the full solution with outputs saved.
- `mnist_train_binary.csv`, `mnist_test_binary.csv` - the dataset.

## Solution overview

**1. Load the dataset and display sample images.** The csv files are parsed with Python's
built-in `csv` module and loaded directly into `torch.Tensor` objects. Column 0 is the
label, columns 1-784 are the pixels, so the loader returns `X` of shape `(n, 784)` and `y`
of shape `(n,)`. Nine training images are reshaped from 784 pixels back to 28x28 and shown
in a 3x3 grid with their labels.

**2. Build the neural network.**

| Layer | Neurons | Weight matrix | Activation |
|---|---|---|---|
| Input | 784 | - | - |
| Hidden | 100 | `W1`: 784 x 100 | sigmoid |
| Output | 1 | `W2`: 100 x 1 | sigmoid |

Bias terms are omitted, so each layer is a plain matrix multiplication. Sigmoid is written
from scratch as `1 / (1 + torch.exp(-z))`. Weights are initialised non-zero with
`torch.randn(...) * 0.01`; the small scale keeps the pre-activations near 0, where the
sigmoid is steepest, instead of saturating it.

Forward pass for a batch `X` of shape `(m, 784)`:

```
a1 = sigmoid(X  @ W1)    # (m, 100)
a2 = sigmoid(a1 @ W2)    # (m, 1)
```

**3. Calculate a2 for the first 64 training samples.** Pixels are normalised from [0, 255]
to [0, 1], the seed is reset to 0, and the first 64 training rows are pushed through the
network to give `a2` of shape `(64, 1)`.

## Result

All 64 outputs land between 0.4936 and 0.4951. This is expected: the network is untrained
and the weights are near zero, so `a1 @ W2` is approximately 0 and `sigmoid(0) = 0.5`. The
small differences between samples come from the different input images.

## Constraints followed

- PyTorch only - no NumPy, no TensorFlow.
- No `torch.nn.Linear`; weights are raw tensors and the forward pass is written by hand.
- Sigmoid defined from scratch.
- Random seed set to 0, weights initialised to non-zero values, inputs normalised to [0, 1].

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch matplotlib jupyter
jupyter notebook HW1.ipynb
```

Run the cells in order from the top.
