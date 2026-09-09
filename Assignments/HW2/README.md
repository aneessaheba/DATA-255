# HW2 - Neural Network with One Hidden Layer

A 784 -> 200 -> 1 neural network built from scratch in PyTorch, used to compute the
binary cross entropy loss and accuracy on a batch of the binary MNIST dataset.

## Dataset

| File | Samples |
|---|---|
| `mnist_train_binary.csv` | 11,339 |
| `mnist_test_binary.csv` | 1,850 |

Each row holds 785 values: the label followed by 784 pixel values (0-255) of a 28x28
grayscale image. **Label 1 = digit 5, label 0 = digit 6.**

## Contents

- `HW2.ipynb` - the full solution with outputs saved.
- `mnist_train_binary.csv`, `mnist_test_binary.csv` - the dataset.

## Solution overview

**1. Build the neural network.**

| Layer | Neurons | Weight matrix | Activation |
|---|---|---|---|
| Input | 784 | - | - |
| Hidden | 200 | `W1`: 200 x 784 | ReLU |
| Output | 1 | `W2`: 1 x 200 | Sigmoid |

No `nn.Linear` is used - both layers are plain matrix multiplications with hand-written
weights. ReLU and Sigmoid are both defined from scratch. `W1` uses He initialization,
`W2` uses Xavier initialization, and the random seed is fixed to 0.

Forward pass for a batch `X` of shape `(m, 784)`:

```
a1 = relu(X @ W1.T)      # (m, 200)
a2 = sigmoid(a1 @ W2.T)  # (m, 1)
```

**2. Calculate loss and accuracy.** Pixels are normalised from [0, 255] to [0, 1], and the
first 64 training samples are pushed through the network to get predictions `a2`. Binary
cross entropy loss and accuracy (threshold 0.5) are computed on these 64 samples.

## Result

| Metric | Value |
|---|---|
| Loss | 0.6675 |
| Accuracy | 0.65625 |

## Constraints followed

- PyTorch only - no NumPy, no TensorFlow.
- No `torch.nn.Linear`; weights are raw tensors and the forward pass is written by hand.
- ReLU and Sigmoid defined from scratch.
- Random seed set to 0, weights initialised with He/Xavier, inputs normalised to [0, 1].

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch pandas jupyter
jupyter notebook HW2.ipynb
```

Run the cells in order from the top.
