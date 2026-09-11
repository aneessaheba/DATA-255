# HW3 - Deep Neural Network from Scratch

A 784 -> 200 -> 1 neural network built and trained entirely from scratch in PyTorch to
classify the binary MNIST dataset. Forward pass, binary cross entropy loss,
backpropagation and the gradient descent update are all written by hand.

## Dataset

| File | Samples |
|---|---|
| `mnist_train_binary.csv` | 11,339 |
| `mnist_test_binary.csv` | 1,850 |

Each row holds 785 values: the label followed by 784 pixel values (0-255) of a 28x28
grayscale image. **Label 1 = digit 5, label 0 = digit 6.**

## Contents

- `HW3.ipynb` - the full solution with outputs saved.
- `mnist_train_binary.csv`, `mnist_test_binary.csv` - the dataset.

## Solution overview

**1. Build the network.**

| Layer | Neurons | Weight matrix | Activation |
|---|---|---|---|
| Input | 784 | - | - |
| Hidden | 200 | `W1`: 784 x 200 | ReLU |
| Output | 1 | `W2`: 200 x 1 | Sigmoid |

No `nn.Linear` is used, both layers are plain matrix multiplications with hand written
weights. ReLU and Sigmoid are defined from scratch. `W1` uses He initialization, `W2`
uses Xavier initialization, and the random seed is fixed to 42.

Forward pass for a batch `X` of shape `(m, 784)`:

```
Z1 = X @ W1 + b1
A1 = relu(Z1)
Z2 = A1 @ W2 + b2
A2 = sigmoid(Z2)
```

**2. Loss, accuracy and backward propagation.** Binary cross entropy and accuracy
(threshold 0.5) are implemented from scratch. The gradients are derived by hand:

```
dZ2 = (A2 - y) / m
dW2 = A1.T @ dZ2                      db2 = sum(dZ2)
dZ1 = (dZ2 @ W2.T) * relu'(Z1)
dW1 = X.T @ dZ1                       db1 = sum(dZ1)
```

**3. Training.** Gradient descent over the full training set, learning rate 0.5, 300
epochs. Each epoch runs a forward pass, computes the loss, backpropagates, then applies
`W := W - lr * dW` manually. Train and test accuracy are printed every 25 epochs.

## Result

| Metric | Value |
|---|---|
| Final train accuracy | 0.9934 |
| Final test accuracy | 0.9870 |
| Final train loss | 0.0241 |

Both accuracies clear the 0.97 requirement.

## Constraints followed

- PyTorch only, no NumPy, no TensorFlow.
- No `torch.nn.Linear`, weights are raw tensors and the forward pass is written by hand.
- No autograd. `torch.set_grad_enabled(False)` is set so `loss.backward()`,
  `optimizer.zero_grad()` and `optimizer.step()` are never used.
- No PyTorch `DataLoader`, the CSVs are read with the built in `csv` module.
- ReLU, Sigmoid, binary cross entropy, accuracy and backpropagation defined from scratch.

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch matplotlib jupyter
jupyter notebook HW3.ipynb
```

Run the cells in order from the top.
