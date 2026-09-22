# HW5 - Deep Neural Network from Scratch (10 classes)

A 784 -> 100 -> 10 neural network built and trained entirely from scratch in PyTorch to
classify all ten MNIST handwritten digits. Forward pass, Softmax, cross entropy loss,
backpropagation and the gradient descent update are all written by hand.

## Dataset

| File | Samples |
|---|---|
| `mnist_train.csv` | 60,000 |
| `mnist_test.csv` | 10,000 |

Each row holds 785 values: the label (0 to 9) followed by 784 pixel values (0 to 255) of a
28x28 grayscale image.

The CSVs are not committed here. `mnist_train.csv` is about 105 MB, which is over the
GitHub 100 MB file limit. Download both from Canvas/Files or from
[Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and place them next to
the notebook before running it.

## Contents

- `assignment5.ipynb` - the full solution with outputs saved.

## Solution overview

**1. Build the network.**

| Layer | Neurons | Weight matrix | Activation |
|---|---|---|---|
| Input | 784 | | |
| Hidden | 100 | `W1`: 784 x 100 | Sigmoid |
| Output | 10 | `W2`: 100 x 10 | Softmax |

No `nn.Linear` is used, both layers are plain matrix multiplications with hand written
weights. Sigmoid and Softmax are defined from scratch. Both layers use Xavier
initialization and the random seed is fixed to 42. Softmax subtracts the row max before
taking `exp` so it cannot overflow, which cancels out and does not change the result.

Labels are one hot encoded into a `(m, 10)` target matrix because the output layer has ten
neurons.

Forward pass for a batch `X` of shape `(m, 784)`:

```
Z1 = X @ W1 + b1
A1 = sigmoid(Z1)
Z2 = A1 @ W2 + b2
A2 = softmax(Z2)
```

**2. Loss, accuracy and backward propagation.** Cross entropy and accuracy (argmax over
the ten outputs) are implemented from scratch. Softmax combined with cross entropy makes
the output gradient simply `A2 - Y`, and the rest is derived by hand:

```
dZ2 = (A2 - Y) / m
dW2 = A1.T @ dZ2                      db2 = sum(dZ2)
dZ1 = (dZ2 @ W2.T) * A1 * (1 - A1)
dW1 = X.T @ dZ1                       db1 = sum(dZ1)
```

**3. Training.** Mini batch gradient descent, learning rate 0.5, batch size 128, 30 epochs.
Batches come from `torch.randperm` and plain tensor slicing, so no `DataLoader` is
involved. Each batch runs a forward pass, backpropagates, then applies `W := W - lr * dW`
manually. Train and test accuracy are printed every epoch.

## Result

| Metric | Value |
|---|---|
| Final train accuracy | 0.9912 |
| Final test accuracy | 0.9764 |
| Final train loss | 0.0375 |

Both accuracies clear the 0.91 requirement. Test accuracy passes 0.91 after the first
epoch.

## Constraints followed

- PyTorch only, no NumPy, no TensorFlow.
- No `torch.nn.Linear`, weights are raw tensors and the forward pass is written by hand.
- No autograd. `torch.set_grad_enabled(False)` is set so `loss.backward()`,
  `optimizer.zero_grad()` and `optimizer.step()` are never used.
- No PyTorch `DataLoader`, the CSVs are read with the built in `csv` module.
- Sigmoid, Softmax, cross entropy, accuracy and backpropagation defined from scratch.

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyter
jupyter notebook assignment5.ipynb
```

Run the cells in order from the top.
