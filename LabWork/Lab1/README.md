# Lab 1 — Custom CNN Image Classification (20-class ImageNet subset)

DATA 255, Fall 2026

## Task

Build, train, and deploy a from-scratch CNN (no pretrained weights) to classify a 20-category
subset of ImageNet (300 train / 50 val images per class), then compile the trained model to run
on NPU hardware.

- Input: 240x240x3 RGB images
- Output: 20 classes

## Files

| File | Description |
|---|---|
| `Lab1_Training_Notebook.ipynb` | Data loading, model architecture, training loop, evaluation (per-class accuracy, confusion matrix), ONNX export |
| `Lab1_Step3_Report.pdf` | Architecture justification, parameter count, hyperparameters, preprocessing, per-class/overall accuracy |
| `compile_to_mxq.py` | ONNX -> MXQ compilation script (Mobilint Qubee), including calibration data preprocessing |
| `custom_cnn.mxq` | Final quantized model compiled for NPU inference |

## Model

Custom 5-block CNN (Conv-BN-ReLU x2 per block, channels doubling 32->64->128->256->256) with
global average pooling and a dropout MLP classifier head. ~1.2M parameters.

## Results

| Stage | Accuracy |
|---|---|
| PyTorch validation (best checkpoint, epoch 44) | 73.30% |
| NPU (`.mxq`, quantized, MACCEL runtime) | 74.20% (742/1000) |
