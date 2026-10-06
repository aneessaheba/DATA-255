# Lab 2: Super Resolution on NPU

DATA 255, Fall 2026

## Task

Build and train a super resolution model from scratch in PyTorch that restores degraded low resolution
images (2217 train / 110 valid pairs), then compile it to run on NPU hardware. Score is mean PSNR on the NPU.

* Input: 256x256x3 LR image
* Output: 256x256x3 restored image

## Files

| File | Description |
|---|---|
| `model.py` | TriScaleSR architecture |
| `best_model.pth` | Trained weights (best validation checkpoint) |
| `Lab2_Training_Notebook.ipynb` | Data loading, augmentation, training, evaluation, ONNX export |
| `sr_model.onnx` | ONNX export of the trained model |
| `compile_to_mxq.py` | ONNX to MXQ compilation (Mobilint Qubee) with 100 training images as calibration data |
| `npu_eval.py` | Runs the MXQ model on the NPU over the validation set and reports mean PSNR |
| `sr_model.mxq` | Final quantized model compiled for NPU inference |

## Model

TriScaleSR: residual block encoder decoder at full, 1/2 and 1/4 resolution (64, 128, 256 channels) with
skip connections between scales and a global skip, so the network learns only the correction to the LR
image. Uses only Conv, ConvTranspose, ReLU and Add for NPU compatibility. About 9.1M parameters.

## Training

Random 192x192 crops with rotation and flip augmentation, batch 16, Adam with cosine decay from 0.0002 over
80000 steps, mixed precision. L1 loss for the first 80% of steps, then MSE. EMA of weights evaluated every
2000 steps, best checkpoint kept. Trained on an RTX 5090 in about 2.5 hours.

## Results

| Stage | Mean PSNR (dB) |
|---|---|
| LR input vs HR (no model) | 25.121 |
| PyTorch validation (best checkpoint) | 26.595 |
| NPU (`.mxq`, quantized, MACCEL runtime) | 26.556 |
