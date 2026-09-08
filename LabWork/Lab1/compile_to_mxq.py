"""
ONNX -> MXQ compilation script using Mobilint Qubee (real API).
Calibration data reshaped to [240, 240, 3] (NHWC, no batch dim) to match
what Qubee's compiler expects, based on the successful random-calib run.
"""

import os
import glob
import numpy as np
from PIL import Image
import qubee

ONNX_PATH = "/NPU_Projects/custom_cnn.onnx"
CALIB_JPEG_DIR = "/NPU_Projects/calibration_data"
CALIB_NPY_DIR = "/NPU_Projects/calibration_npy"
OUTPUT_MXQ_PATH = "/NPU_Projects/custom_cnn.mxq"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_calibration_images():
    os.makedirs(CALIB_NPY_DIR, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(CALIB_JPEG_DIR, "*.JPEG")))
    print(f"Found {len(image_paths)} calibration JPEGs")

    for path in image_paths:
        img = Image.open(path).convert("RGB").resize((240, 240))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        # Keep HWC order, no batch dimension -> shape (240, 240, 3)
        base = os.path.splitext(os.path.basename(path))[0]
        np.save(os.path.join(CALIB_NPY_DIR, base + ".npy"), arr.astype(np.float32))

    print(f"Saved {len(image_paths)} preprocessed .npy calibration files to {CALIB_NPY_DIR}")


def write_calib_list_file():
    npy_files = sorted(glob.glob(os.path.join(CALIB_NPY_DIR, "*.npy")))
    list_path = "/NPU_Projects/calib_list.txt"
    with open(list_path, "w") as f:
        for path in npy_files:
            f.write(path + "\n")
    print(f"Wrote calibration list file with {len(npy_files)} entries: {list_path}")
    return list_path


def main():
    preprocess_calibration_images()
    calib_list_path = write_calib_list_file()

    print("Starting mxq_compile with REAL calibration data...")
    qubee.mxq_compile(
        model=ONNX_PATH,
        calib_data_path=calib_list_path,
        backend="onnx",
        device="cpu",
        save_path=OUTPUT_MXQ_PATH,
        model_nickname="custom_cnn",
    )
    print(f"Compilation finished. Output should be at: {OUTPUT_MXQ_PATH}")


if __name__ == "__main__":
    main()