import glob, os, random
import numpy as np
from PIL import Image
import qubee

ONNX_PATH = "/NPU_Projects/sr_model.onnx"
TRAIN_LR_DIR = "/NPU_Projects/train/LR2"
CALIB_DIR = "/NPU_Projects/calib_npy"
LIST_PATH = "/NPU_Projects/calib_list.txt"
MXQ_PATH = "/NPU_Projects/sr_model.mxq"

os.makedirs(CALIB_DIR, exist_ok=True)
random.seed(0)
paths = sorted(glob.glob(os.path.join(TRAIN_LR_DIR, "*.*")))
paths = random.sample(paths, min(100, len(paths)))
with open(LIST_PATH, "w") as f:
    for p in paths:
        im = Image.open(p).convert("RGB").resize((256, 256), Image.BICUBIC)
        out = os.path.join(CALIB_DIR, os.path.splitext(os.path.basename(p))[0] + ".npy")
        np.save(out, np.asarray(im, np.float32) / 255)
        f.write(out + "\n")
print(len(paths), "calibration images from training set")

qubee.mxq_compile(model=ONNX_PATH, calib_data_path=LIST_PATH, backend="onnx", device="cpu", save_path=MXQ_PATH, model_nickname="sr_model")
print("saved", MXQ_PATH)
