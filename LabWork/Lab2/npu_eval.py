import glob, os
import numpy as np
from PIL import Image
import maccel

MXQ_PATH = "/NPU_Projects/sr_model.mxq"
VALID_DIR = "/NPU_Projects/valid"

load = lambda p: np.asarray(Image.open(p).convert("RGB").resize((256, 256), Image.BICUBIC), np.float32) / 255
lr = sorted(glob.glob(os.path.join(VALID_DIR, "**/LR*/*.*"), recursive=True))
hr = sorted(glob.glob(os.path.join(VALID_DIR, "**/HR*/*.*"), recursive=True))
assert len(lr) == len(hr) > 0

acc = maccel.Accelerator()
model = maccel.Model(MXQ_PATH, maccel.ModelConfig())
model.launch(acc)
scores = []
for a, b in zip(lr, hr):
    y = np.squeeze(np.asarray(model.infer([load(a)])[0]))
    if y.shape[0] == 3:
        y = y.transpose(1, 2, 0)
    print("output shape", y.shape) if not scores else None
    scores.append(-10 * np.log10(np.mean((np.clip(y, 0, 1) - load(b)) ** 2)))
model.dispose()
print(f"mean PSNR on NPU over {len(scores)} images: {np.mean(scores):.3f}")
