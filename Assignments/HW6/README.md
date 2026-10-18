# HW6 - VGG11 on a 20 Class ImageNet Subset

A VGG11 network built in PyTorch and trained with SGD to classify images from 20 ImageNet
classes. The network is initialized from ImageNet pretrained VGG11 weights, with the last
fully connected layer replaced by a new 20 output layer.

## Dataset

| Split | Folder | Label file | Images |
|---|---|---|---|
| Train | `imagenet_train20a/` | `imagenet_train20.txt` | 6,000 |
| Validation | `imagenet_val20/` | `imagenet_val20.txt` | 1,000 |

Each line of a label file holds an image file name and its class index (0 to 19). Training
images sit in one subfolder per class, named by the file name prefix. The assignment PDF
calls the training label file `imagenet_train20a1.txt`. It is the same file.

The image zips are not committed here. `imagenet_train20.zip` is about 714 MB, which is
over the GitHub 100 MB file limit. Download `imagenet_train20.zip` and `imagenet_val20.zip`
from Canvas and unzip both next to the notebook before running it.

## Contents

- `assignment6.ipynb` - the full solution with outputs saved.
- `imagenet_train20.txt`, `imagenet_val20.txt` - image names and labels.

## Solution overview

**1. Build VGG11.**

| Block | Layers | Output size |
|---|---|---|
| Input | | 224 x 224 x 3 |
| 1 | Conv 64, MaxPool | 112 x 112 x 64 |
| 2 | Conv 128, MaxPool | 56 x 56 x 128 |
| 3 | Conv 256, Conv 256, MaxPool | 28 x 28 x 256 |
| 4 | Conv 512, Conv 512, MaxPool | 14 x 14 x 512 |
| 5 | Conv 512, Conv 512, MaxPool | 7 x 7 x 512 |
| Classifier | Flatten, FC 4096, FC 4096, FC 20 | 20 |

Every convolution is 3x3, stride 1, same padding, followed by ReLU. Every max pool is 2x2
with stride 2. Dropout (p=0.5) follows each of the first two FC layers. That is not in the
assignment table, but it is standard in VGG and matches the pretrained model. The table in
the PDF lists the 4th max pool output as 28x28x256. It should be 14x14x512, which is what
the code produces.

Softmax is applied inside `nn.CrossEntropyLoss`, and the predicted class is the argmax of
the 20 outputs.

**2. Weight initialization.** Weights come from torchvision's ImageNet pretrained VGG11.
Every layer except the last one (`classifier.6`, 1000 outputs) is loaded. The notebook
prints the result of `load_state_dict`, which shows that only `classifier.6.weight` and
`classifier.6.bias` are missing.

**3. Training.** SGD with learning rate 0.001 and momentum 0.9, batch size 32, 10 epochs.
Training images use `RandomResizedCrop(224)` and a random horizontal flip. Validation
images are resized to 256 and center cropped to 224. Both are normalized with the ImageNet
mean and standard deviation. Train and validation accuracy are printed every epoch.

## Result

| Metric | Value |
|---|---|
| Final train accuracy | 0.9400 |
| Final validation accuracy | 0.9690 |
| Best validation accuracy | 0.9760 (epochs 5 and 6) |

Both accuracies clear the 50% requirement. Validation accuracy is already 0.965 after the
first epoch, thanks to the pretrained initialization.

Validation accuracy is higher than training accuracy because they are measured under
different conditions. Training accuracy is computed during training, with random crops,
flips and dropout all active. Validation accuracy is computed in eval mode, with dropout off
and a clean center crop. The model is not overfitting.

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision pillow jupyter
jupyter notebook assignment6.ipynb
```

Run the cells in order from the top. The notebook uses a CUDA GPU or Apple `mps` if one is
available and falls back to the CPU otherwise. The pretrained weights download
automatically on the first run.
