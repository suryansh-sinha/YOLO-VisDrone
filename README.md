# YOLOv8n vs YOLO26n on Drone Images (VisDrone)

I trained two small object detectors, YOLOv8n and YOLO26n, on VisDrone (photos taken from drones) and compared them on three things: accuracy, the effect of image size, and speed after converting them to faster formats.

Training curves: [Weights & Biases Report](https://forge.coreweave.com/wandb/suryanshsinha01-na/VisDrone/reports/Training-and-Validation-Curves--VmlldzoxODI0NDU3Mw?accessToken=yppf9k57ixletl9e6j1k7i7h5y11z3az5zui7ghpcfdxsy19i6lpf676gqiirznn)

## At a glance

| Model | Image size | mAP50-95 on test set (TensorRT) | Time per image (TensorRT) |
|---|---|---|---|
| YOLOv8n | 640 | 0.148 | 0.88 ms |
| YOLO26n | 640 | 0.144 | 0.73 ms |
| YOLOv8n | 960 | 0.200 | 1.29 ms |
| YOLO26n | 960 | 0.191 | 1.15 ms |

Accuracy is on the VisDrone test-dev split. Time is for one image at a time on an RTX 5070 Ti. Details and the full tables are below.

## Main findings

1. **Bigger input images help a lot.** Going from 640 to 960 pixels raised accuracy by about 5 points on the test set (about 7 points on the validation set) for both models. Drone images are full of tiny objects, so more pixels help.
2. **YOLOv8n is slightly more accurate.** On the test set it leads by 0.5 points at 640 (inside my noise margin, so effectively a tie) and by 0.9 points at 960. On the validation set the lead is roughly 1 point at both sizes.
3. **Converting to faster formats barely changes accuracy.** After converting to ONNX and TensorRT, scores stayed within 0.25 points of the original on the test set (0.31 on the validation set).
4. **TensorRT is 2 to 3.3 times faster than PyTorch** for the whole detection process, one image at a time.
5. **YOLO26n is faster than YOLOv8n only with TensorRT** (11 to 17% faster). It skips the step that removes duplicate boxes (called NMS), which saves about 0.26 ms per image, but its main network step takes about 0.1 ms longer. With ONNX Runtime the two models tie, and with PyTorch YOLOv8n is faster.
6. **The trade-off:** with TensorRT on the test set, YOLO26n gives up about 0.4 points of accuracy at 640 pixels (within noise) for 17% more speed, and about 0.9 points at 960 pixels for 11% more speed. The 640 setting is the better deal.

Each setup was trained once. When I retrained YOLO26n, its score moved by about 0.3 points, so treat gaps smaller than roughly 0.5 points as a tie.

## Results

**mAP** is the standard accuracy score for object detection (0 to 1, higher is better). mAP50 is the easier version. mAP50-95 is stricter, and it is the one I focus on.

I picked each model's best checkpoint using the validation images, so validation scores are slightly optimistic. The **test-set numbers are the unbiased ones** and are the main results. The test set is the VisDrone test-dev split. I kept the validation tables for reference.

### Accuracy (test set)

| Model | Image size | mAP50 | mAP50-95 | mAP50-95 after ONNX | mAP50-95 after TensorRT |
|---|---|---|---|---|---|
| YOLOv8n | 640 | 0.270 | 0.149 | 0.149 | 0.148 |
| YOLO26n | 640 | 0.269 | 0.145 | 0.144 | 0.144 |
| YOLOv8n | 960 | 0.354 | 0.202 | 0.201 | 0.200 |
| YOLO26n | 960 | 0.343 | 0.193 | 0.191 | 0.191 |

Test scores are lower than validation scores. Part of that is because I chose checkpoints on the validation images. I did not investigate the rest.

### Accuracy by object type (test set, mAP50-95, PyTorch)

| Class | YOLOv8n 640 | YOLO26n 640 | YOLOv8n 960 | YOLO26n 960 |
|---|---|---|---|---|
| pedestrian | 0.084 | 0.087 | 0.130 | 0.136 |
| people | 0.037 | 0.045 | 0.061 | 0.071 |
| bicycle | 0.020 | 0.027 | 0.051 | 0.047 |
| car | 0.406 | 0.403 | 0.479 | 0.473 |
| van | 0.191 | 0.181 | 0.261 | 0.236 |
| truck | 0.198 | 0.177 | 0.263 | 0.240 |
| tricycle | 0.072 | 0.067 | 0.121 | 0.111 |
| awning-tricycle | 0.065 | 0.071 | 0.109 | 0.107 |
| bus | 0.336 | 0.295 | 0.401 | 0.373 |
| motor | 0.085 | 0.093 | 0.143 | 0.138 |

YOLO26n matches or slightly beats YOLOv8n on the smallest, most crowded objects (pedestrian, people). YOLOv8n is clearly better on larger vehicles, especially vans, trucks and buses. Bicycles are the hardest class for both models, even at 960 pixels.

<details>
<summary><b>Validation-set accuracy tables</b> (548 images, used to pick checkpoints)</summary>

| Model | Image size | mAP50 | mAP50-95 | mAP50-95 after ONNX | mAP50-95 after TensorRT |
|---|---|---|---|---|---|
| YOLOv8n | 640 | 0.331 | 0.186 | 0.186 | 0.185 |
| YOLO26n | 640 | 0.322 | 0.177 | 0.177 | 0.176 |
| YOLOv8n | 960 | 0.432 | 0.256 | 0.254 | 0.253 |
| YOLO26n | 960 | 0.418 | 0.245 | 0.244 | 0.244 |

| Class | YOLOv8n 640 | YOLO26n 640 | YOLOv8n 960 | YOLO26n 960 |
|---|---|---|---|---|
| pedestrian | 0.143 | 0.151 | 0.229 | 0.230 |
| people | 0.094 | 0.100 | 0.148 | 0.148 |
| bicycle | 0.032 | 0.033 | 0.083 | 0.077 |
| car | 0.495 | 0.488 | 0.576 | 0.562 |
| van | 0.259 | 0.233 | 0.343 | 0.308 |
| truck | 0.198 | 0.172 | 0.260 | 0.243 |
| tricycle | 0.118 | 0.103 | 0.178 | 0.169 |
| awning-tricycle | 0.069 | 0.065 | 0.101 | 0.085 |
| bus | 0.301 | 0.272 | 0.404 | 0.404 |
| motor | 0.150 | 0.156 | 0.234 | 0.226 |

</details>

### Speed

Speed does not depend on which split I use, so I measured it on the 548 validation images.

Time per image in milliseconds, one image at a time (lower is better). I ran the whole measurement three times and give the middle value. The script only saves its latest run, so I picked the middle value by hand. The three runs never differed by more than 0.04 ms.

| Model | Image size | PyTorch | ONNX Runtime | TensorRT | TensorRT vs PyTorch |
|---|---|---|---|---|---|
| YOLOv8n | 640 | 2.00 | 2.00 | 0.88 | 2.3x faster |
| YOLOv8n | 960 | 2.71 | 2.86 | 1.29 | 2.1x faster |
| YOLO26n | 640 | 2.39 | 1.99 | 0.73 | 3.3x faster |
| YOLO26n | 960 | 2.89 | 2.83 | 1.15 | 2.5x faster |

Where YOLO26n's time goes in TensorRT: the cleanup step takes about 0.07 ms, compared with 0.33 ms for YOLOv8n. But its main network step takes about 0.45 ms, compared with 0.34 ms (at 640 pixels). YOLO26n does less arithmetic per image than YOLOv8n (see the table below), yet its network step is not faster when running one image at a time.

| | Parameters | Arithmetic per image (GFLOPs) |
|---|---|---|
| YOLOv8n | 3.13M | 8.6 |
| YOLO26n | 2.39M | 5.4 |

### A side experiment

YOLO26n also has an older-style mode that uses NMS. On an earlier training run of YOLO26n, that mode scored 0.5 to 1.0 points higher (mAP50-95, validation set) than the no-NMS mode, which put it roughly level with YOLOv8n at 960 pixels. So most of the accuracy gap above looks like the price of removing NMS.

## What I checked along the way

- **The data:** image and label files matched in every split, I drew boxes on sample images to confirm they sat on the objects, and the class counts looked right.
- **Detection cap:** Ultralytics warned that some images have more objects than its default limit of 300 detections (the busiest training image has 902). I set the limit myself: 902 for training, 350 for validation and speed (the busiest validation image has 317, and I rounded up), and 461 for the test set, based on its busiest image.
- **Fair training:** both models used the same seed, epochs and settings. At 960 pixels, batch size 32 ran out of GPU memory, so I used 16 and let Ultralytics combine updates, so every run still learns from 64 images at a time.
- **Choosing the best checkpoint:** by default, YOLO26n picks its best checkpoint using the older NMS mode, not the one it actually ships with. I noticed this and retrained both YOLO26n runs with the correct setting.
- **Converted models:** I re-measured accuracy after every conversion instead of assuming nothing was lost.
- **Speed:** I repeated the timing three times and confirmed the results were stable. Ultralytics' timers wait for the GPU to finish each step before reading the clock, so the timings reflect real work.

## How I measured

- **Accuracy:** one image at a time, same settings for PyTorch, ONNX and TensorRT, on both the validation and test images. I used the Ultralytics default confidence threshold for accuracy (0.001), not 0.25. YOLO26n used its no-NMS mode. PyTorch and ONNX ran at full precision (FP32), and TensorRT ran at half precision (FP16).
- **Detection cap:** the validation accuracy and speed numbers come from exports made with a cap of 350. For the test run I re-exported all models with a cap of 461.
- **Speed:** Ultralytics' built-in timers, averaged over all 548 validation images, after one warm-up pass that I discarded. A "per image" time covers three steps: preparing the image for the GPU, running the network, and cleaning up the results. It does not include reading the image from disk or resizing it. I used a confidence threshold of 0.25 and an overlap threshold of 0.45.

## Things to keep in mind

- **One training run per setup.** Small differences between the models may be noise.
- **Test set used only for final accuracy.** I chose checkpoints on the validation images, never on the test images. Test scores are the unbiased ones, and validation scores are slightly optimistic.
- **Speed and test accuracy come from different exports.** Speed used exports with a detection cap of 350, and the test accuracy used re-exports with a cap of 461. I did not re-time the 461 exports. The cap should change little, but I did not verify that.
- **Speed limits:** I report average times only (no worst-case numbers), for one image at a time, on one GPU that also runs my display.
- **TensorRT mixes two changes.** It uses half precision and an NVIDIA-specific engine, and I did not separate how much each one contributes to the speedup.
- **Busy scenes:** I used a confidence threshold of 0.25. A much lower threshold on a crowded image would make YOLOv8n's NMS step slower than reported here.
- **Portability:** TensorRT files only work on the same GPU, driver and TensorRT version they were built with.

## Setup

| | |
|---|---|
| GPU | NVIDIA RTX 5070 Ti (16 GB) |
| OS | CachyOS (Arch Linux) |
| Python | 3.12.14 |
| PyTorch | 2.15.0 (CUDA 13.2) |
| Ultralytics | 8.4.193 |
| ONNX Runtime | 1.24.4 (GPU) |
| TensorRT | 11.4.0.106 |

**Training settings:** pretrained Ultralytics weights, 100 epochs, seed 42, image sizes 640 and 960, batch size 32 at 640 and 16 at 960, everything else at Ultralytics defaults. Both models had flat accuracy curves over the last 15 to 20 epochs, so 100 epochs was enough.

## Run it yourself

```bash
uv sync
wandb login                  # optional, for experiment tracking
yolo settings wandb=True

uv run train.py              # train both models
uv run export.py             # convert the trained models to ONNX and TensorRT
uv run benchmark.py          # measure accuracy and speed for every model / size / format
```

`benchmark.py` has switches at the top: `SPLIT` ("val" or "test"), `RUN_ACCURACY` and `RUN_SPEED`. `config.py` sets the detection cap (`MAX_DET`), which is currently 461 (the test setting). Set it to 350 and re-run `export.py` to reproduce the validation and speed numbers.

The `VisDrone.yaml` file in `dataset/` tells Ultralytics how to download VisDrone and convert it to the right format. The first time you train, it downloads the data into a `datasets` folder next to the project folder (not inside the repo).

## Dataset and license

VisDrone has 10 object types: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus and motor. The validation set has 548 images and 38,759 objects. Cars (14,064) and pedestrians (8,844) dominate, while buses (251) and awning-tricycles (532) are rare. The test set contains 1610 images with 75,102 instances in total.

VisDrone is for academic and research use only, and models trained on it carry that restriction. This is a learning and portfolio project, not a commercial model. Ultralytics is released under the AGPL-3.0 license (or a paid commercial license).

Thanks to the VisDrone team (AISKYEYE, Tianjin University) and to Ultralytics.
