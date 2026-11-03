import json
import platform
from pathlib import Path

import onnxruntime as ort
import torch
import ultralytics
from ultralytics import YOLO

from config import MAX_DET, MODELS, NMS, SIZES

DATA = "dataset/VisDrone.yaml"
RUN_DIR = Path("runs/detect/VisDrone")  # {model}-{imgsz}/weights/best.*
SAVE_DIR = Path("benchmark_results")
SAVE_PATH = SAVE_DIR / "pipeline_latency_results.json"

SPLIT = "test"  # Use "test" to get results for test set.
RUN_ACCURACY = True
RUN_SPEED = False  # set to False for the test-split run

CONF = 0.25
IOU = 0.45

EXTS = ["pt", "onnx", "engine"]
LABELS = {"pt": "PyTorch_FP32", "onnx": "ONNX_FP32", "engine": "TensorRT_FP16"}

# ensures that ONNX model is running on GPU instead of silently moving to CPU
assert "CUDAExecutionProvider" in ort.get_available_providers(), (
    "install onnxruntime-gpu (ONNX is defaulting to CPU)"
)

# environment information
env = {
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "ultralytics": ultralytics.__version__,
    "onnxruntime": ort.__version__,
    "cuda": torch.version.cuda,
    "python": platform.python_version(),
    "os": platform.platform(),
}


def extract_metrics(results, names):
    """Helper to extract overall and per-class metrics"""
    return {
        "mAP50": float(results.box.map50),
        "mAP50:95": float(results.box.map),
        "per_class_mAP50:95": {
            name: float(ap) for name, ap in zip(names.values(), results.box.ap)
        },
    }


def model_path(model_name, imgsz, ext):
    """Path to best.{ext}; Also checks if it doesnt exist"""
    path = RUN_DIR / f"{model_name}-{imgsz}" / "weights" / f"best.{ext}"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path} (run train.py and export.py first)"
        )
    return path


if RUN_ACCURACY:
    SAVE_DIR.mkdir(exist_ok=True)

    for model_name in MODELS:
        for imgsz in SIZES:
            acc_args = {
                "data": DATA,
                "imgsz": imgsz,
                "batch": 1,
                "seed": 42,
                "max_det": MAX_DET,
                "device": "cuda:0",
                "split": SPLIT,
                "quantize": 32,
                "nms": NMS,
            }

            results_log = {
                "model": model_name,
                "imgsz": imgsz,
                "max_det": MAX_DET,
                "nms": NMS,
                "split": SPLIT,
                "metrics": {},
            }

            print(f"\n{'=' * 70}")
            print(
                f"ACCURACY: {model_name} (imgsz={imgsz}, max_det={MAX_DET}, split={SPLIT})"
            )
            print(f"\n{'=' * 70}")

            for ext in EXTS:
                path = model_path(model_name, imgsz, ext)
                print(f"\n{LABELS[ext]}: {path}")

                model = YOLO(str(path), task="detect")
                results = model.val(**acc_args)
                results_log["metrics"][LABELS[ext]] = extract_metrics(
                    results, model.names
                )
                print(f"  mAP50 {results.box.map50:.4f} | mAP50:95 {results.box.map:.4f}")

                del model
                torch.cuda.empty_cache()

            output_file = SAVE_DIR / f"{model_name}_{imgsz}_{SPLIT}_results.json"
            with open(output_file, "w") as f:
                json.dump(results_log, f, indent=4)
            print(f"\nResults saved to {output_file}")


if RUN_SPEED:
    results = []

    for model_name in MODELS:
        for imgsz in SIZES:
            for ext in EXTS:
                path = model_path(model_name, imgsz, ext)
                print(f"SPEED: {model_name} (imgsz={imgsz}, format={ext})")

                model = YOLO(str(path), task="detect")
                speed_args = {
                    "data": DATA,
                    "imgsz": imgsz,
                    "batch": 1,
                    "split": "val",  # speed does not depend on the split, so this stays "val"
                    "device": "cuda:0",
                    "quantize": 32,
                    "max_det": MAX_DET,
                    "nms": NMS,
                    "conf": CONF,
                    "iou": IOU,
                    "rect": False,
                    "plots": False,
                    "verbose": False,
                }

                model.val(**speed_args)  # warm-up pass, discarded
                s = model.val(**speed_args).speed  # timed pass

                row = {
                    "model": model_name,
                    "imgsz": imgsz,
                    "format": ext,
                    "preprocess_ms": round(s["preprocess"], 2),
                    "inference_ms": round(s["inference"], 2),
                    "postprocess_ms": round(s["postprocess"], 2),
                    "total_ms": round(
                        s["preprocess"] + s["inference"] + s["postprocess"], 2
                    ),
                }
                results.append(row)
                print(
                    f"  pre {row['preprocess_ms']} | inf {row['inference_ms']} | "
                    f"post {row['postprocess_ms']} | total {row['total_ms']} ms"
                )

                del model
                torch.cuda.empty_cache()

    with open(SAVE_PATH, "w") as f:
        json.dump(
            {
                "env": env,
                "settings": {
                    "conf": CONF,
                    "iou": IOU,
                    "max_det": MAX_DET,
                    "batch": 1,
                    "split": "val"
                },
                "results": results,
            },
            f,
            indent=4,
        )

    print(f"\nSaved to {SAVE_PATH}\n")
    # :<9 is left align in a field 9 characters wide. only affects terminal output xD I didnt know this earlier.
    # :>9 is right align etc.
    print(
        f"{'model':<9}{'imgsz':<7}{'fmt':<8}{'pre':<8}{'inf':<8}{'post':<8}{'total':<8}"
    )
    print("-" * 60)
    for r in results:
        print(
            f"{r['model']:<9}{r['imgsz']:<7}{r['format']:<8}{r['preprocess_ms']:<8}"
            f"{r['inference_ms']:<8}{r['postprocess_ms']:<8}{r['total_ms']:<8}"
        )
