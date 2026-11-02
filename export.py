from pathlib import Path

from ultralytics import YOLO

from config import MAX_DET, MODELS, NMS, SIZES


def export_model(model_name: str, img_sz: int, max_det: int, nms: bool = False):
    pt_path = f"runs/detect/VisDrone/{model_name}-{img_sz}/weights/best.pt"

    if not Path(pt_path).exists():
        raise FileNotFoundError(
            f"Missing trained model weights: {pt_path} (run train.py first)"
        )

    export_args = {
        "imgsz": img_sz,
        "batch": 1,
        "max_det": max_det,
        "device": 0,
        "nms": nms,
    }

    print(f"\n{'=' * 70}")
    print(f"EXPORTING: {model_name} (imgsz={img_sz}, max_det={max_det})")
    print(f"Loading: {pt_path}")
    print(f"\n{'=' * 70}")

    # Load PyTorch model (best.pt)
    model = YOLO(pt_path)

    # ONNX Export (FP32)
    onnx_path = model.export(format="onnx", simplify=True, **export_args)
    print(f"ONNX saved to {onnx_path}")

    # TensorRT Export (FP16)
    engine_path = model.export(format="engine", quantize=16, **export_args)
    print(f"TensorRT engine saved to {engine_path}")


if __name__ == "__main__":
    for model_name in MODELS:
        for img_sz in SIZES:
            export_model(model_name, img_sz, MAX_DET, NMS)

    print("\nAll exports complete.\n Use `uv run benchmark.py` to get some benchmarks.")
