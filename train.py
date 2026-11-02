from ultralytics import YOLO

train_args = {
    "data": "dataset/VisDrone.yaml",
    "epochs": 100,
    "imgsz": 960,
    "batch": 16,    # 32 for imgsz 640
    "seed": 42,
    "max_det": 902,         # Explicitly set to match the densest training image
    "project": "VisDrone",  # W&B project name
    "nms": False            # nms false so it uses one-to-one head WHEN AVAILABLE
}

models_to_train = [
    {"name": f"yolov8n-{train_args['imgsz']}", "weights": "yolov8n.pt"},
    {"name": f"yolo26n-{train_args['imgsz']}", "weights": "yolo26n.pt"},
]

for model_config in models_to_train:
    print(f"\n{'='*60}")
    print(f"Starting Training: {model_config['name']} ({model_config['weights']})")
    print(f"\n{'='*60}")

    model = YOLO(model_config['weights'])

    results = model.train(
        name=model_config['name'],
        **train_args
    )

    print(f"Finished: {model_config['name']}")

print("All training runs complete! Check Weights & Biases dashboard for the comparison.")
