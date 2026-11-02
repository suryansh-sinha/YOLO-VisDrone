# Shared settings: export.py and benchmark.py both import these,
# so the two scripts can never disagree about models, sizes or max_det.
MODELS = ["yolov8n", "yolo26n"]
SIZES = [640, 960]
MAX_DET = 461
NMS = False  # only has an effect for YOLO26n (selects its NMS-free head)
