import base64
import json
import threading
from pathlib import Path

import cv2
import numpy as np
from django.http import HttpResponseBadRequest, JsonResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO

MODEL_CONF = 0.3

# best.pt is stored at repo root (one level above the Django project folder)
MODEL_PATH = Path(__file__).resolve().parents[2] / "best.pt"

_MODEL = None
_MODEL_LOCK = threading.Lock()


def _get_model() -> YOLO:
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = YOLO(str(MODEL_PATH))
    return _MODEL


@login_required
def index(request):
    return render(request, "detection/index.html")


@login_required
@csrf_exempt
def detect(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST only")

    try:
        if request.content_type and "application/json" in request.content_type:
            payload = json.loads(request.body.decode("utf-8"))
            image_b64 = payload.get("image")
        else:
            image_b64 = request.POST.get("image")
    except Exception:
        return HttpResponseBadRequest("Invalid JSON")

    if not image_b64:
        return HttpResponseBadRequest("Missing image")

    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return HttpResponseBadRequest("Invalid image data")

    if frame is None:
        return HttpResponseBadRequest("Unable to decode image")

    model = _get_model()
    results = model.predict(
        source=frame,
        conf=MODEL_CONF,
        device="cpu",
        verbose=False,
    )

    detections = []
    height, width = frame.shape[:2]

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names.get(cls, str(cls))
            label_upper = label.upper()
            if "NO_HAT" in label_upper or "NO HAT" in label_upper:
                status = "WARNING"
            elif "HAT" in label_upper:
                status = "SAFE"
            else:
                status = "WARNING"
            color = "#00C853" if status == "SAFE" else "#D50000"
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(
                {
                    "label": label,
                    "confidence": round(conf, 4),
                    "status": status,
                    "color": color,
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )

    return JsonResponse(
        {"detections": detections, "image_size": {"width": width, "height": height}}
    )
