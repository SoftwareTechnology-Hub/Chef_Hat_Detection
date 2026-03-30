import base64
import json
import threading
from pathlib import Path
from urllib import parse, request as urlrequest

import cv2
import numpy as np
from django.conf import settings
from django.http import HttpResponseBadRequest, JsonResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from ultralytics import YOLO

MODEL_CONF = 0.3
HAT_THRESHOLD = 0.7
NO_HAT_THRESHOLD = 0.4

# best.pt is stored at repo root (one level above the Django project folder)
MODEL_PATH = Path(__file__).resolve().parents[2] / "best.pt"

_MODEL = None
_MODEL_LOCK = threading.Lock()
_ALERT_LOCK = threading.Lock()

ALERT_FILES_MAX = 5
ALERT_PREFIX = "latest_alert"


def _get_model() -> YOLO:
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = YOLO(str(MODEL_PATH))
    return _MODEL


def _next_alert_index() -> int:
    index_path = Path(settings.MEDIA_ROOT) / "alert_index.txt"
    if not index_path.exists():
        index_path.write_text("0", encoding="utf-8")
        return 1
    try:
        current = int(index_path.read_text(encoding="utf-8").strip() or "0")
    except Exception:
        current = 0
    next_index = (current % ALERT_FILES_MAX) + 1
    index_path.write_text(str(next_index), encoding="utf-8")
    return next_index


def _save_alert_image(frame: np.ndarray, box, conf: float) -> str:
    Path(settings.MEDIA_ROOT).mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        frame,
        f"NO HAT {conf:.2f}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    with _ALERT_LOCK:
        idx = _next_alert_index()
        filename = f"{ALERT_PREFIX}{idx}.jpg"
        filepath = Path(settings.MEDIA_ROOT) / filename
        cv2.imwrite(str(filepath), frame)
    return filename


def _send_telegram_message(text: str) -> None:
    token = getattr(settings, "TELEGRAM_BOT_TOKEN", "")
    chat_id = getattr(settings, "TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urlrequest.Request(url, data=data, method="POST")
    try:
        urlrequest.urlopen(req, timeout=5).read()
    except Exception:
        # Ignore notification failures to avoid breaking inference
        return


def _notify_async(text: str) -> None:
    threading.Thread(target=_send_telegram_message, args=(text,), daemon=True).start()


@login_required
def index(request):
    return render(request, "detection/index.html")


@login_required
def alerts(request):
    media_root = Path(settings.MEDIA_ROOT)
    files = []
    if media_root.exists():
        for i in range(1, ALERT_FILES_MAX + 1):
            name = f"{ALERT_PREFIX}{i}.jpg"
            path = media_root / name
            if path.exists():
                modified_dt = timezone.make_aware(
                    timezone.datetime.fromtimestamp(path.stat().st_mtime)
                )
                modified = timezone.localtime(modified_dt).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                files.append(
                    {
                        "name": name,
                        "url": f"{settings.MEDIA_URL}{name}",
                        "time": modified,
                    }
                )
    return render(request, "detection/alerts.html", {"files": files})


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
    has_safe = False
    best_warning_box = None
    best_warning_conf = 0.0

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names.get(cls, str(cls))
            label_upper = label.upper()
            if "NO_HAT" in label_upper or "NO HAT" in label_upper:
                if conf < NO_HAT_THRESHOLD:
                    continue
                status = "WARNING"
            elif "HAT" in label_upper:
                if conf < HAT_THRESHOLD:
                    continue
                status = "SAFE"
            else:
                # Unknown class: drop low-confidence noise
                if conf < MODEL_CONF:
                    continue
                status = "WARNING"
            if status == "SAFE":
                has_safe = True
            if status == "WARNING" and conf > best_warning_conf:
                best_warning_conf = conf
                best_warning_box = box
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

    saved_file = None
    if best_warning_box is not None and not has_safe:
        saved_file = _save_alert_image(frame.copy(), best_warning_box, best_warning_conf)
        image_url = request.build_absolute_uri(
            f"{settings.MEDIA_URL}{saved_file}"
        )
        timestamp = timezone.localtime().strftime("%Y-%m-%d %H:%M:%S")
        message = (
            "NO HAT detected\n"
            f"Time: {timestamp}\n"
            f"Confidence: {best_warning_conf:.2f}\n"
            f"Image: {image_url}"
        )
        _notify_async(message)

    return JsonResponse(
        {
            "detections": detections,
            "image_size": {"width": width, "height": height},
            "saved_file": saved_file,
        }
    )
