import cv2
from ultralytics import YOLO

# 🔥 Load model
model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

# ⚙️ Thresholds
MODEL_CONF = 0.3
HAT_THRESHOLD = 0.7

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=MODEL_CONF)

    hat_detected = False
    best_hat_box = None
    best_conf = 0

    # 🔍 Check all detections
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # ✅ Only care about Chef-Hat
            if label == "Chef-Hat" and conf >= HAT_THRESHOLD:
                hat_detected = True

                # store best hat
                if conf > best_conf:
                    best_conf = conf
                    best_hat_box = box

    # 🎯 DECISION
    if hat_detected and best_hat_box is not None:
        # 🟢 Show HAT
        x1, y1, x2, y2 = map(int, best_hat_box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"HAT {best_conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

    else:
        # 🔴 Show NO HAT (big warning)
        cv2.putText(frame, "NO HAT",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    4)

    # 🖥️ Show frame
    cv2.imshow("Chef Hat Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()