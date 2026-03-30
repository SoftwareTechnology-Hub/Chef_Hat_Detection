import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

MODEL_CONF = 0.3
HAT_THRESHOLD = 0.7
NO_HAT_THRESHOLD = 0.4

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=MODEL_CONF)

    hat_detected = False
    best_hat_box = None
    best_hat_conf = 0

    best_nohat_box = None
    best_nohat_conf = 0

    # 🔍 Check detections
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # 🟢 HAT detection
            if label == "Chef-Hat" and conf >= HAT_THRESHOLD:
                hat_detected = True
                if conf > best_hat_conf:
                    best_hat_conf = conf
                    best_hat_box = box

            # 🔴 NO HAT detection
            elif label == "No_Hat" and conf >= NO_HAT_THRESHOLD:
                if conf > best_nohat_conf:
                    best_nohat_conf = conf
                    best_nohat_box = box

    # 🎯 FINAL DECISION
    if hat_detected and best_hat_box is not None:
        # 🟢 Draw HAT box
        x1, y1, x2, y2 = map(int, best_hat_box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"HAT {best_hat_conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

    elif best_nohat_box is not None:
        # 🔴 Draw NO HAT box
        x1, y1, x2, y2 = map(int, best_nohat_box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"NO HAT {best_nohat_conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)

    else:
        # 🔴 If nothing detected → show warning
        cv2.putText(frame, "NO HAT",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    4)

    cv2.imshow("Chef Hat Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()