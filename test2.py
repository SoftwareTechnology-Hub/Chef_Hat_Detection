import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

# 🔥 Use LOW threshold here
MODEL_CONF = 0.3

# 🎯 Custom thresholds
HAT_THRESHOLD = 0.75
NO_HAT_THRESHOLD = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=MODEL_CONF)

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 🧠 Decision logic
            if label == "Chef-Hat" and conf >= HAT_THRESHOLD:
                color = (0, 255, 0)
                text = f"HAT {conf:.2f}"

            elif label == "No_Hat" and conf >= NO_HAT_THRESHOLD:
                color = (0, 0, 255)
                text = f"NO HAT {conf:.2f}"

            else:
                continue  # ignore weak detections

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

    cv2.imshow("Chef Hat Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()