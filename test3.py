import cv2
from ultralytics import YOLO

# 🔥 Load trained model
model = YOLO("best.pt")  # keep best.pt in same folder

# 🎥 Open webcam
cap = cv2.VideoCapture(0)

# ⚙️ Confidence threshold (same as Colab)
CONF_THRESHOLD = 0.7

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔍 Run detection
    results = model(frame, conf=CONF_THRESHOLD)

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 🎨 Color logic
            if label == "Chef-Hat":
                color = (0, 255, 0)  # 🟢 Green
                text = f"HAT {conf:.2f}"
            else:
                color = (0, 0, 255)  # 🔴 Red
                text = f"NO HAT {conf:.2f}"

            # 📦 Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 🏷️ Label
            cv2.putText(frame, text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

    # 🖥️ Show output
    cv2.imshow("Chef Hat Detection", frame)

    # ❌ Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🔚 Cleanup
cap.release()
cv2.destroyAllWindows()