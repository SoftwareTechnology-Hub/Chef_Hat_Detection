# Chef Hat Detection

End-to-end Django app that uses a YOLOv8 model to detect chef hats from a live browser camera feed, logs detections, and saves "no-hat" alert snapshots. Includes a simple login-protected dashboard, live detection UI, and an alerts gallery.

**Repository**
```
https://github.com/SoftwareTechnology-Hub/Chef_Hat_Detection.git
```

**Demo Flow**
1. User logs in via Django auth.
2. Browser opens camera stream.
3. Frames are sent to `/api/detect/` for YOLO inference.
4. Results are drawn on the canvas overlay.
5. "No hat" events are saved and optionally sent to Telegram.

## Features
- Live camera detection using YOLOv8 (CPU).
- Login-protected dashboard and alerts gallery.
- Real-time status updates and detection logs on the UI.
- Saves up to 5 latest "no-hat" alert images (FIFO).
- Optional Telegram alert notifications.
- Clean, responsive UI for desktop and mobile.

## Tech Stack
- **Backend:** Django 5.x
- **Model:** Ultralytics YOLOv8 (`best.pt`)
- **Computer Vision:** OpenCV + NumPy
- **Frontend:** HTML/CSS/JS (camera + canvas overlay)
- **Auth:** Django built-in auth

## Project Structure
```
chefhat_project/
  chefhat_project/        # Django settings & URLs
    settings.py
    urls.py
  detection/              # Detection app
    views.py
    urls.py
    templates/detection/
      index.html          # Live detection UI
      alerts.html         # Alerts gallery
  users/                  # Users app (Django auth templates)
  templates/
    base.html             # Base layout + sidebar
  static/                 # Static assets (if any)
  media/                  # Saved alert images
  db.sqlite3              # SQLite database (dev)
  .env                    # Optional Telegram config
  manage.py
best.pt                   # YOLO model weights (repo root)
```

## How It Works (Step-by-Step)
1. **Login required:** `/accounts/login/` uses Django auth. After login, user is redirected to `/`.
2. **Camera start:** Browser requests camera access (client-side JS in `index.html`).
3. **Frame capture:** Every ~750ms, a frame is captured and sent to `/api/detect/` as base64 JPEG.
4. **YOLO inference:** Backend loads `best.pt` once and runs `model.predict(..., device="cpu")`.
5. **Detection filtering:** Low-confidence detections are skipped using:
   - `MODEL_CONF = 0.3`
   - `HAT_THRESHOLD = 0.7`
   - `NO_HAT_THRESHOLD = 0.4`
6. **Overlay + UI update:** Boxes and labels are drawn on canvas, status and latency update in UI.
7. **Alert capture:** If best detection is `NO HAT` and no `SAFE` exists, an alert image is saved.
8. **Optional Telegram:** A background thread posts a message to Telegram (if configured).
9. **Alerts gallery:** `/alerts/` shows last 5 saved alert images.

## Key URLs
- `/` or `/dashboard/` → Live detection UI
- `/alerts/` → Alerts gallery
- `/api/detect/` → JSON detection endpoint (POST only)
- `/accounts/login/` → Login page

## Detection API (Backend)
**Endpoint**
```
POST /api/detect/
```

**Payload**
```
{ "image": "data:image/jpeg;base64,..." }
```

**Response**
```
{
  "detections": [
    {
      "label": "HAT",
      "confidence": 0.91,
      "status": "SAFE",
      "color": "#00C853",
      "box": { "x1": 10, "y1": 20, "x2": 200, "y2": 180 }
    }
  ],
  "image_size": { "width": 640, "height": 480 },
  "saved_file": "latest_alert3.jpg"
}
```

## Telegram Alerts 
Create a `.env` file in `chefhat_project/`:
```
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```
If these are missing, Telegram notifications are skipped without errors.

## Local Setup (From Git Clone)
1. **Clone repo**
```
git clone https://github.com/SoftwareTechnology-Hub/Chef_Hat_Detection.git
cd Chef_Hat_Detection
```

2. **Create and activate virtual environment**
```
python -m venv venv
```
Windows:
```
venv\Scripts\activate
```
macOS/Linux:
```
source venv/bin/activate
```

3. **Install dependencies**
```
pip install django ultralytics opencv-python numpy python-dotenv
```

4. **Run migrations**
```
python chefhat_project/manage.py migrate
```

5. **Create a superuser (for login)**
```
python chefhat_project/manage.py createsuperuser
```

6. **Start the server**
```
python chefhat_project/manage.py runserver
```

7. **Open in browser**
```
http://127.0.0.1:8000/
```

## Usage Notes
- Allow camera access in the browser.
- If detections are slow, reduce camera resolution in the browser or increase the interval in `index.html`.
- Alert images are stored in `chefhat_project/media/` as `latest_alert1.jpg` ... `latest_alert5.jpg`.
- `DEBUG` is enabled and `ALLOWED_HOSTS` is `["*"]` for development only.

## Troubleshooting
- **Camera not starting:** Ensure HTTPS on production or use localhost in dev.
- **Model not found:** Confirm `best.pt` exists in the repo root.
- **Blank detections:** Check thresholds or confirm model labels include `HAT` / `NO_HAT`.
- **Telegram errors:** Missing token/chat ID will silently skip notifications.
