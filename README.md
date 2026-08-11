# Computer Vision Based Geo Tracking of Waste and Alerting System

A prototype system that detects garbage bins from CCTV/photo images, classifies
their fill level (Empty / Partial / Full), geo-tags each detection, stores it in
a database, and visualizes it on an interactive map with a Full → Partial →
Empty collection priority order for sanitary workers.

---

## 1. Overview

| Stage | What happens |
|---|---|
| Image capture | Photo from a fixed municipal CCTV camera (or manual upload for testing) |
| Bin detection | YOLOv8 (fine-tuned) locates the garbage bin in the frame |
| Fill classification | MobileNetV2 (fine-tuned) classifies the cropped bin as Empty / Partial / Full |
| Geo-tagging | Detection is tagged with the camera's fixed GPS location + timestamp |
| Storage | Saved to a local SQLite database |
| Visualization | Plotted on an interactive map, colour-coded by severity |
| Route priority | Bins sorted Full → Partial → Empty for collection staff |

Two interchangeable front-ends are included:
- **Flask + Leaflet** (`backend/app.py` + `frontend/index.html`) — REST API + HTML/JS dashboard
- **Streamlit** (`streamlit_app.py`) — single-command Python-only dashboard using `pydeck` for the map

Both use the exact same `ai_model/predict.py` and database schema, so predictions are identical regardless of which UI you run.

---

## 2. Folder Structure

```
Computer-Vision-based-geo-tracking-of-waste-and-alerting-system/
├── ai_model/
│   ├── best.pt              # fine-tuned YOLOv8 weights (bin detection)
│   ├── classifier.pth       # fine-tuned MobileNetV2 weights (fill classification)
│   └── predict.py           # shared inference pipeline used by both front-ends
├── backend/
│   ├── app.py                # Flask REST API (upload / history / priority endpoints)
│   └── database.py           # SQLite schema + init
├── frontend/
│   └── index.html             # Leaflet + OpenStreetMap dashboard (used with Flask)
├── database/
│   └── garbage.db             # SQLite database (auto-created on first run)
├── uploads/                    # saved copies of uploaded/captured images
├── streamlit_app.py            # Streamlit dashboard (alternative to Flask+HTML)
├── requirements.txt
└── README.md
```

---

## 3. Requirements

- Python 3.9+
- pip

Install all dependencies:
```bash
pip install -r requirements.txt
```
`requirements.txt` includes: `streamlit`, `pydeck`, `pandas`, `ultralytics`, `torch`, `torchvision`, `pillow`, plus `flask` if you're running the Flask version (add it if missing: `pip install flask`).

---

## 4. Running the Project

### Option A — Streamlit (recommended, single command)
```bash
streamlit run streamlit_app.py
```
Opens automatically at `http://localhost:8501`. Upload a bin photo, enter (or leave default) latitude/longitude, click **Upload & Predict**.

### Option B — Flask + Leaflet
```bash
python -m backend.app
```
Then open `http://127.0.0.1:5000` in your browser. **Do not** open `frontend/index.html` directly by double-clicking — it must be served by Flask, or the API calls and browser geolocation will fail.

---

## 5. Retraining the Models

Both models were fine-tuned via transfer learning on Google Colab:

- **YOLOv8** — fine-tuned on a small Roboflow "dustbin detection" dataset (~150 images, 1 class)
- **MobileNetV2** — fine-tuned on the CDCM (Clean/Dirty Containers Montevideo) dataset for Empty/Full, plus a self-collected set of ~50-60 photos for the Partial class

See the project report for full training steps, hyperparameters, and the exact dataset-preparation scripts. Retrained weights should be dropped in as `ai_model/best.pt` and `ai_model/classifier.pth`, replacing the existing files — no code changes needed since `predict.py` loads by filename.

---

## 6. How CCTV Image Retrieval Works (Deployment Model)

The prototype accepts a manually uploaded photo for demonstration. In a real deployment, images would instead be pulled automatically from existing municipal CCTV infrastructure. There are three practical ways to do this, in increasing order of integration effort:

| Method | How it works | Best for |
|---|---|---|
| **RTSP live stream pull** | Most IP CCTV cameras expose a live video stream over RTSP (e.g. `rtsp://camera-ip:554/stream1`). A script uses OpenCV (`cv2.VideoCapture(rtsp_url)`) to grab a frame every N seconds and pass it to `predict()` | Cameras with network/IP access, most common for modern municipal setups |
| **NVR/VMS scheduled export** | Most cities record CCTV footage into an NVR (Network Video Recorder) or VMS (Video Management System, e.g. Milestone, Hikvision iVMS). These typically expose an HTTP/SDK API to pull recent clips or snapshots on a schedule | Cities with centralized recording systems already in place |
| **Manual/periodic snapshot job** | A scheduled task (e.g. `cron`) hits each camera's snapshot URL (many IP cameras expose a JPEG snapshot endpoint like `http://camera-ip/snapshot.jpg`) at fixed intervals and saves it to `uploads/` | Simplest integration; works even with older/basic IP cameras |

Example of the RTSP-pull pattern (conceptual — actual camera URL/credentials depend on the vendor):
```python
import cv2
import time
from ai_model.predict import predict

cap = cv2.VideoCapture("rtsp://username:password@camera-ip:554/stream1")

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("uploads/frame.jpg", frame)
        result = predict("uploads/frame.jpg")
        # then insert_detection(...) with this camera's fixed lat/lng
    time.sleep(60)  # sample once a minute, not every frame
```

For a college prototype, none of this needs to be built — the manual upload button demonstrates the same downstream pipeline (detection → classification → geo-tag → map) that would run automatically in production.

---

## 7. How GPS / Geo-Tagging Works

This is a common point of confusion, so it's worth being explicit: **the GPS coordinate does not come from the image itself in the CCTV deployment model** — CCTV cameras are fixed in place, so their location is known in advance and never changes. There is no live GPS sensor on the camera.

The geo-tagging approach differs depending on the image source:

| Source | How location is obtained |
|---|---|
| **Fixed municipal CCTV camera** | Each camera is registered once, at installation time, with its known fixed latitude/longitude (surveyed manually or read once off a phone/GPS device at the install site). This is stored in a simple lookup — e.g. a `cameras` table (`camera_id, latitude, longitude, location_name`) — and every detection from that camera is tagged with its registered coordinate. |
| **Phone camera upload (current prototype)** | The browser's Geolocation API (Flask/HTML version) or a manually entered value (Streamlit version) supplies the coordinate at the moment of upload, since a phone moves around and has no fixed registered position. |
| **Dashcam / moving vehicle (earlier design, since replaced)** | Would have required a live GPS module reporting the vehicle's real-time position alongside each captured frame — this is why the fixed-CCTV approach is actually simpler: no live GPS hardware is needed at all once a camera's location is registered. |

**In short: for fixed CCTV, GPS is set once per camera at registration time, not re-detected per image.** This is a real practical advantage of the CCTV-based design over the original dashcam idea — it removes the need for any GPS hardware entirely, since the location is already known and constant.

### How this maps to the current code
Right now, `predict()` only returns the prediction — it doesn't know about camera identity or location. To reflect the "fixed CCTV" model properly, the calling code (wherever `predict()` is invoked — `streamlit_app.py` or `backend/app.py`) should attach a **pre-registered coordinate per camera**, rather than asking for GPS at upload time. A simple way to do this:

```python
# One-time setup: register each camera's fixed location
CAMERA_LOCATIONS = {
    "camera_01": {"name": "Gandhi Road Junction", "lat": 11.3410, "lng": 78.1450},
    "camera_02": {"name": "Market Street Bin Cluster", "lat": 11.3420, "lng": 78.1465},
}

# When a frame arrives from a known camera:
camera_id = "camera_01"  # this would come from whichever camera captured the frame
loc = CAMERA_LOCATIONS[camera_id]
result = predict(frame_path)
insert_detection(frame_path, result["prediction"], result["confidence"], loc["lat"], loc["lng"])
```

For your report/viva: this distinction — **"GPS is fixed per camera, registered once, not derived from the image"** — is exactly the kind of detail reviewers ask about, and having a clear answer (rather than implying each photo carries live GPS metadata) will read as more technically sound.

---

## 8. Known Limitations (Prototype Scope)

- YOLOv8 was fine-tuned on a small (~150 image) single-source dataset — detection may miss unusual angles or lighting not represented in training.
- The "Partial" fill-level class was self-collected (~50-60 images) separately from the "Empty"/"Full" classes (sourced from CDCM), which introduces some visual-style mismatch between classes — documented as a known limitation, not a bug.
- CCTV image retrieval (Section 6) is a proposed deployment design, not yet implemented in code — the current system accepts manual image uploads for demonstration.
- Route priority uses a simple severity + distance sort, not a full shortest-path algorithm (Dijkstra/A*), by design for this prototype scope.

---

## 9. Future Scope

- Automate frame capture from live CCTV (RTSP) instead of manual upload
- Expand and rebalance the fill-level training dataset across all three classes
- Migrate from SQLite to PostgreSQL for multi-camera, city-scale deployment
- Add a `cameras` table for proper per-camera fixed-location registration (see Section 7)
- Replace the heuristic route ordering with a real shortest-path optimization across the road network

---

## 10. Credits / Datasets Used

- Detection dataset: "Dustbin Detection" (Roboflow Universe)
- Classification dataset: CDCM — Clean/Dirty Containers Montevideo (Kaggle)
- Frameworks: YOLOv8 (Ultralytics), PyTorch/TorchVision (MobileNetV2), Flask, Streamlit, Leaflet.js, OpenStreetMap
