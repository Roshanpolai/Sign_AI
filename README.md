# 🤟 SignSpeak — Sign Language Recognition App

A Streamlit web app that uses MediaPipe Hands + your trained scikit-learn model to recognise sign language gestures and build spoken sentences.

---

## Project Structure

```
your-project/
├── app.py                  ← Main Streamlit application
├── gesture_model.pkl       ← YOUR trained model (add this!)
├── requirements.txt        ← Python dependencies
├── render.yaml             ← Render deployment config
└── README.md
```

---

## Local Setup

### 1 — Prerequisites
- Python 3.10.x (use `pyenv` or the official installer)
- Your `gesture_model.pkl` file in the same folder as `app.py`

### 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### 3 — Run locally
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## Deploy on Render (Free tier works)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/signspeakapp.git
git push -u origin main
```

> ⚠️ Make sure `gesture_model.pkl` is included in the repository,  
> or upload it via Render's persistent disk (see Step 4).

### Step 2 — Create a new Web Service on Render
1. Go to https://dashboard.render.com → **New → Web Service**
2. Connect your GitHub repo
3. Render will auto-detect `render.yaml` — review the settings

### Step 3 — Configure (if not using render.yaml)
| Field | Value |
|---|---|
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true` |
| **Python version** | `3.10.14` (set via `PYTHON_VERSION` env var) |

### Step 4 — Model file on Render
Because Render's free tier has an ephemeral filesystem, the simplest options are:

| Option | How |
|---|---|
| **Commit model to repo** | Include `gesture_model.pkl` in git (works if < 100 MB) |
| **Render Disk** | Add a persistent disk, mount at `/app`, place model there, update `load_model("/app/gesture_model.pkl")` |
| **Download at startup** | Host model on S3/GCS, add a `build.sh` that downloads it before Streamlit starts |

### Step 5 — Deploy
Click **Create Web Service**. Render builds and starts the app automatically. Your public URL will be shown in the dashboard (e.g. `https://signspeakapp.onrender.com`).

---

## How the app works

```
Image (camera / upload)
        │
        ▼
  OpenCV converts to RGB
        │
        ▼
  MediaPipe Hands
  → 21 landmarks (x, y, z)
        │
        ▼
  Normalization
  • Translate: wrist → origin
  • Scale: divide by max abs value → [-1, 1]
  → 63 features flat array
        │
        ▼
  gesture_model.pkl  (.predict / .predict_proba)
        │
        ▼
  Predicted label + confidence
        │
        ▼
  Sentence builder  →  gTTS  →  Audio playback
```

---

## Model compatibility

The app expects your model to:
- Accept a `(1, 63)` numpy array (21 landmarks × x, y, z, normalised)
- Implement `predict()` (required)
- Optionally implement `predict_proba()` for confidence display

Standard scikit-learn classifiers (RandomForest, SVM, etc.) work out of the box.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Model file not found` | Place `gesture_model.pkl` next to `app.py` |
| `No hand detected` | Ensure hand is visible, well-lit, and unobstructed |
| Camera not working | Switch to the **Upload Image** tab |
| gTTS error | Render needs outbound internet access (enabled by default) |
| `libGL` / OpenCV error on deploy | `opencv-python-headless` is already in requirements — do not install `opencv-python` |
