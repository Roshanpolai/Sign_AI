import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import tempfile
from PIL import Image
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SignSpeak",
    page_icon="🤟",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

h1, h2, h3 { color: #f0f0f0; }

.gesture-box {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    backdrop-filter: blur(10px);
    margin: 1rem 0;
}

.gesture-label {
    font-size: 3.5rem;
    font-weight: 700;
    color: #a78bfa;
    letter-spacing: 0.05em;
}

.confidence-label {
    font-size: 1rem;
    color: rgba(255,255,255,0.5);
    margin-top: 0.25rem;
}

.sentence-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    color: #e2e8f0;
    font-size: 1.2rem;
    min-height: 3.5rem;
    word-wrap: break-word;
}

.status-pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}

.status-ok   { background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid rgba(52,211,153,0.4); }
.status-warn { background: rgba(251,191,36,0.2);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.4); }
.status-err  { background: rgba(248,113,113,0.2); color: #f87171; border: 1px solid rgba(248,113,113,0.4); }

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.5rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #8b5cf6, #7c3aed);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(124,58,237,0.4);
}

div[data-testid="stTabs"] button {
    color: rgba(255,255,255,0.6);
    font-family: 'Space Grotesk', sans-serif;
}

.section-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.4);
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path="gesture_model.pkl"):
    if not os.path.exists(path):
        return None, "Model file not found. Place gesture_model.pkl in the app directory."
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


# ── MediaPipe setup ───────────────────────────────────────────────────────────
@st.cache_resource
def get_mediapipe_hands():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )
    return hands, mp_hands


# ── Landmark extraction & preprocessing ──────────────────────────────────────
def extract_and_preprocess(image_rgb, hands_detector):
    """
    Returns (features, annotated_image, error_string)
    features: flat numpy array of 63 normalised values (21 landmarks × x,y,z)
    """
    results = hands_detector.process(image_rgb)

    annotated = image_rgb.copy()
    mp_drawing = mp.solutions.drawing_utils
    mp_hands_mod = mp.solutions.hands

    if not results.multi_hand_landmarks:
        return None, annotated, "No hand detected in the image."

    hand_landmarks = results.multi_hand_landmarks[0]

    # Draw landmarks on the copy
    mp_drawing.draw_landmarks(
        annotated,
        hand_landmarks,
        mp_hands_mod.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(167, 139, 250), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(109, 40, 217), thickness=2),
    )

    # Extract raw x, y, z for all 21 landmarks
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])  # (21, 3)

    # Normalize: translate so wrist (landmark 0) is origin
    coords -= coords[0]

    # Scale: divide by the max absolute value to put everything in [-1, 1]
    scale = np.abs(coords).max()
    if scale > 0:
        coords /= scale

    features = coords.flatten()  # (63,)
    return features, annotated, None


# ── Predict ───────────────────────────────────────────────────────────────────
def predict_gesture(model, features):
    """Returns (label, confidence) or raises."""
    features_2d = features.reshape(1, -1)
    label = model.predict(features_2d)[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_2d)[0]
        confidence = float(proba.max())
    return str(label), confidence


# ── Text-to-speech ────────────────────────────────────────────────────────────
def synthesize_speech(text):
    """Returns BytesIO of mp3 audio or None on failure."""
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf
    except Exception as e:
        st.warning(f"TTS unavailable: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🤟 SignSpeak")
st.markdown("Real-time sign language recognition · MediaPipe + ML")

# ── Model status ──────────────────────────────────────────────────────────────
model, model_err = load_model()
hands_detector, _ = get_mediapipe_hands()

if model_err:
    st.markdown(f'<span class="status-pill status-err">⚠ {model_err}</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="status-pill status-ok">✓ Model loaded</span>', unsafe_allow_html=True)

st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
if "sentence" not in st.session_state:
    st.session_state.sentence = []
if "last_gesture" not in st.session_state:
    st.session_state.last_gesture = None
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None

# ── Input tabs ────────────────────────────────────────────────────────────────
tab_cam, tab_upload = st.tabs(["📷  Camera", "🖼  Upload Image"])

image_rgb = None  # will hold the image to process

with tab_cam:
    st.markdown("Take a photo of your hand gesture using the camera below.")
    cam_image = st.camera_input("Point your hand at the camera and snap a photo")
    if cam_image is not None:
        pil_img = Image.open(cam_image).convert("RGB")
        image_rgb = np.array(pil_img)

with tab_upload:
    st.markdown("Upload a photo if the camera is unavailable or you prefer a file.")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(pil_img)

# ── Process ───────────────────────────────────────────────────────────────────
if image_rgb is not None:
    features, annotated_img, detect_err = extract_and_preprocess(image_rgb, hands_detector)

    col_img, col_result = st.columns([1, 1], gap="medium")

    with col_img:
        st.markdown('<p class="section-title">Detected Landmarks</p>', unsafe_allow_html=True)
        st.image(annotated_img, use_column_width=True)

    with col_result:
        st.markdown('<p class="section-title">Prediction</p>', unsafe_allow_html=True)

        if detect_err:
            st.markdown(f'<div class="gesture-box"><span class="gesture-label">—</span><p class="confidence-label">{detect_err}</p></div>', unsafe_allow_html=True)
            st.session_state.last_gesture = None

        elif model is None:
            st.markdown('<div class="gesture-box"><span class="gesture-label">—</span><p class="confidence-label">No model loaded</p></div>', unsafe_allow_html=True)

        else:
            try:
                label, confidence = predict_gesture(model, features)
                st.session_state.last_gesture = label
                st.session_state.last_confidence = confidence

                conf_str = f"{confidence:.0%} confidence" if confidence is not None else ""
                st.markdown(
                    f'<div class="gesture-box">'
                    f'<div class="gesture-label">{label}</div>'
                    f'<div class="confidence-label">{conf_str}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Add to sentence button
                if st.button(f'➕  Add "{label}" to sentence'):
                    st.session_state.sentence.append(label)
                    st.rerun()

            except Exception as e:
                st.error(f"Prediction error: {e}")

# ── Sentence builder ──────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📝 Sentence Builder")

sentence_text = " ".join(st.session_state.sentence) if st.session_state.sentence else ""
display_text = sentence_text if sentence_text else "Your sentence will appear here…"
st.markdown(f'<div class="sentence-box">{display_text}</div>', unsafe_allow_html=True)

col_add_space, col_clear, col_speak = st.columns([1, 1, 1], gap="small")

with col_add_space:
    if st.button("␣  Add Space"):
        if st.session_state.sentence:
            st.session_state.sentence.append(" ")
            st.rerun()

with col_clear:
    if st.button("🗑  Clear"):
        st.session_state.sentence = []
        st.session_state.last_gesture = None
        st.rerun()

with col_speak:
    speak_disabled = not sentence_text.strip()
    if st.button("🔊  Speak", disabled=speak_disabled):
        audio_buf = synthesize_speech(sentence_text.strip())
        if audio_buf:
            st.audio(audio_buf, format="audio/mp3")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="text-align:center;color:rgba(255,255,255,0.25);font-size:0.8rem;">'
    "SignSpeak · MediaPipe Hands + scikit-learn · Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
