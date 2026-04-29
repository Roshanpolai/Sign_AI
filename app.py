import streamlit as st
import cv2
import numpy as np
import pickle
from gtts import gTTS
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sign Language AI",
    page_icon="🤟",
    layout="centered"
)

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
.big-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #4CAF50;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #f9f9f9;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
}
.result-box {
    font-size: 28px;
    text-align: center;
    font-weight: bold;
    color: #2196F3;
}
.sentence-box {
    font-size: 22px;
    padding: 10px;
    background-color: #eef6ff;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="big-title">🤟 Sign Language AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Gesture → Text → Speech</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model_path = os.path.join(os.getcwd(), "gesture_model.pkl")

model = None
try:
    with open(model_path, "rb") as f: model = pickle.load(f)
except:
    st.error("❌ Model not found. Please check deployment files.")

# ---------------- MEDIAPIPE ----------------
import mediapipe as mp

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------- SESSION ----------------
if "sentence" not in st.session_state:
    st.session_state.sentence = []

if "prediction" not in st.session_state:
    st.session_state.prediction = "NONE"

# ---------------- SPEECH FUNCTION ----------------
def speak(text):
    if text.strip() == "":
        st.warning("⚠️ No text to speak")
        return

    tts = gTTS(text=text, lang='en')
    tts.save("speech.mp3")
    st.audio("speech.mp3")


# ---------------- CAMERA ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📷 Capture Gesture")

img_file = st.camera_input("Take a picture")

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:

            x_list, y_list, z_list = [], [], []

            for lm in hand.landmark:
                x_list.append(lm.x)
                y_list.append(lm.y)
                z_list.append(lm.z)

            min_x, min_y, min_z = min(x_list), min(y_list), min(z_list)

            row = []
            for lm in hand.landmark:
                row.append(lm.x - min_x)
                row.append(lm.y - min_y)
                row.append(lm.z - min_z)

            max_val = max(row)
            if max_val != 0:
                row = [i / max_val for i in row]

            if model:
                st.session_state.prediction = model.predict([row])[0]
            else:
                st.session_state.prediction = "ERROR"

    else:
        st.session_state.prediction = "NONE"
        st.warning("⚠️ No hand detected")

    st.image(frame, channels="BGR", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- RESULT ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("✋ Detected Gesture")
st.markdown(f'<div class="result-box">{st.session_state.prediction}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- BUTTONS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("➕ Add"):
        if st.session_state.prediction != "NONE":
            if st.session_state.prediction == "SPACE":
                st.session_state.sentence.append(" ")
            else:
                st.session_state.sentence.append(st.session_state.prediction)
        else:
            st.warning("⚠️ No valid gesture")

with col2:
    if st.button("🗑 Clear"):
        st.session_state.sentence.clear()

with col3:
    if st.button("🔊 Speak"):
        sentence = "".join(st.session_state.sentence)
        if sentence.strip() == "":
            st.warning("⚠️ Sentence is empty")
        else:
            speak(sentence)

# ---------------- SENTENCE ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📝 Sentence")
st.markdown(
    f'<div class="sentence-box">{"".join(st.session_state.sentence)}</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Made with ❤️ using AI | Final Year Project")