import streamlit as st 
import cv2
import numpy as np

st.set_page_config(page_title = "491 ASL Translator", layout = "wide")
st.title("491 ASL Translator")
frame_placeholder = st.empty()
predict_placeholder = st.empty()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to use webcam")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels = "RGB")

    # Stub predictor
    predict = np.random.choice(["A", "B", "HELLO"])
    predict_placeholder.markdown(f"### Prediction: **{predict}**")