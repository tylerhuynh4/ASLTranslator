import streamlit as st 
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import threading
import numpy as np
from asl_model import load_predictor


# Text & frame page
st.set_page_config(
    page_title = "Signify",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

# CSS injection into Streamlit
st.markdown("""
    <style>
        .main { background-color: #111; }
        section[data-testid = "stSidebar"] { background-color: #1a0f6e; }
            
        /* expander headers */
        section[data-testid = "stSidebar"] details summary {
            background-color: #555;
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 6px;
            color: white;
        }
            
        .transcript-box {
            background-color: #2a2a2a;
            border-radius: 10px;
            padding: 15px;
            min-height: 200px;
            margin-top: 10px;
        }
        .chat-bubble {
            background-color: #3a3a3a;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 8px;
            color: white;
            font-size: 1.05rem;
        }
    </style>
""", unsafe_allow_html = True)

# Session states
if "running" not in st.session_state:
    st.session_state.running = False
if "transcript" not in st.session_state:
    st.session_state.transcript = []        # list of (word, confidence) tuples

# Load model once & cached across reruns but no arguments
@st.cache_resource(show_spinner = "Loading ASL model...")
def get_predictor():    # use the slider
    return load_predictor()     # default from asl_model

# Sidebar
with st.sidebar:
    st.markdown("<h1 style = 'text-align: center; font-size: 2rem; padding: 10px 0;'>Signify</h1>", unsafe_allow_html = True)
    st.divider()

    with st.expander("Camera Input"):
        st.selectbox("Camera", ["Default Webcam", "External Camera"])
        st.radio("Resolution", ["480p", "720p", "1080p"])
    
    with st.expander("Language"):
        st.selectbox("Translate to", ["English", "Spanish", "French", "Mandarin"])

    with st.expander("Model Settings"):
        confidence_thresh = st.slider("Confidence Threshold", 
                                      0.20, 0.90, 0.35, 0.05,
                                      help = "Higher = Fewer but more confident predictions")
        min_frames_value = st.slider("Min frames before inference", 
                                      4, 16, 8, 2,
                                      help = "Buffer warm up period before model starts predicting")
        smooth_val = st.slider("Smoothing window",
                               1, 10, 5,
                               help = "Averages probabilities over N frames to reduce flickering")
    
    with st.expander("Profile"):
        st.text_input("Name", value = "Default")
        st.slider("Signing Speed", 1, 5, 3)
        st.radio("Dominant Hand", ["Right", "Left"])
        if st.button("Save Profile"):
            st.success("Saved")
    
    with st.expander("Voice"):
        st.toggle("Text-to-Speech")
        st.toggle("Microphone Input")
    
    st.divider()

    # Start/Stop Buttons
    col1, col2 = st.columns(2)
    with col1:
        # Green tint when running
        if st.session_state.running:
            st.markdown("""
                <style>
                    section[data-testid = "stSidebar"]
                    div[data-testid = "column"]: first-child button {
                        background-color: #22c55e !important;
                        color: white !important;
                        border: none !important;
                    }
                </style>
            """, unsafe_allow_html = True)
        if st.button("▶ START", use_container_width = True):
            st.session_state.running = True
            # reset predictor buffer on start/restart
            try:
                get_predictor().reset()     # no args
            except Exception:
                pass
            st.rerun()
    with col2:
        if st.button("⏸ STOP", use_container_width = True):
            st.session_state.running = False
            st.rerun()
    
    if st.button("Clear Transcript", use_container_width = True):
        st.session_state.transcript = []
        st.rerun()

# Load predictor
try: 
    predictor = get_predictor()
    # Keep thresh/smooth in sync w/ sidebar sliders w/o reloading model
    predictor.threshold = confidence_thresh
    predictor.min_frames = min_frames_value
    predictor.smooth = smooth_val
    model_ok = True
except Exception as exc:
    st.error(f"Model failed to load: {exc}")
    model_ok = False

# Shared prediction queue (WebRTC thread -> Streamlit main thread)
_prediction_queue: list[tuple[str, float]] = []
_queue_lock = threading.Lock()

# WebRTC networking config so our connection is stable
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Video feed
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.running = False    # controlled by streamlit thread

    # runs per-frame inference inside WebRTC thread
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        if not self.running or not model_ok:
            return frame

        # av.VideoFrame -> numpy BGR for predictor's expected
        bgr = frame.to_ndarray(format = "bgr24")
        label, confidence = predictor.predict(bgr)  # predict() already thread-safe & uses Lock internally

        import cv2
        display = bgr.copy()
        cv2.putText(display, f"{label} ({confidence: .0%})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

        # ignore non-signs
        if label and label != predictor.non_sign_label:
            with _queue_lock:
                _prediction_queue.append((label, confidence))   # flushed to session_state for rerun
        return av.VideoFrame.from_ndarray(display, format = "bgr24")

# Main WebRTC comp
if st.session_state.running:
    st.success("● Live")
else:
    st.info("Press START to begin")

contxt = webrtc_streamer(
    key = "asl",
    rtc_configuration = RTC_CONFIGURATION,
    video_processor_factory = VideoProcessor if model_ok else None,
    media_stream_constraints = {"video": True, "audio": False},
    desired_playing_state = st.session_state.running,
)

# Sync running state to video processor
if contxt.video_processor:
    contxt.video_processor.running = st.session_state.running

# auto refresh to flush predictions
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval = 1000)

# Flush new predictions from WebRTC thread into session_state
with _queue_lock:
    if _prediction_queue:
        st.session_state.transcript.extend(_prediction_queue)
        _prediction_queue.clear()

# Transcript section below video
st.subheader("Transcript")
st.markdown('<div class = "transcript-box">', unsafe_allow_html = True)

if st.session_state.transcript:
    for word, conf in st.session_state.transcript:
        st.markdown(f'<div class = "chat-bubble">{word} '
                    f'<span style = "color: #888; font-size: 0.8rem;">({conf: .0%})</span></div>',
                    unsafe_allow_html = True,)

# Status & display text
elif st.session_state.running:
    st.markdown("<p style = 'color: gray;'>Detecting signs</p>", unsafe_allow_html = True)
else:
    st.markdown("<p style = 'color: gray;'>Transcript will show here once running</p>", unsafe_allow_html = True)

st.markdown('</div>', unsafe_allow_html = True)

# Fully assembled sentence 
if st.session_state.transcript:
    st.markdown("---")
    sentence = " ".join(word for word, _ in st.session_state.transcript)
    st.markdown(f"**Full sentence:** {sentence}")