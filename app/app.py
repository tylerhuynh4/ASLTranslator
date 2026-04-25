import streamlit as st 
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import sys
from pathlib import Path
from asl_model import load_predictor
import shared_q_test as _Q 
import cv2
from streamlit_autorefresh import st_autorefresh

# Proj root importable so we can pull
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from speech.translator import GoogleTranslator
from speech.tts import GoogleTTS

# Language code maps 
LANGUAGE_CODES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Mandarin": "zh",
}
 
TTS_CODES = {
    "English": "en-US",
    "Spanish": "es-US",
    "French": "fr-FR",
    "Mandarin": "cmn-CN",
}
 
PROJECT_ID = "single-cycling-470616-e0"

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
        .translation-box {
            background-color: #202020;
            border-radius: 10px;
            padding: 15px;
            margin-top: 10px;
            color: white;
            font-size: 1.1rem;
            min-height: 55px;
        }
    </style>
""", unsafe_allow_html = True)

# Session states
if "running" not in st.session_state:
    st.session_state.running = False
if "transcript" not in st.session_state:
    st.session_state.transcript = []        # list of (word, confidence) tuples
if "last_spoken_text" not in st.session_state:
    st.session_state.last_spoken_text = ""  # no same text every autorefresh

# retranslation cache | skip API calls when sentence + target no change
if "last_translated_sentence" not in st.session_state:
    st.session_state.last_translated_sentence= ""
if "last_translated_target" not in st.session_state:
    st.session_state.last_translated_target = ""
if "last_translated_output" not in st.session_state:
    st.session_state.last_translated_output = ""

# Load model once & cached across reruns but no arguments
@st.cache_resource(show_spinner = "Loading ASL model...")
def get_predictor():    # use the slider
    return load_predictor()     # default from asl_model

@st.cache_resource(show_spinner = False)
def get_translator(project_id: str):
    return GoogleTranslator(project_id = project_id)

@st.cache_resource(show_spinner = False)
def get_tts(language_code: str):
    return GoogleTTS(language_code = language_code)

# Sidebar
with st.sidebar:
    st.markdown("<h1 style = 'text-align: center; font-size: 2rem; padding: 10px 0;'>Signify</h1>", unsafe_allow_html = True)
    st.divider()

    with st.expander("Camera Input"):
        st.selectbox("Camera", ["Default Webcam", "External Camera"])
        st.radio("Resolution", ["480p", "720p", "1080p"])
    
    with st.expander("Language"):
        selected_language = st.selectbox("Translate to", ["English", "Spanish", "French", "Mandarin"])

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
        enable_tts = st.toggle("Text-to-Speech", value = True)
        st.toggle("Microphone Input")
    
    st.divider()

    # Start/Stop Buttons
    if st.session_state.running:
        st.markdown("""
            <style>
                section[data-testid="stSidebar"] button[kind="primary"] {
                    background-color: #22c55e !important;
                    color: white !important;
                    border: none !important;
                }
                section[data-testid="stSidebar"] button[kind="primary"]:hover {
                    background-color: #16a34a !important;
                    border: none !important;
                }
            </style>
        """, unsafe_allow_html = True)
 
    col1, col2 = st.columns(2)
    with col1:
        start_clicked = st.button(
            "▶ START",
            use_container_width = True,
            type = "primary" if st.session_state.running else "secondary",
            key = "start_btn",
        )
        if start_clicked:
            st.session_state.running = True
            # reset predictor buffer on start/restart
            try:
                get_predictor().reset()     # no args
            except Exception:
                pass
            st.rerun()
    with col2:
        if st.button("⏸ STOP", use_container_width = True, key = "stop_btn"):
            st.session_state.running = False
            st.rerun()
 
    if st.button("Clear Transcript", use_container_width = True):
        st.session_state.transcript = []
        st.session_state.last_spoken_text = ""
        st.session_state.last_translated_sentence = ""
        st.session_state.last_translated_target = ""
        st.session_state.last_translated_output = ""
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

# WebRTC networking config so our connection is stable
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Video feed
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self._last_label = None     # track last label to avoid dupes

    # runs per-frame inference inside WebRTC thread
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        if not model_ok:
            return frame

        # av.VideoFrame -> numpy BGR for predictor's expected
        bgr = frame.to_ndarray(format = "bgr24")
        label, confidence = predictor.predict(bgr)  # predict() already thread-safe & uses Lock internally
        print(f"RECV: label = {label}, confidence = {confidence: .2f}, non_sign = {predictor.non_sign_label}")

        display = bgr.copy()
        cv2.putText(display, f"{label} ({confidence: .0%})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

        # only append when label changes is ignores non-signs
        if label and label != predictor.non_sign_label and label != self._last_label:
            with _Q.lock:
                _Q.data.append((label, confidence))   # flushed to session_state for rerun
            self._last_label = label
        elif label == predictor.non_sign_label:
            self._last_label = None     # resets so same sign can appear after pause

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

# auto refresh to flush predictions but only when running   
if st.session_state.running:
    st_autorefresh(interval = 1000, key = "asl_refresh")

# Flush new predictions from WebRTC thread into session_state
with _Q.lock:
    if _Q.data:
        st.session_state.transcript.extend(_Q.data)
        _Q.data.clear()

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

    # Live translation
    st.subheader("Live Translation")
    target_code = LANGUAGE_CODES[selected_language]
    translated_sentence = ""

    if target_code != "en":
        needs_translation = (sentence != st.session_state.last_translated_sentence or
                             target_code != st.session_state.last_translated_sentence)
        if needs_translation:
            try:
                translator = get_translator(PROJECT_ID)
                translated_sentence = translator.translate_text(sentence, target_language = target_code, source_language = "en")
                st.session_state.last_translated_sentence = sentence
                st.session_state.last_translated_target = target_code
                st.session_state.last_translated_output = translated_sentence
            except Exception as exc:
                st.error(f"Translation failed: {exc}")
        else:
            translated_sentence = st.session_state.last_translated_output     # no translation if english
    else:
        translated_sentence = sentence

    st.markdown(f'<div class = "translation-box"><b>Translated sentence:</b><br>{translated_sentence}</div>',
                unsafe_allow_html = True,)

    # TTS 
    spoken_text = translated_sentence if translated_sentence else sentence

    if enable_tts and spoken_text and spoken_text != st.session_state.last_spoken_text:
        try:
            tts_language = "en-US" if selected_language == "English" else TTS_CODES[selected_language]
            tts = get_tts(tts_language)
            result = tts.synthesize(spoken_text)
            audio_bytes = Path(result.audio_path).read_bytes()
            st.audio(audio_bytes, format = "audio/mp3", autoplay = True)
            st.session_state.last_spoken_text = spoken_text
        except Exception as exc:
            st.error(f"TTS Failed: {exc}")
