import streamlit as st 
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase

# Text & frame placeholders
st.set_page_config(
    page_title = "491 ASL Translator",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

# CSS
st.markdown("""
    <style>
        .main { background-color: #111; }
        section[data-testid = "stSidebar"] { background-color: #1a0f6e; }
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
        }
    </style>
""", unsafe_allow_html = True)

# Session states
if "running" not in st.session_state:
    st.session_state.running = False
if "transcript" not in st.session_state:
    st.session_state.transcript = []

# Sidebar
with st.sidebar:
    st.title("ASL Translator")
    st.divider()

    with st.expander("Camera Input"):
        st.selectbox("Camera", ["Default Webcam", "External Camera"])
        st.radio("Resolution", ["480p", "720p", "1080p"])
    
    with st.expander("Language"):
        st.selectbox("Translate to", ["English", "Spanish", "French", "Mandarin"])
    
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
        if st.button("▶ START", use_container_width = True):
            st.session_state.running = True
            st.rerun()
    with col2:
        if st.button("⏸ STOP", use_container_width = True):
            st.session_state.running = False
            st.rerun()

# WebRTC networking config so our connection is stable
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Video feed (DO NOT USE TTS HERE; Can convert for ML later)
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        return frame

# Main WebRTC comp
if st.session_state.running:
    st.success("● Live")
else:
    st.info("Press START to begin")

webrtc_streamer(
    key = "asl",
    rtc_configuration = RTC_CONFIGURATION,
    video_processor_factory = VideoProcessor,
    media_stream_constraints = {"video": True, "audio": False},
    desired_playing_state = st.session_state.running,
)

# Transcript section below video
st.subheader("Transcript")
st.markdown('<div class = "transcript-box">', unsafe_allow_html = True)

# Status & display text
if st.session_state.running:
    #placeholder (replace w/ transcript)
    sample = [
        "Hello my name is Tyler.",
        "Welcome to 491.",
    ]
    for line in sample:
        st.markdown(f'<div class = "chat-bubble">{line}</div', unsafe_allow_html = True)
else:
    st.markdown("<p style = 'color: gray;'>Transcript will show here once running</p>", unsafe_allow_html = True)

st.markdown('</div>', unsafe_allow_html = True)