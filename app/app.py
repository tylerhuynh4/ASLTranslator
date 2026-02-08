import streamlit as st 
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase

# Text & frame placeholders
st.set_page_config(page_title = "491 ASL Translator", layout = "wide")
st.title("491 ASL Translator")

# Session states
if "running" not in st.session_state:
    st.session_state.running = False
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None

# Start/Stop Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Start"):
        st.session_state.running = True
with col2:
    if st.button("Stop"):
        st.session_state.running = False

# UI placeholders
display_placeholder = st.empty()
status_placeholder = st.empty()

# WebRTC networking config so our connection is stable
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Video feed (DO NOT USE TTS HERE; Can convert for ML later)
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        return frame

# Main WebRTC comp
webrtc_streamer(
    key = "asl",
    rtc_configuration = RTC_CONFIGURATION,
    video_processor_factory = VideoProcessor,
    media_stream_constraints = {"video": True, "audio": False},
    desired_playing_state = st.session_state.running,
)

# Status & display text
if st.session_state.running:
    status_placeholder.success("Camera running")
else:
    status_placeholder.info("Camera stopped")

if st.session_state.last_pred is not None:
    display_placeholder.markdown(f"# Display: **{st.session_state.last_pred}**")
else:
    display_placeholder.markdown(f"# Display: *(waiting for frame...)*")