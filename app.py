import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from twilio.rest import Client

def get_ice_servers():
    try:
        account_sid = st.secrets["twilio_account_sid"]
        auth_token = st.secrets["twilio_auth_token"]
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        st.warning(f"Failed to retrieve Twilio ICE servers: {str(e)}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown(
    "This demo uses a model and code from "
    "https://github.com/robmarkcole/object-detection-app. "
    "Many thanks to the project."
)
