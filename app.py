import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

FACE_DETECTOR_PATH = "haarcascade_frontalface_default.xml"


class FaceDetector(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + FACE_DETECTOR_PATH)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return img


st.title("Real-time Face Detection")
st.markdown(
    "This app uses the Streamlit-WebRTC component to access the webcam in real time and perform face detection using OpenCV.")

webrtc_streamer(key="example1", video_transformer_factory=FaceDetector)
