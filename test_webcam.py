#!/usr/bin/env python3
"""
Simple test script to verify webcam functionality
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np

class SimpleVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Add a simple text overlay to show it's working
        cv2.putText(img, "Webcam Test - Working!", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img

def main():
    st.title("Webcam Test")
    st.write("Testing webcam functionality...")
    
    # WebRTC configuration
    rtc_configuration = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    
    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="test",
        video_transformer_factory=SimpleVideoTransformer,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if webrtc_ctx.state.playing:
        st.success("Webcam is working!")
    else:
        st.info("Click 'Start Webcam' to test")

if __name__ == "__main__":
    main() 