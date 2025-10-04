#!/usr/bin/env python3
"""
Streamlit Web App for Space Station Object Detection
Real-time detection of Toolbox, Oxygen Tank, and Fire Extinguisher
"""

import streamlit as st
import cv2
import numpy as np
import yaml
import tempfile
import os
from pathlib import Path
from ultralytics import YOLO
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import pandas as pd
import zipfile
import io
import base64
from datetime import datetime
import threading
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import queue

model = YOLO("yolov8n.pt")

# Page configuration
st.set_page_config(
    page_title="Space Station Object Detection",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class LiveObjectDetection(VideoTransformerBase):
    def __init__(self, model, class_names, class_colors, conf_threshold=0.25):
        self.model = model
        self.class_names = class_names
        self.class_colors = class_colors
        self.conf_threshold = conf_threshold
        self.detection_queue = queue.Queue()
        self.frame_count = 0
        
    def transform(self, frame):
        if self.model is None:
            return frame
        
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # Run detection every 3 frames to maintain performance
            self.frame_count += 1
            if self.frame_count % 3 == 0:
                try:
                    # Run prediction
                    results = self.model.predict(
                        source=img,
                        conf=self.conf_threshold,
                        save=False
                    )
                    
                    if results and len(results) > 0:
                        result = results[0]
                        
                        # Draw detections
                        if result.boxes is not None and len(result.boxes) > 0:
                            for box in result.boxes:
                                # Get coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # Get class and confidence
                                class_id = int(box.cls[0])
                                confidence = float(box.conf[0])
                                
                                if class_id < len(self.class_names):
                                    class_name = self.class_names[class_id]
                                    
                                    # Get color
                                    color = self.class_colors.get(class_name, (255, 255, 255))
                                    if isinstance(color, str):
                                        # Convert hex to BGR
                                        color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))[::-1]
                                    
                                    # Draw bounding box
                                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                                    
                                    # Draw label
                                    label = f"{class_name}: {confidence:.2f}"
                                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                    cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                                                 (x1 + label_size[0], y1), color, -1)
                                    cv2.putText(img, label, (x1, y1 - 5), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                    
                                    # Add detection info to queue for display
                                    detection_info = {
                                        'class': class_name,
                                        'confidence': confidence,
                                        'bbox': [x1, y1, x2, y2],
                                        'timestamp': time.time()
                                    }
                                    try:
                                        self.detection_queue.put_nowait(detection_info)
                                    except queue.Full:
                                        pass  # Queue is full, skip this detection
                                        
                except Exception as e:
                    print(f"Detection error: {e}")
            
            return img
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame

class ObjectDetectionApp:
    def __init__(self):
        """Initialize the Streamlit app."""
        self.model = None
        self.config = self.load_config()
        # ✅ Fixed class mapping based on training
        self.class_names = [
            'Fire Extinguisher',   # class 2
            'Toolbox',            # class 0
            'Oxygen Tank'        # class 1
        ]
        self.class_colors = {
            'Toolbox': "#E56626",
            'Oxygen Tank': "#D4CEC1",
            'Fire Extinguisher': "#FF0000"
        }
        
    def load_config(self):
        """Load configuration from YAML file."""
        try:
            with open('config.yaml', 'r') as file:
                config = yaml.safe_load(file)
            # ✅ Keep training order
            config['names'] = {
                0: 'Fire Extinguisher',
                1: 'Toolbox',
                2: 'Oxygen Tank'
            }
            return config
        except Exception as e:
            st.error(f"Error loading config: {e}")
            return {'names': {1: 'Toolbox', 2: 'Oxygen Tank', 0: 'Fire Extinguisher'}}

    def load_model(self, model_path):
        """Load trained YOLOv8 model."""
        try:
            self.model = YOLO(model_path)
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def predict_image(self, image, conf_threshold=0.25):
        """Run prediction on image."""
        if self.model is None:
            return None
        
        try:
            results = self.model.predict(
                source=image,
                conf=conf_threshold,
                save=False
            )
            return results[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def draw_detections(self, image, result):
        """Draw bounding boxes and labels on image."""
        if result is None or len(result.boxes) == 0:
            return image
        
        # Convert PIL to OpenCV format
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        annotated_image = image.copy()
        
        for box in result.boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get class and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.class_names[class_id]
            
            # Get color
            color = self.class_colors.get(class_name, (255, 255, 255))
            if isinstance(color, str):
                # Convert hex to BGR
                color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))[::-1]
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Convert back to RGB for display
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        return annotated_image
    
    def create_detection_stats(self, result):
        """Create statistics from detection results."""
        if result is None:
            return pd.DataFrame()
        
        stats = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.class_names[class_id]
            
            stats.append({
                'Class': class_name,
                'Confidence': confidence,
                'Bounding Box': f"({int(box.xyxy[0][0])}, {int(box.xyxy[0][1])}, {int(box.xyxy[0][2])}, {int(box.xyxy[0][3])})"
            })
        
        return pd.DataFrame(stats)
    
    def create_confusion_chart(self, result):
        """Create a pie chart of detected classes."""
        if result is None or len(result.boxes) == 0:
            return None
        
        class_counts = {}
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if not class_counts:
            return None
        
        fig = go.Figure(data=[go.Pie(
            labels=list(class_counts.keys()),
            values=list(class_counts.values()),
            hole=0.3,
            marker_colors=[self.class_colors.get(cls, '#808080') for cls in class_counts.keys()]
        )])
        
        fig.update_layout(
            title="Detected Objects Distribution",
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_download_zip(self, files_to_include):
        """Create a ZIP file containing specified files for download."""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in files_to_include:
                if os.path.exists(file_path):
                    # Get the filename without path
                    filename = os.path.basename(file_path)
                    zip_file.write(file_path, filename)
        
        zip_buffer.seek(0)
        return zip_buffer
    
    def get_file_download_link(self, file_path, file_label):
        """Generate a download link for a single file."""
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, "rb") as f:
            bytes_data = f.read()
        
        b64 = base64.b64encode(bytes_data).decode()
        filename = os.path.basename(file_path)
        
        return f'<a href="data:file/{filename};base64,{b64}" download="{filename}">{file_label}</a>'
    
    def get_performance_files(self):
        """Get list of available performance analysis files."""
        performance_files = {
            'confusion_matrix.png': 'Confusion Matrix',
            'performance_metrics.png': 'Performance Metrics', 
            'training_curves.png': 'Training Curves',
            'performance_dashboard.html': 'Interactive Dashboard',
            'performance_report.md': 'Performance Report',
            'PERFORMANCE_SUMMARY.md': 'Performance Summary'
        }
        
        available_files = []
        for file_path, label in performance_files.items():
            if os.path.exists(file_path):
                available_files.append((file_path, label))
        
        return available_files

def main():
    st.title("🚀 Space Station Object Detection")
    st.markdown("Detect Toolbox, Oxygen Tank, and Fire Extinguisher in space station environments")
    
    # Initialize app
    app = ObjectDetectionApp()
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Model selection
    model_path = st.sidebar.selectbox(
        "Select Model",
        ["runs/detect/train/weights/best.pt", "runs/detect/train/weights/last.pt", "runs/detect/train2/weights/best.pt", "runs/detect/train2/weights/last.pt"],
        help="Choose the trained model to use for detection"
    )
    
    # Load model
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            if app.load_model(model_path):
                st.sidebar.success("Model loaded successfully!")
            else:
                st.sidebar.error("Failed to load model!")
    
    # Confidence threshold
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Image Upload", "📹 Live Webcam", "📊 Analytics", "📥 Downloads"])
    
    with tab1:
        st.header("Image Upload")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to detect objects"
        )
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Run detection
            if app.model is not None:
                with st.spinner("Running detection..."):
                    result = app.predict_image(image, conf_threshold)
                
                if result is not None:
                    # Draw detections
                    annotated_image = app.draw_detections(image, result)
                    
                    with col2:
                        st.subheader("Detection Results")
                        st.image(annotated_image, caption="Detected Objects", use_column_width=True)
                    
                    # Show statistics
                    stats_df = app.create_detection_stats(result)
                    if not stats_df.empty:
                        st.subheader("Detection Statistics")
                        st.dataframe(stats_df, use_container_width=True)
                    
                    # Download detection results
                    st.subheader("📥 Download Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Download annotated image
                        annotated_pil = Image.fromarray(annotated_image)
                        img_buffer = io.BytesIO()
                        annotated_pil.save(img_buffer, format='PNG')
                        img_buffer.seek(0)
                        
                        st.download_button(
                            label="📷 Download Annotated Image",
                            data=img_buffer.getvalue(),
                            file_name=f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png"
                        )
                    
                    with col2:
                        # Download statistics as CSV
                        if not stats_df.empty:
                            csv_buffer = io.BytesIO()
                            stats_df.to_csv(csv_buffer, index=False)
                            csv_buffer.seek(0)
                            
                            st.download_button(
                                label="📊 Download Statistics (CSV)",
                                data=csv_buffer.getvalue(),
                                file_name=f"detection_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    with col3:
                        # Download detection report
                        report_text = f"""
# Detection Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Image Information
- Original filename: {uploaded_file.name}
- Detection confidence threshold: {conf_threshold}
- Total detections: {len(result.boxes)}

## Detection Results
"""
                        for i, box in enumerate(result.boxes):
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            class_name = app.class_names[class_id]
                            report_text += f"\n### Detection {i+1}\n"
                            report_text += f"- Class: {class_name}\n"
                            report_text += f"- Confidence: {confidence:.3f}\n"
                            report_text += f"- Bounding Box: {box.xyxy[0].cpu().numpy().tolist()}\n"
                        
                        report_buffer = io.BytesIO()
                        report_buffer.write(report_text.encode())
                        report_buffer.seek(0)
                        
                        st.download_button(
                            label="📄 Download Report (MD)",
                            data=report_buffer.getvalue(),
                            file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    
                    # Show chart
                    chart = app.create_confusion_chart(result)
                    if chart is not None:
                        st.plotly_chart(chart, use_container_width=True)
                else:
                    st.error("Detection failed!")
            else:
                st.warning("Please load a model first!")
    
    with tab2:
        st.header("📹 Live Webcam Detection")
        st.markdown("Real-time object detection using your webcam")
        
        if app.model is None:
            st.warning("⚠️ Please load a model first in the sidebar before using live webcam detection!")
            st.info("💡 Click 'Load Model' in the sidebar to get started.")
        else:
            # Webcam controls
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎥 Webcam Controls")
                st.markdown("""
                **Instructions:**
                1. Click 'Start Webcam' to begin live detection
                2. Point your camera at objects to detect
                3. Supported objects: Toolbox, Fire Extinguisher, Oxygen Tank
                4. Click 'Stop Webcam' to end the session
                """)
                
                # Detection settings
                st.subheader("⚙️ Detection Settings")
                detection_fps = st.slider("Detection FPS", 1, 30, 10, help="Frames per second for object detection")
                show_confidence = st.checkbox("Show Confidence Scores", value=True)
                
            with col2:
                st.subheader("📊 Live Statistics")
                # Placeholder for live stats
                stats_placeholder = st.empty()
                chart_placeholder = st.empty()
            
            # Live detection statistics
            if 'detection_history' not in st.session_state:
                st.session_state.detection_history = []
            
            # WebRTC configuration
            rtc_configuration = RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            })
            
            # Create video transformer
            video_transformer = LiveObjectDetection(
                model=app.model,
                class_names=app.class_names,
                class_colors=app.class_colors,
                conf_threshold=conf_threshold
            )
            
            # WebRTC streamer
            webrtc_ctx = webrtc_streamer(
                key="live-detection",
                video_transformer_factory=lambda: video_transformer,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            # Live detection display
            if webrtc_ctx.state.playing:
                st.success("🎥 Live webcam detection is active!")
                
                # Create columns for live display
                live_col1, live_col2 = st.columns(2)
                
                with live_col1:
                    st.subheader("🎯 Current Detections")
                    detection_display = st.empty()
                    
                    # Display current detections
                    try:
                        while not video_transformer.detection_queue.empty():
                            detection = video_transformer.detection_queue.get_nowait()
                            st.session_state.detection_history.append(detection)
                            
                            # Keep only last 50 detections
                            if len(st.session_state.detection_history) > 50:
                                st.session_state.detection_history = st.session_state.detection_history[-50:]
                    except:
                        pass
                    
                    # Display recent detections
                    if st.session_state.detection_history:
                        recent_detections = st.session_state.detection_history[-10:]  # Show last 10
                        for i, detection in enumerate(reversed(recent_detections)):
                            time_diff = time.time() - detection['timestamp']
                            if time_diff < 5:  # Only show detections from last 5 seconds
                                st.markdown(f"""
                                **{detection['class']}** 
                                - Confidence: {detection['confidence']:.2f}
                                - Time: {time_diff:.1f}s ago
                                ---
                                """)
                
                with live_col2:
                    st.subheader("📈 Detection Statistics")
                    
                    # Create live statistics
                    if st.session_state.detection_history:
                        # Count detections by class
                        class_counts = {}
                        for detection in st.session_state.detection_history:
                            class_name = detection['class']
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        
                        # Display counts
                        for class_name, count in class_counts.items():
                            color = app.class_colors.get(class_name, "#808080")
                            st.markdown(f"""
                            <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 5px 0;">
                                <strong>{class_name}:</strong> {count} detections
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Create live chart
                        if class_counts:
                            fig = go.Figure(data=[go.Pie(
                                labels=list(class_counts.keys()),
                                values=list(class_counts.values()),
                                hole=0.3,
                                marker_colors=[app.class_colors.get(cls, '#808080') for cls in class_counts.keys()]
                            )])
                            
                            fig.update_layout(
                                title="Live Detection Distribution",
                                showlegend=True,
                                height=300
                            )
                            
                            chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Download live session data
                st.subheader("📥 Download Live Session Data")
                if st.session_state.detection_history:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download detection log
                        detection_log = []
                        for detection in st.session_state.detection_history:
                            detection_log.append({
                                'Timestamp': datetime.fromtimestamp(detection['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                                'Class': detection['class'],
                                'Confidence': detection['confidence'],
                                'Bounding Box': detection['bbox']
                            })
                        
                        df = pd.DataFrame(detection_log)
                        csv_buffer = io.BytesIO()
                        df.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)
                        
                        st.download_button(
                            label="📊 Download Detection Log (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name=f"live_detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Download session report
                        session_report = f"""
# Live Webcam Detection Session Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Session Information
- Detection confidence threshold: {conf_threshold}
- Total detections: {len(st.session_state.detection_history)}
- Session duration: {time.time() - min([d['timestamp'] for d in st.session_state.detection_history]):.1f} seconds

## Detection Summary
"""
                        for class_name in app.class_names:
                            count = sum(1 for d in st.session_state.detection_history if d['class'] == class_name)
                            session_report += f"- {class_name}: {count} detections\n"
                        
                        session_report += "\n## Detailed Detection Log\n"
                        for i, detection in enumerate(st.session_state.detection_history):
                            session_report += f"\n### Detection {i+1}\n"
                            session_report += f"- Time: {datetime.fromtimestamp(detection['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}\n"
                            session_report += f"- Class: {detection['class']}\n"
                            session_report += f"- Confidence: {detection['confidence']:.3f}\n"
                            session_report += f"- Bounding Box: {detection['bbox']}\n"
                        
                        report_buffer = io.BytesIO()
                        report_buffer.write(session_report.encode())
                        report_buffer.seek(0)
                        
                        st.download_button(
                            label="📄 Download Session Report (MD)",
                            data=report_buffer.getvalue(),
                            file_name=f"live_session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
            else:
                st.info("🎥 Click 'Start Webcam' to begin live object detection")
                
                # Show supported objects
                st.subheader("🎯 Supported Objects")
                obj_col1, obj_col2, obj_col3 = st.columns(3)
                
                with obj_col1:
                    st.markdown("""
                    <div style="text-align: center; padding: 20px; background-color: #E56626; border-radius: 10px; color: white;">
                        <h3>🔧 Toolbox</h3>
                        <p>Detects various types of toolboxes</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with obj_col2:
                    st.markdown("""
                    <div style="text-align: center; padding: 20px; background-color: #FF0000; border-radius: 10px; color: white;">
                        <h3>🧯 Fire Extinguisher</h3>
                        <p>Detects fire extinguishers</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with obj_col3:
                    st.markdown("""
                    <div style="text-align: center; padding: 20px; background-color: #D4CEC1; border-radius: 10px; color: black;">
                        <h3>🫧 Oxygen Tank</h3>
                        <p>Detects oxygen tanks</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab3:
        st.header("Analytics Dashboard")
        
        # Model information
        st.subheader("Model Information")
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.metric("Classes", len(app.class_names))
        
        with info_col2:
            st.metric("Model Path", model_path.split('/')[-1])
        
        with info_col3:
            st.metric("Confidence Threshold", f"{conf_threshold:.2f}")
        
        # Class information
        st.subheader("Detection Classes")
        class_df = pd.DataFrame([
            {
                'Class': name,
                'Color': color,
                'Description': f"Detects {name.lower()} in space station environments"
            }
            for name, color in app.class_colors.items()
        ])
        st.dataframe(class_df, use_container_width=True)
        
        # Performance metrics
        st.subheader("Performance Metrics")
        
        # Check for performance analysis files
        performance_files = {
            'confusion_matrix.png': 'Confusion Matrix',
            'performance_metrics.png': 'Performance Metrics',
            'training_curves.png': 'Training Curves',
            'performance_dashboard.html': 'Interactive Dashboard'
        }
        
        for file_path, title in performance_files.items():
            if os.path.exists(file_path):
                if file_path.endswith('.html'):
                    # Display interactive dashboard
                    with open(file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    st.subheader(title)
                    st.components.v1.html(html_content, height=800)
                else:
                    # Display image
                    st.subheader(title)
                    st.image(file_path, caption=title, use_container_width=True)
            else:
                st.info(f"{title} not available. Run 'python performance_analysis.py' to generate.")
        
        # Performance report
        if os.path.exists('performance_report.md'):
            st.subheader("Performance Report")
            with open('performance_report.md', 'r', encoding='utf-8') as f:
                report_content = f.read()
            st.markdown(report_content)
        
        # Download buttons for individual files
        st.subheader("📥 Download Individual Files")
        available_files = app.get_performance_files()
        
        if available_files:
            cols = st.columns(3)
            for i, (file_path, label) in enumerate(available_files):
                col_idx = i % 3
                with cols[col_idx]:
                    if file_path.endswith('.png'):
                        with open(file_path, "rb") as file:
                            st.download_button(
                                label=f"📊 {label}",
                                data=file.read(),
                                file_name=os.path.basename(file_path),
                                mime="image/png"
                            )
                    elif file_path.endswith('.html'):
                        with open(file_path, "rb") as file:
                            st.download_button(
                                label=f"📊 {label}",
                                data=file.read(),
                                file_name=os.path.basename(file_path),
                                mime="text/html"
                            )
                    elif file_path.endswith('.md'):
                        with open(file_path, "rb") as file:
                            st.download_button(
                                label=f"📊 {label}",
                                data=file.read(),
                                file_name=os.path.basename(file_path),
                                mime="text/markdown"
                            )
        else:
            st.info("No performance files available. Run 'python performance_analysis.py' to generate reports.")
    
    with tab4:
        st.header("📥 Downloads Center")
        st.markdown("Download reports, models, and analysis files")
        
        # Download all performance files as ZIP
        st.subheader("📦 Download All Reports")
        available_files = app.get_performance_files()
        
        if available_files:
            file_paths = [file_path for file_path, _ in available_files]
            zip_buffer = app.create_download_zip(file_paths)
            
            st.download_button(
                label="📦 Download All Reports (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"performance_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
            
            st.markdown("**Included files:**")
            for file_path, label in available_files:
                st.markdown(f"• {label} (`{os.path.basename(file_path)}`)")
        else:
            st.info("No performance files available. Run 'python performance_analysis.py' to generate reports.")
        
        # Model downloads
        st.subheader("🤖 Download Models")
        model_files = [
            ("runs/train/simple_precision_boost/weights/best.pt", "Best Precision Model"),
            ("runs/detect/train/weights/best.pt", "Training Run 1 Best"),
            ("runs/detect/train2/weights/best.pt", "Training Run 2 Best")
        ]
        
        model_cols = st.columns(3)
        for i, (model_path, label) in enumerate(model_files):
            if os.path.exists(model_path):
                with model_cols[i]:
                    with open(model_path, "rb") as file:
                        st.download_button(
                            label=f"🤖 {label}",
                            data=file.read(),
                            file_name=os.path.basename(model_path),
                            mime="application/octet-stream"
                        )
            else:
                with model_cols[i]:
                    st.info(f"{label} not available")
        
        # Configuration files
        st.subheader("⚙️ Download Configuration")
        config_files = [
            ("config.yaml", "Main Configuration"),
            ("config_improved.yaml", "Improved Configuration"),
            ("config_high_precision.yaml", "High Precision Config"),
            ("yolo_params.yaml", "YOLO Parameters")
        ]
        
        config_cols = st.columns(2)
        for i, (config_path, label) in enumerate(config_files):
            if os.path.exists(config_path):
                with config_cols[i % 2]:
                    with open(config_path, "rb") as file:
                        st.download_button(
                            label=f"⚙️ {label}",
                            data=file.read(),
                            file_name=os.path.basename(config_path),
                            mime="text/yaml"
                        )
            else:
                with config_cols[i % 2]:
                    st.info(f"{label} not available")
        
        # Training scripts
        st.subheader("📝 Download Scripts")
        script_files = [
            ("train.py", "Training Script"),
            ("train_improved.py", "Improved Training"),
            ("train_high_precision.py", "High Precision Training"),
            ("performance_analysis.py", "Performance Analysis"),
            ("fix_detection_issues.py", "Detection Fixer")
        ]
        
        script_cols = st.columns(2)
        for i, (script_path, label) in enumerate(script_files):
            if os.path.exists(script_path):
                with script_cols[i % 2]:
                    with open(script_path, "rb") as file:
                        st.download_button(
                            label=f"📝 {label}",
                            data=file.read(),
                            file_name=os.path.basename(script_path),
                            mime="text/plain"
                        )
            else:
                with script_cols[i % 2]:
                    st.info(f"{label} not available")
        
        # Complete project download
        st.subheader("📦 Download Complete Project")
        st.markdown("""
        **Note:** This will include all project files including:
        - Models and weights
        - Configuration files
        - Training scripts
        - Performance reports
        - Documentation
        """)
        
        # Create a comprehensive project ZIP
        project_files = []
        
        # Add all available files
        for file_path, _ in available_files:
            project_files.append(file_path)
        
        for model_path, _ in model_files:
            if os.path.exists(model_path):
                project_files.append(model_path)
        
        for config_path, _ in config_files:
            if os.path.exists(config_path):
                project_files.append(config_path)
        
        for script_path, _ in script_files:
            if os.path.exists(script_path):
                project_files.append(script_path)
        
        # Add additional important files
        additional_files = [
            "README.md", "requirements.txt", "environment.yml",
            "classes.txt", "setup.py", "quick_start.py"
        ]
        
        for file_path in additional_files:
            if os.path.exists(file_path):
                project_files.append(file_path)
        
        if project_files:
            project_zip = app.create_download_zip(project_files)
            
            st.download_button(
                label="📦 Download Complete Project (ZIP)",
                data=project_zip.getvalue(),
                file_name=f"space_station_detection_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
            
            st.markdown(f"**Total files included:** {len(project_files)}")
        else:
            st.info("No project files available for download.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>Space Station Object Detection System | Built with YOLOv8 and Streamlit</p>
        <p>Detecting Toolbox, Oxygen Tank, and Fire Extinguisher in space environments</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()