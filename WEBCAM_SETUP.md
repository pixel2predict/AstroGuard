# Webcam Setup Guide

## Overview
The Space Station Object Detection app now includes live webcam functionality for real-time detection of:
- 🔧 **Toolbox**
- 🧯 **Fire Extinguisher** 
- 🫧 **Oxygen Tank**

## Prerequisites

### 1. Install Dependencies
```bash
pip install streamlit-webrtc av
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. System Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Webcam**: Any USB webcam or built-in camera
- **Browser**: Chrome, Firefox, Safari, or Edge (with WebRTC support)
- **Internet**: Required for WebRTC connection

## Usage Instructions

### 1. Start the Application
```bash
streamlit run app.py
```

### 2. Load a Model
1. In the sidebar, select a trained model from the dropdown
2. Click "Load Model" button
3. Wait for the success message

### 3. Access Live Webcam
1. Click on the "📹 Live Webcam" tab
2. Click "Start Webcam" to begin live detection
3. Point your camera at objects to detect
4. View real-time detection results and statistics

## Features

### Live Detection
- Real-time object detection with bounding boxes
- Confidence scores displayed on each detection
- Color-coded detection boxes for each object type

### Live Statistics
- Real-time detection counts by object type
- Interactive pie chart showing detection distribution
- Recent detection history (last 10 detections)

### Session Management
- Download detection logs as CSV
- Generate session reports in Markdown format
- Track detection timestamps and confidence scores

## Troubleshooting

### Webcam Not Working
1. **Check Browser Permissions**: Allow camera access when prompted
2. **Try Different Browser**: Some browsers have better WebRTC support
3. **Check Webcam**: Ensure webcam is not being used by another application
4. **Test Basic Webcam**: Run `test_webcam.py` to verify basic functionality

### Performance Issues
1. **Lower Detection FPS**: Reduce the detection FPS slider in settings
2. **Close Other Applications**: Free up system resources
3. **Check Model Size**: Use smaller models for better performance

### Connection Issues
1. **Check Internet**: WebRTC requires internet connection for STUN servers
2. **Firewall**: Ensure firewall allows WebRTC connections
3. **Corporate Network**: Some corporate networks block WebRTC

## Model Information

### Supported Objects
- **Toolbox** (Class 0): Various types of toolboxes
- **Fire Extinguisher** (Class 2): Fire extinguishers
- **Oxygen Tank** (Class 1): Oxygen tanks

### Model Files
- `runs/detect/train/weights/best.pt` - Best model from training run 1
- `runs/detect/train2/weights/best.pt` - Best model from training run 2

## Advanced Configuration

### Detection Settings
- **Confidence Threshold**: Adjust minimum confidence for detections (0.1-0.9)
- **Detection FPS**: Control how often detection runs (1-30 FPS)
- **Show Confidence**: Toggle confidence score display

### Performance Optimization
- Detection runs every 3 frames by default for performance
- Queue system prevents memory overflow
- Automatic cleanup of old detection history

## Security Notes
- Webcam access requires user permission
- No video data is stored permanently
- All processing happens locally in the browser
- Detection results can be downloaded for analysis

## Support
If you encounter issues:
1. Check the troubleshooting section above
2. Run the test script: `streamlit run test_webcam.py`
3. Verify all dependencies are installed correctly
4. Check browser console for error messages 