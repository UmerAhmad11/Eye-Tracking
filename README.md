Eye Tracker - Syed Umer Ahmad - TAMUQ
Real-time gaze tracker using MediaPipe FaceMesh with exact eyelid mesh extraction for high accuracy.
Tracks gaze direction (left, right, up, down, center) with smooth cursor control.

Features
Real-time eye tracking with smooth EMA filtering

Exact eyelid boundary detection (not bounding boxes)

Resolution and FPS selection via GUI

Performance metrics: FPS, reaction time, movement range

No manual calibration (currently tracks using normalized gaze)

Technologies
Python 3

OpenCV

MediaPipe

Tkinter

PyAutoGUI

Matplotlib

NumPy

How to Run
bash
Copy
Edit
pip install opencv-python mediapipe pyautogui numpy matplotlib
python eye_tracker.py
Select resolution and FPS → Start tracking.

Press k to save results, q to quit.

