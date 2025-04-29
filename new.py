# Advanced Eye Tracker - Syed Umer Ahmad - TAMUQ

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import time

# ======== GLOBAL SETUP ========
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

screen_w, screen_h = pyautogui.size()

cap = None
running = False
saving = False

# Data storage
fps_list = []
performance_list = []
resolution_list = []

# Calibration
calibration_data = {
    "top_left": [],
    "top_right": [],
    "bottom_left": [],
    "bottom_right": [],
    "center": []
}
calibrated_map = {}
calibration_order = list(calibration_data.keys())
calibration_index = 0
calibration_done = False

# Smoothing
alpha_x = 0.1
alpha_y = 0.1

# Landmark indices
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_PUPIL = 468
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_PUPIL = 473
NOSE_BRIDGE = 168

# Reaction Time
reaction_times = []
last_gaze_time = None
last_gaze_x, last_gaze_y = None, None
movement_min_x = float('inf')
movement_max_x = float('-inf')
movement_min_y = float('inf')
movement_max_y = float('-inf')

# ======== FUNCTIONS ========
def collect_calibration_point(gaze_x, gaze_y):
    global calibration_index, calibration_done
    if calibration_index < len(calibration_order):
        key = calibration_order[calibration_index]
        calibration_data[key].append((gaze_x, gaze_y))
        if len(calibration_data[key]) >= 20:
            calibration_index += 1
        if calibration_index >= len(calibration_order):
            finalize_calibration()

def finalize_calibration():
    global calibrated_map, calibration_done
    for key, samples in calibration_data.items():
        avg_x = np.mean([x for x, _ in samples])
        avg_y = np.mean([y for _, y in samples])
        calibrated_map[key] = (avg_x, avg_y)
    calibration_done = True
    print("[Calibration] Completed.")

def map_calibrated_gaze(gaze_x, gaze_y):
    cx, cy = calibrated_map["center"]
    dx = gaze_x - cx
    dy = gaze_y - cy
    if dx < 0 and dy < 0:
        return "top_left"
    elif dx > 0 and dy < 0:
        return "top_right"
    elif dx < 0 and dy > 0:
        return "bottom_left"
    elif dx > 0 and dy > 0:
        return "bottom_right"
    else:
        return "center"

def get_pupil_relative_y(landmarks, eye_top, eye_bottom, pupil_idx, h):
    top_y = landmarks[eye_top].y * h
    bottom_y = landmarks[eye_bottom].y * h
    pupil_y = landmarks[pupil_idx].y * h
    return (pupil_y - top_y) / (bottom_y - top_y + 1e-6)

def get_gaze_ratios_enhanced(landmarks, w, h):
    nose_x = int(landmarks[NOSE_BRIDGE].x * w)
    lp_x = int(landmarks[LEFT_PUPIL].x * w)
    rp_x = int(landmarks[RIGHT_PUPIL].x * w)
    eye_center_x = (lp_x + rp_x) / 2
    lx = int(landmarks[LEFT_EYE_LEFT].x * w)
    rx = int(landmarks[LEFT_EYE_RIGHT].x * w)
    gaze_x = (eye_center_x - nose_x) / (rx - lx + 1e-6)
    gaze_x = np.clip(0.5 + gaze_x, 0, 1)
    l_rel_y = get_pupil_relative_y(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_PUPIL, h)
    r_rel_y = get_pupil_relative_y(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_PUPIL, h)
    gaze_y = (l_rel_y + r_rel_y) / 2
    gaze_y = np.clip(gaze_y, 0, 1)
    return gaze_x, gaze_y

def interpolate(val, min_val, max_val, screen_min, screen_max):
    return np.clip(screen_min + (val - min_val) / (max_val - min_val + 1e-6) * (screen_max - screen_min), screen_min, screen_max)

def start_tracking():
    global cap, running, saving, last_gaze_time, last_gaze_x, last_gaze_y
    global movement_min_x, movement_max_x, movement_min_y, movement_max_y, reaction_times

    last_gaze_time = None
    last_gaze_x, last_gaze_y = None, None
    movement_min_x = float('inf')
    movement_max_x = float('-inf')
    movement_min_y = float('inf')
    movement_max_y = float('-inf')
    reaction_times = []

    width, height = map(int, resolution_var.get().split('x'))
    fps_setting = int(fps_var.get())

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps_setting)

    last_time = time.time()
    local_fps = []
    local_performance = []
    prev_x, prev_y = 0, 0

    running = True
    saving = False

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        frame_h, frame_w, _ = frame.shape

        current_time = time.time()
        fps = 1 / (current_time - last_time + 1e-6)
        last_time = current_time

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                gaze_x, gaze_y = get_gaze_ratios_enhanced(face_landmarks.landmark, frame_w, frame_h)

                if not calibration_done:
                    collect_calibration_point(gaze_x, gaze_y)
                else:
                    zone = map_calibrated_gaze(gaze_x, gaze_y)
                    cv2.putText(frame, f"{zone}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if last_gaze_x is not None and last_gaze_y is not None:
                    dx = abs(gaze_x - last_gaze_x)
                    dy = abs(gaze_y - last_gaze_y)
                    if dx > 0.05 or dy > 0.05:
                        if last_gaze_time is not None:
                            reaction_times.append(current_time - last_gaze_time)
                        last_gaze_time = current_time
                else:
                    last_gaze_time = current_time

                last_gaze_x, last_gaze_y = gaze_x, gaze_y

                target_x = interpolate(gaze_x, 0, 1, 0, screen_w)
                target_y = interpolate(gaze_y, 0, 1, 0, screen_h)

                curr_x = alpha_x * target_x + (1 - alpha_x) * prev_x
                curr_y = alpha_y * target_y + (1 - alpha_y) * prev_y
                pyautogui.moveTo(int(curr_x), int(curr_y))

                movement_min_x = min(movement_min_x, curr_x)
                movement_max_x = max(movement_max_x, curr_x)
                movement_min_y = min(movement_min_y, curr_y)
                movement_max_y = max(movement_max_y, curr_y)

                movement_scale = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
                performance_score = fps / (1 + movement_scale)

                local_fps.append(fps)
                local_performance.append(performance_score)
                prev_x, prev_y = curr_x, curr_y

        cv2.imshow("Eye Tracker", frame)
        key = cv2.waitKey(1)
        if key == ord('k'):
            running = False
            saving = True
            break
        elif key == ord('q'):
            running = False
            saving = False
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()

    if saving:
        avg_fps = np.mean(local_fps)
        avg_performance = np.mean(local_performance)
        avg_reaction_time = np.mean(reaction_times) if reaction_times else 0
        horizontal_range = movement_max_x - movement_min_x
        vertical_range = movement_max_y - movement_min_y

        fps_list.append(avg_fps)
        resolution_list.append(f"{width}x{height} {fps_setting}fps")
        performance_list.append({
            'fps': avg_fps,
            'performance': avg_performance,
            'avg_reaction_time': avg_reaction_time,
            'horizontal_range': horizontal_range,
            'vertical_range': vertical_range
        })

def plot_results():
    if not performance_list:
        return

    fps_values = [p['fps'] for p in performance_list]
    perf_values = [p['performance'] for p in performance_list]
    reaction_values = [p['avg_reaction_time'] for p in performance_list]
    horiz_range_values = [p['horizontal_range'] for p in performance_list]
    vert_range_values = [p['vertical_range'] for p in performance_list]
    labels = resolution_list

    plt.figure(figsize=(16, 10))
    plt.subplot(2, 2, 1)
    plt.scatter(fps_values, perf_values, color='blue')
    plt.xlabel("Average FPS")
    plt.ylabel("Performance Score")
    plt.title("Performance vs FPS")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(labels, reaction_values, marker='o', color='red')
    plt.xlabel("Resolution + FPS")
    plt.ylabel("Average Reaction Time (s)")
    plt.title("Reaction Time per Test")
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(labels, horiz_range_values, marker='x', color='green')
    plt.xlabel("Resolution + FPS")
    plt.ylabel("Horizontal Movement Range (px)")
    plt.title("Horizontal Control Range")
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(labels, vert_range_values, marker='x', color='purple')
    plt.xlabel("Resolution + FPS")
    plt.ylabel("Vertical Movement Range (px)")
    plt.title("Vertical Control Range")
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ======== GUI ========
root = tk.Tk()
root.title("Gaze Tracker Performance Tester")
root.geometry('450x300')
root.configure(bg="#1e1e2f")

label = tk.Label(root, text="Gaze Tracker Pro", font=("Helvetica", 22), bg="#1e1e2f", fg="#00FFAA")
label.pack(pady=10)

res_label = tk.Label(root, text="Select Resolution:", bg="#1e1e2f", fg="white")
res_label.pack()
resolution_var = tk.StringVar()
resolution_options = ["640x480", "800x600", "1280x720", "1920x1080"]
resolution_dropdown = ttk.Combobox(root, textvariable=resolution_var, values=resolution_options)
resolution_dropdown.current(0)
resolution_dropdown.pack()

fps_label = tk.Label(root, text="Select FPS:", bg="#1e1e2f", fg="white")
fps_label.pack()
fps_var = tk.StringVar()
fps_options = ["15", "30", "60"]
fps_dropdown = ttk.Combobox(root, textvariable=fps_var, values=fps_options)
fps_dropdown.current(1)
fps_dropdown.pack()

start_button = tk.Button(root, text="Start Eye Tracking", command=start_tracking, bg="#00FFAA")
start_button.pack(pady=10)

plot_button = tk.Button(root, text="Plot Results", command=plot_results, bg="#00FFAA")
plot_button.pack(pady=10)

root.mainloop()
