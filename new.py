#Eye Tracker - Syed Umer Ahmad - TAMUQ


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
    "top_left": None,
    "top_right": None,
    "bottom_left": None,
    "bottom_right": None,
    "center": None
}
calibration_done = False
calibration_index = 0
calibration_order = ["top_left", "top_right", "bottom_left", "bottom_right", "center"]

def piecewise_interpolate(gaze_x, gaze_y):
    """
    Gaze_x and Gaze_y are between 0 and 1.
    We interpolate based on the 5 calibration points.
    """
    if not calibration_done:
        return interpolate(gaze_x, 0, 1, 0, screen_w), interpolate(gaze_y, 0, 1, 0, screen_h)

    # Extract calibration screen positions
    tl = calibration_data['top_left']
    tr = calibration_data['top_right']
    bl = calibration_data['bottom_left']
    br = calibration_data['bottom_right']
    center = calibration_data['center']

    # Determine which quadrant we are closer to
    if gaze_x <= 0.5 and gaze_y <= 0.5:
        # Top Left Quadrant
        screen_x = interpolate(gaze_x, 0, 0.5, tl[0], center[0])
        screen_y = interpolate(gaze_y, 0, 0.5, tl[1], center[1])
    elif gaze_x > 0.5 and gaze_y <= 0.5:
        # Top Right Quadrant
        screen_x = interpolate(gaze_x, 0.5, 1, center[0], tr[0])
        screen_y = interpolate(gaze_y, 0, 0.5, tr[1], center[1])
    elif gaze_x <= 0.5 and gaze_y > 0.5:
        # Bottom Left Quadrant
        screen_x = interpolate(gaze_x, 0, 0.5, bl[0], center[0])
        screen_y = interpolate(gaze_y, 0.5, 1, center[1], bl[1])
    else:
        # Bottom Right Quadrant
        screen_x = interpolate(gaze_x, 0.5, 1, center[0], br[0])
        screen_y = interpolate(gaze_y, 0.5, 1, center[1], br[1])

    return screen_x, screen_y


# Gaze Smoothing
smoothed_gaze_x = None
smoothed_gaze_y = None
ema_alpha = 0.2  # Smoothing factor (0.1-0.3 good)

# Smoothing
smooth_factor = 0.08
y_smoothing_factor = 0.08

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

# Reaction Time and Range Tracking
reaction_times = []
last_gaze_time = None
last_gaze_x, last_gaze_y = None, None

movement_min_x = float('inf')
movement_max_x = float('-inf')
movement_min_y = float('inf')
movement_max_y = float('-inf')

# ======== FUNCTIONS ========

def get_gaze_ratios(landmarks, w, h):
    # Eye Openness
    l_open = abs(landmarks[LEFT_EYE_TOP].y - landmarks[LEFT_EYE_BOTTOM].y) * h
    r_open = abs(landmarks[RIGHT_EYE_TOP].y - landmarks[RIGHT_EYE_BOTTOM].y) * h
    avg_open = (l_open + r_open) / 2

    # Safer range
    safe_min_open = 16   # Eyes partially closed
    safe_max_open = 22   # Eyes fully open

    # Normalize
    openness_normalized = (avg_open - safe_min_open) / (safe_max_open - safe_min_open + 1e-6)
    openness_normalized = np.clip(openness_normalized, 0, 1)

    # Invert logic
    gaze_y = 1 - openness_normalized

    # Horizontal Gaze
    nose_x = int(landmarks[NOSE_BRIDGE].x * w)
    lp_x = int(landmarks[LEFT_PUPIL].x * w)
    rp_x = int(landmarks[RIGHT_PUPIL].x * w)
    eye_center_x = (lp_x + rp_x) / 2
    lx = int(landmarks[LEFT_EYE_LEFT].x * w)
    rx = int(landmarks[LEFT_EYE_RIGHT].x * w)

    gaze_x = (eye_center_x - nose_x) / (rx - lx + 1e-6)
    gaze_x = np.clip(0.5 + gaze_x, 0, 1)

    return gaze_x, gaze_y

def interpolate(val, min_val, max_val, screen_min, screen_max):
    return np.clip(screen_min + (val - min_val) / (max_val - min_val + 1e-6) * (screen_max - screen_min), screen_min, screen_max)

def start_tracking():
    global cap, running, saving, last_gaze_time, last_gaze_x, last_gaze_y
    global movement_min_x, movement_max_x, movement_min_y, movement_max_y, reaction_times

    # Reset tracking variables
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
    direction_log = [] 

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
                # Correct Mesh Indices
                LEFT_EYE_MESH = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
                RIGHT_EYE_MESH = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]

                def get_eye_mask(frame_shape, landmarks, indices, frame_w, frame_h):
                    eye_points = np.array([
                        (int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in indices
                    ], np.int32)

                    mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
                    cv2.fillPoly(mask, [eye_points], 255)
                    return mask, eye_points

                # Generate precise eyelid masks
                left_mask, left_poly = get_eye_mask(frame.shape, face_landmarks.landmark, LEFT_EYE_MESH, frame_w, frame_h)
                right_mask, right_poly = get_eye_mask(frame.shape, face_landmarks.landmark, RIGHT_EYE_MESH, frame_w, frame_h)

                # Draw Eyelid Mesh Outlines
                cv2.polylines(frame, [left_poly], isClosed=True, color=(0, 255, 0), thickness=1)
                cv2.polylines(frame, [right_poly], isClosed=True, color=(0, 255, 0), thickness=1)

                # Optional: Mask and Crop (for precision use)
                left_eye_exact = cv2.bitwise_and(rgb_frame, rgb_frame, mask=left_mask)
                right_eye_exact = cv2.bitwise_and(rgb_frame, rgb_frame, mask=right_mask)

                # Cropping exact region (optional for later use)
                lx, ly, lw, lh = cv2.boundingRect(left_poly)
                rx, ry, rw, rh = cv2.boundingRect(right_poly)

                left_eye_crop = left_eye_exact[ly:ly+lh, lx:lx+lw]
                right_eye_crop = right_eye_exact[ry:ry+rh, rx:rx+rw]

                # Display for verification (optional)
                cv2.imshow("Exact Left Eye", left_eye_crop)
                cv2.imshow("Exact Right Eye", right_eye_crop)


                # Gaze Processing
                gaze_x, gaze_y = get_gaze_ratios(face_landmarks.landmark, frame_w, frame_h)

                # Smoothing Gaze
                global smoothed_gaze_x, smoothed_gaze_y

                if smoothed_gaze_x is None:
                    smoothed_gaze_x = gaze_x
                    smoothed_gaze_y = gaze_y
                else:
                    smoothed_gaze_x = (1 - ema_alpha) * smoothed_gaze_x + ema_alpha * gaze_x
                    smoothed_gaze_y = (1 - ema_alpha) * smoothed_gaze_y + ema_alpha * gaze_y


                # Reaction Time Tracking
                gaze_change_threshold = 0.05
                if last_gaze_x is not None and last_gaze_y is not None:
                    dx = abs(gaze_x - last_gaze_x)
                    dy = abs(gaze_y - last_gaze_y)
                    if dx > gaze_change_threshold or dy > gaze_change_threshold:
                        if last_gaze_time is not None:
                            reaction_times.append(current_time - last_gaze_time)
                        last_gaze_time = current_time
                else:
                    last_gaze_time = current_time

                last_gaze_x, last_gaze_y = gaze_x, gaze_y

                # Cursor Move
                target_x, target_y = piecewise_interpolate(smoothed_gaze_x, smoothed_gaze_y)



                curr_x = prev_x + (target_x - prev_x) * smooth_factor
                curr_y = prev_y + (target_y - prev_y) * y_smoothing_factor

                curr_x = np.clip(curr_x, 0, screen_w - 1)
                curr_y = np.clip(curr_y, 0, screen_h - 1)

                pyautogui.moveTo(int(curr_x), int(curr_y))

                movement_min_x = min(movement_min_x, curr_x)
                movement_max_x = max(movement_max_x, curr_x)
                movement_min_y = min(movement_min_y, curr_y)
                movement_max_y = max(movement_max_y, curr_y)

                # Performance Calculation
                movement_scale = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
                performance_score = min(100, (fps / (1 + movement_scale)) * 10)


                local_fps.append(fps)
                local_performance.append(performance_score)

                prev_x, prev_y = curr_x, curr_y

                # Arrow Drawing
                center_x = frame.shape[1] // 2
                center_y = frame.shape[0] // 2

                # --- Vertical Movement (Up/Down)
                if gaze_y < 0.4:
                    cv2.arrowedLine(frame, (center_x, center_y + 50), (center_x, center_y - 50), (0, 255, 0), 5, tipLength=0.5)
                    cv2.putText(frame, 'Looking Up', (center_x - 70, center_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    direction_log.append(("up", curr_y))
                elif gaze_y > 0.65:
                    cv2.arrowedLine(frame, (center_x, center_y - 50), (center_x, center_y + 50), (0, 0, 255), 5, tipLength=0.5)
                    cv2.putText(frame, 'Looking Down', (center_x - 90, center_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    direction_log.append(("down", curr_y))
                # --- Horizontal Movement (Left/Right)
                if gaze_x < 0.45:
                    cv2.arrowedLine(frame, (center_x + 50, center_y), (center_x - 50, center_y), (255, 0, 0), 5, tipLength=0.5)
                    cv2.putText(frame, 'Looking Left', (center_x - 90, center_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    direction_log.append(("left", curr_x))
                elif gaze_x > 0.65:
                    cv2.arrowedLine(frame, (center_x - 50, center_y), (center_x + 50, center_y), (255, 0, 0), 5, tipLength=0.5)
                    cv2.putText(frame, 'Looking Right', (center_x - 90, center_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    direction_log.append(("right", curr_x))

                # --- Center (Neutral)
                if 0.4 <= gaze_x <= 0.65 and 0.4 <= gaze_y <= 0.65:
                    cv2.putText(frame, 'Center', (center_x - 40, center_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


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


    # Analyze average position per direction
    up_positions = [pos for dir, pos in direction_log if dir == "up"]
    down_positions = [pos for dir, pos in direction_log if dir == "down"]
    left_positions = [pos for dir, pos in direction_log if dir == "left"]
    right_positions = [pos for dir, pos in direction_log if dir == "right"]

    def compute_accuracy(positions, expected_area, axis_max):
        if not positions:
            return 0
        score = np.mean([1 - abs(p - expected_area) / axis_max for p in positions])
        return max(0, min(100, score * 100))

    screen_h_half = screen_h / 2
    screen_w_half = screen_w / 2

    accuracy_up = compute_accuracy(up_positions, 0, screen_h)  # top
    accuracy_down = compute_accuracy(down_positions, screen_h, screen_h)  # bottom
    accuracy_left = compute_accuracy(left_positions, 0, screen_w)  # left
    accuracy_right = compute_accuracy(right_positions, screen_w, screen_w)  # right

    avg_accuracy = np.mean([accuracy_up, accuracy_down, accuracy_left, accuracy_right])

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
            'performance': avg_accuracy,
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

    # Labels for x-axis = resolution + fps
    labels = resolution_list

    plt.figure(figsize=(16, 10))

    # Plot 1: FPS vs Classic Performance
    plt.subplot(2, 2, 1)
    plt.scatter(fps_values, perf_values, color='blue')
    plt.xlabel("Average FPS")
    plt.ylabel("Performance Score")
    plt.title("Performance vs FPS")
    plt.grid(True)

    # Plot 2: Reaction Time
    plt.subplot(2, 2, 2)
    plt.plot(labels, reaction_values, marker='o', color='red')
    plt.xlabel("Resolution + FPS")
    plt.ylabel("Average Reaction Time (s)")
    plt.title("Reaction Time per Test")
    plt.xticks(rotation=45)
    plt.grid(True)

    # Plot 3: Horizontal Range
    plt.subplot(2, 2, 3)
    plt.plot(labels, horiz_range_values, marker='x', color='green')
    plt.xlabel("Resolution + FPS")
    plt.ylabel("Horizontal Movement Range (pixels)")
    plt.title("Horizontal Control Range")
    plt.xticks(rotation=45)
    plt.grid(True)

    # Plot 4: Vertical Range
    plt.subplot(2, 2, 4)
    plt.plot(labels, vert_range_values, marker='x', color='purple')
    plt.xlabel("Resolution + FPS")
    plt.ylabel("Vertical Movement Range (pixels)")
    plt.title("Vertical Control Range")
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ======== TKINTER GUI ========
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
