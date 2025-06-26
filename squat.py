import tkinter as tk
from tkinter import Label
import cv2
import numpy as np
import time
from PIL import Image, ImageTk, ImageSequence
import mediapipe as mp
import sys

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Tkinter GUI setup
root = tk.Tk()
root.title("Squat Counter")
root.attributes('-fullscreen', True)

video_label = Label(root)
video_label.pack(side="left", padx=10, pady=10)

gif_label = Label(root)
gif_label.pack(side="right", padx=10, pady=10)

# Load squat GIF
gif_path = "squat.gif"
gif = Image.open(gif_path)
frames = [ImageTk.PhotoImage(frame.copy().convert("RGBA")) for frame in ImageSequence.Iterator(gif)]
frame_count = len(frames)
gif_index = 0

def update_gif_frame():
    global gif_index
    gif_label.configure(image=frames[gif_index])
    gif_index = (gif_index + 1) % frame_count
    gif_label.after(100, update_gif_frame)

cap = cv2.VideoCapture("squat.mp4")#"pull-up.mp4"
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


try:
    sets = int(sys.argv[1])
    reps_per_set = int(sys.argv[2])
    rest_time = int(sys.argv[3])
except (IndexError, ValueError):
    # Fallback default values if arguments are not passed correctly
    sets = 2
    reps_per_set = 5
    rest_time = 5
# Exercise state variables
counter = 0
stage = None
# sets = 2
# reps_per_set = 5
current_set = 1
state = "exercise"
# rest_time = 5
rest_remaining = rest_time
last_rest_time = time.time()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def update_frame():
    global counter, stage, current_set, state, rest_remaining, last_rest_time

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.resize(frame, (960, 500)) 
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if state == "exercise":
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            left_angle = calculate_angle(l_hip, l_knee, l_ankle)
            right_angle = calculate_angle(r_hip, r_knee, r_ankle)
            avg_angle = (left_angle + right_angle) / 2

            cv2.putText(image, f"{int(avg_angle)}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if avg_angle > 160:
                stage = "up"
            if avg_angle < 70 and stage == "up":
                stage = "down"
                counter += 1

        # Draw counter
        cv2.rectangle(image, (0, 0), (300, 100), (66, 135, 245), -1)
        cv2.putText(image, f"Set {current_set}/{sets}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, f"Reps: {counter}/{reps_per_set}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if counter >= reps_per_set:
            counter = 0
            stage = None
            if current_set < sets:
                current_set += 1
                state = "rest"
                rest_remaining = rest_time
                last_rest_time = time.time()
            else:
                cap.release()
                root.destroy()
                return

    elif state == "rest":
        elapsed = int(time.time() - last_rest_time)
        rest_remaining = rest_time - elapsed
        image[:] = 0
        cv2.putText(image, f"Rest: {rest_remaining}s", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        if rest_remaining <= 0:
            state = "exercise"

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    delay = 1000 if state == "rest" else 10
    root.after(delay, update_frame)

# STOP button
stop_btn = tk.Button(root, text="STOP", command=lambda: (cap.release(), root.destroy()),
                     bg="red", fg="white", font=("Helvetica", 14, "bold"),
                     relief="raised", width=10, height=2)
stop_btn.place(x=750, y=750)

update_frame()
update_gif_frame()
root.mainloop()
