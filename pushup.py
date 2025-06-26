import tkinter as tk
from tkinter import Label
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageSequence
import mediapipe as mp
from utils import detection_body_part, calculate_angle, score_table
from PushIt import TypeOfExercise  # Assuming BodyPartAngle and TypeOfExercise are defined in PushIt.py
import time
import sys

# Tkinter setup
root = tk.Tk()
root.title("Push-Up Counter")
root.attributes('-fullscreen', True)

video_label = Label(root)
video_label.pack(side="left", padx=10, pady=10)

gif_label = Label(root)
gif_label.pack(side="right", padx=10, pady=10)

# Load GIF
gif_path = "pushup.gif"  # Make sure this file exists
gif = Image.open(gif_path)
frames = [ImageTk.PhotoImage(frame.copy().convert("RGBA")) for frame in ImageSequence.Iterator(gif)]
frame_count = len(frames)
gif_index = 0

def update_gif_frame():
    global gif_index
    gif_label.configure(image=frames[gif_index])
    gif_index = (gif_index + 1) % frame_count
    gif_label.after(100, update_gif_frame)

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture("Push-up (side view).mp4")#"Push-up (side view).mp4"


try:
    sets = int(sys.argv[1])
    reps_per_set = int(sys.argv[2])
    rest_time = int(sys.argv[3])
except (IndexError, ValueError):
    # Fallback default values if arguments are not passed correctly
    sets = 2
    reps_per_set = 5
    rest_time = 5
# Exercise state
counter = 0
status = True
# sets = 2
# reps_per_set = 5
current_set = 1
state = "exercise"  # or "rest"
# rest_time = 5
rest_remaining = rest_time
last_rest_time = 0

def update_frame():
    global counter, status, current_set, state, rest_remaining, last_rest_time

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
            counter, status = TypeOfExercise(landmarks).calculate_exercise("push-up", counter, status)

        # Draw counter
        cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)
        cv2.putText(image, f"Set {current_set}/{sets}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, f"Reps: {counter}/{reps_per_set}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if counter >= reps_per_set:
            counter = 0
            status = True
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
