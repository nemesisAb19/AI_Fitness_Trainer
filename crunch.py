import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk, ImageSequence
import cv2
import numpy as np
import time
import mediapipe as mp
import pyttsx3
from utils import *
from crunch2 import TypeOfExercise
import sys

# --- Text-to-Speech ---
engine = pyttsx3.init()
engine.setProperty('rate', 150)
def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

# --- MediaPipe Setup ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Tkinter GUI Setup ---
root = tk.Tk()
root.title("Crunch Counter")
root.attributes('-fullscreen', True)

video_label = Label(root)
video_label.pack(side="left", padx=10, pady=10)

gif_label = Label(root)
gif_label.pack(side="right", padx=10, pady=10)

# --- Load GIF ---
gif = Image.open("situps.gif")
gif_frames = [ImageTk.PhotoImage(frame.copy().convert("RGBA")) for frame in ImageSequence.Iterator(gif)]
frame_count = len(gif_frames)
gif_index = 0

# --- Load Video ---
cap = cv2.VideoCapture("sit-up.mp4")  # or use cv2.VideoCapture(0) for webcam "sit-up.mp4"
cap.set(3, 800)
cap.set(4, 480)

try:
    sets = int(sys.argv[1])
    reps_per_set = int(sys.argv[2])
    rest_time = int(sys.argv[3])
except (IndexError, ValueError):
    # Fallback default values if arguments are not passed correctly
    sets = 2
    reps_per_set = 5
    rest_time = 5

# --- Exercise State ---
counter = 0
status = True
# sets = 2
# reps_per_set = 5
current_set = 1
state = "exercise"
# rest_time = 5
rest_remaining = rest_time
last_rest_time = time.time()

# --- Frame Update ---
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
            try:
                landmarks = results.pose_landmarks.landmark
                counter_before = counter
                counter, status = TypeOfExercise(landmarks).calculate_exercise("sit-up", counter, status)

                if counter > counter_before:
                    text_to_speech(str(counter))  # Speak rep count

                score_table("sit-up", counter, status)
            except Exception as e:
                print("Error in landmark processing:", e)

        # Draw info
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
    # elif state == "rest":
    #     frame[:] = 0
    #     cv2.putText(frame, f"Rest: {rest_remaining}s", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
    #     rest_remaining -= 1
    #     if rest_remaining <= 0:
    #         state = "exercise"

    # Draw landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(1000 if state == "rest" else 10, update_frame)

# --- GIF Loop ---
def update_gif():
    global gif_index
    gif_label.configure(image=gif_frames[gif_index])
    gif_index = (gif_index + 1) % frame_count
    gif_label.after(100, update_gif)

# --- STOP Button ---
stop_btn = tk.Button(root, text="STOP", command=lambda: (cap.release(), root.destroy()),
                     bg="red", fg="white", font=("Helvetica", 14, "bold"),
                     relief="raised", width=10, height=2)
stop_btn.place(x=750, y=700)

# --- MediaPipe Model ---
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Start Loops ---
update_frame()
update_gif()
root.mainloop()
