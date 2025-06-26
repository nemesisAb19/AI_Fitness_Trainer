import tkinter as tk
from tkinter import Label
import cv2
import numpy as np
import time
from PIL import Image, ImageTk, ImageSequence
import subprocess

# Your posture module and utilities must be available
# import pm  # posture module
import PoseModule as pm
from ExercisesModule import utilities  # your helper functions

# ---- GUI Setup ----
root = tk.Tk()
root.title("Bicep Curls Tracker")
root.attributes('-fullscreen', True)

# Left: Video stream
video_label = Label(root)
video_label.pack(side="left", padx=10, pady=10)

# Right: GIF display
gif_label = Label(root)
gif_label.pack(side="right", padx=10, pady=10)

# Load the GIF
gif_path = "bicep_curl.gif"
gif = Image.open(gif_path)
frames = [ImageTk.PhotoImage(frame.copy().convert("RGBA")) for frame in ImageSequence.Iterator(gif)]
frame_count = len(frames)
gif_index = 0

# ---- OpenCV Setup ----
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detector = pm.posture_detector()

sets = 1
reps_per_set = 3
rest_time = 5

current_set = 1
count = 0
direction = 0
start_time = time.process_time()
state = "exercise"  # can be 'exercise' or 'rest'
rest_remaining = rest_time

# ---- Frame Update Functions ----
def update_gif_frame():
    global gif_index
    gif_label.configure(image=frames[gif_index])
    gif_index = (gif_index + 1) % frame_count
    gif_label.after(100, update_gif_frame)

def update_opencv_frame():
    global count, direction, current_set, state, rest_remaining, start_time

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    if state == "exercise":
        success, frame = cap.read()
        if success:
            frame = cv2.resize(frame, (960, 500))
            frame = detector.find_person(frame, False)
            landmarks = detector.find_landmarks(frame, False)

            if landmarks:
                angle = detector.find_angle(frame, 12, 14, 16)
                per = np.interp(angle, (50, 160), (0, 100))
                bar_pos = np.interp(per, (0, 100), (450, 100))
                color = utilities().get_performance_bar_color(per)

                if per == 100 or per == 0:
                    rep = utilities().repitition_counter(per, count, direction)
                    count = rep["count"]
                    direction = rep["direction"]

                utilities().draw_performance_bar(frame, per, bar_pos, color, count)

            utilities().display_rep_count(frame, count, reps_per_set)
            utilities().position_info_standing_exercise(frame, True)
            cv2.putText(frame, f"Set {current_set}/{sets}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if count >= reps_per_set:
                count = 0
                direction = 0
                if current_set < sets:
                    state = "rest"
                    rest_remaining = rest_time
                    current_set += 1
                else:
                    # time.sleep(5)
                    # cap.release()
                    
                    root.destroy()
                    subprocess.Popen(["python", "thank_you_page.py"])
                    # cv2.putText(frame, "Workout Complete!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

    elif state == "rest":
        frame[:] = 0
        cv2.putText(frame, f"Rest: {rest_remaining}s", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        rest_remaining -= 1
        if rest_remaining <= 0:
            state = "exercise"

    # Convert frame to Tkinter image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(rgb))
    video_label.imgtk = img
    video_label.configure(image=img)
    video_label.after(1000 if state == "rest" else 10, update_opencv_frame)

# ---- STOP Button ----
stop_btn = tk.Button(
    root,
    text="STOP",
    command=lambda: (cap.release(), root.destroy()),
    bg="#ff0000",
    fg="#ffffff",
    font=("Helvetica", 12, "bold"),
    bd=4,
    relief="raised",
    cursor="hand2",
    width=12,
    height=2
)
stop_btn.place(x=750, y=750)
stop_btn.bind("<Enter>", lambda e: e.widget.config(bg="#f80404"))
stop_btn.bind("<Leave>", lambda e: e.widget.config(bg="#ff0000"))

# ---- Start GUI Loop ----
update_gif_frame()
update_opencv_frame()
root.mainloop()
