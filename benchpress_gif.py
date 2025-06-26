import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk, ImageSequence
import subprocess

# Create main window
root = tk.Tk()
root.title("OpenCV")
# root.geometry("850x600")
root.attributes('-fullscreen', True)

# Left side: OpenCV feed
video_label = Label(root)
video_label.pack(side="left", padx=10, pady=10)

# Right side: GIF display
gif_label = Label(root)
gif_label.pack(side="right", padx=10, pady=10)

# Capture video (0 = default webcam)
cap = cv2.VideoCapture(0)

# Load GIF
gif_path = "benchpress.gif"  # Replace with your GIF path
gif = Image.open(gif_path)
frames = [ImageTk.PhotoImage(frame.copy().convert("RGBA")) for frame in ImageSequence.Iterator(gif)]
frame_count = len(frames)
gif_index = 0

def update_opencv_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (960, 500))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        video_label.imgtk = img
        video_label.configure(image=img)
    video_label.after(10, update_opencv_frame)

def update_gif_frame():
    global gif_index
    gif_label.configure(image=frames[gif_index])
    gif_index = (gif_index + 1) % frame_count
    gif_label.after(100, update_gif_frame)  # Adjust delay for smoothness

# Start both updates
update_opencv_frame()
update_gif_frame()

prev_btn = tk.Button(
    root,
    text="STOP",
    command=lambda: (root.destroy()),#, subprocess.Popen(["python", "ai_dash.py"])
    bg="#ff0000",
    fg="#000000",
    font=("Helvetica", 12, "bold"),
    bd=4,
    relief="raised",
    cursor="hand2",
    width=12,
    height=2
)

prev_btn.place(x=750, y=750)  # Bottom-left corner

prev_btn.bind("<Enter>", lambda e: e.widget.config(bg="#f80404"))
prev_btn.bind("<Leave>", lambda e: e.widget.config(bg="#ff0000"))
# Run the application
root.mainloop()

# Cleanup on close
cap.release()
