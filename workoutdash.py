from PIL import Image, ImageTk
import tkinter as tk
import os
import subprocess

def launch_gui():
    root = tk.Tk()
    root.geometry("925x500+300+200")
    root.resizable(False, False)
    root.title("Fitness Dashboard")

    # Load background
    image_path = "workout_background.jpeg"
    bg_image = Image.open(image_path).resize((925, 500), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Button config: (button_text, x, y)
    button_data = [
        ("EXERCISES", 500, 100),
        ("BUILD YOUR OWN WORKOUT", 500, 250),
    ]

    def handle_button_click(name):
        if name == "EXERCISES":
            subprocess.Popen(["python", "bmi.py"])
        elif name == "Dietplan":
            subprocess.Popen(["python", "diet_update.py"])
        # elif name == "Workouts":
        #     subprocess.Popen(["python", "workout_main.py"])
        else:
            print(f"{name} clicked!")

    # def on_enter(e):
    #     e.widget['bg'] = '#45a049'  # darker green on hover

    # def on_leave(e):
    #     e.widget['bg'] = '#4CAF50'  # original green

    for name, x, y in button_data:
        btn = tk.Button(
            root,
            text=name,
            command=lambda n=name: handle_button_click(n),
            bg="#48b484",
            fg="white",
            font=("Helvetica", 18, "bold"),
            bd=4,
            relief="raised",
            cursor="hand2",
            width=25,
            height=5
        )
        btn.place(x=x, y=y)

        # btn.bind("<Enter>", on_enter)
        # btn.bind("<Leave>", on_leave)

    # === Add Previous Button ===
    prev_btn = tk.Button(
        root,
        text="Previous",
        command=root.destroy,  # Closes the current window
        bg="#f44336",          # Red color
        fg="white",
        font=("Helvetica", 8, "bold"),
        bd=4,
        relief="raised",
        cursor="hand2",
        width=12,
        height=2
    )
    prev_btn.place(x=20, y=430)  # Bottom-left corner

    prev_btn.bind("<Enter>", lambda e: e.widget.config(bg="#d32f2f"))
    prev_btn.bind("<Leave>", lambda e: e.widget.config(bg="#f44336"))

    root.bg_photo = bg_photo
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
