import tkinter as tk
from PIL import Image, ImageTk, ImageEnhance
import subprocess

def launch_gui():
    root = tk.Tk()
    root.geometry("925x500+300+200")
    root.resizable(False, False)
    root.title("Fitness Dashboard")

    # Load background image
    image_path = "squat_page.jpeg"
    bg_image = Image.open(image_path).resize((925, 500), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Button config: (image_path, x, y)
    button_data = [
        ("start_button.jpeg", 680, 400),
        # ("lower_body.jpeg", 530, 55),
        # ("workouts_button.jpeg", 445, 250),
    ]

    def handle_button_click(name):
        if name == "Start":
            subprocess.Popen(["python", "squat_count.py"])
            root.destroy()
        # elif name == "Dietplan":
        #     subprocess.Popen(["python", "diet_update.py"])
        # elif name == "Workouts":
        #     subprocess.Popen(["python", "workout_main.py"])
        else:
            print(f"{name} clicked!")

    buttons = []

    for img_path, x, y in button_data:
        # Load and resize original image
        normal_img = Image.open(img_path).resize((150, 45), Image.LANCZOS).convert("RGBA")

        # Brightened version for hover effect
        enhancer = ImageEnhance.Brightness(normal_img)
        bright_img = enhancer.enhance(1.4)

        normal_photo = ImageTk.PhotoImage(normal_img)
        bright_photo = ImageTk.PhotoImage(bright_img)

        button_name = img_path.split("_")[0].capitalize()

        btn = tk.Button(
            root,
            image=normal_photo,
            bd=0,
            relief="flat",
            bg=root["bg"],
            activebackground=root["bg"],
            highlightthickness=0,
            command=lambda name=button_name: handle_button_click(name)
        )

        # Store both images to prevent garbage collection
        btn.normal_photo = normal_photo
        btn.bright_photo = bright_photo

        # Hover bindings
        def on_enter(e, b=btn):
            b.config(image=b.bright_photo)

        def on_leave(e, b=btn):
            b.config(image=b.normal_photo)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

        btn.place(x=x, y=y)
        buttons.append(btn)

    # prev_btn = tk.Button(
    #     root,
    #     text="Previous",
    #     command=root.destroy,  # Closes the current window
    #     bg="#ecc474",          # Red color
    #     fg="#000000",
    #     font=("Helvetica", 6, "bold"),
    #     bd=4,
    #     relief="raised",
    #     cursor="hand2",
    #     width=12,
    #     height=2
    # )
    prev_btn = tk.Button(
    root,
    text="Previous",
    command=lambda: (root.destroy(), subprocess.Popen(["python", "ai_dash2.py"])),
    bg="#ecc474",
    fg="#000000",
    font=("Helvetica", 6, "bold"),
    bd=4,
    relief="raised",
    cursor="hand2",
    width=12,
    height=2
    )

    prev_btn.place(x=0, y=468)  # Bottom-left corner

    prev_btn.bind("<Enter>", lambda e: e.widget.config(bg="#eccc7c"))
    prev_btn.bind("<Leave>", lambda e: e.widget.config(bg="#ecc474"))

    root.bg_photo = bg_photo
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
