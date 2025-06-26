import tkinter as tk
from PIL import Image, ImageTk, ImageEnhance
import subprocess

def launch_gui():
    root = tk.Tk()
    root.geometry("925x500+300+200")
    root.resizable(False, False)
    root.title("Fitness Dashboard")

    # Load background image
    image_path = "background.PNG"
    bg_image = Image.open(image_path).resize((925, 500), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Button config: (image_path, x, y)
    button_data = [
        ("character.jpeg", 0, 0),
        ("no_ai.jpeg", 455, 0),
        # ("workouts_button.jpeg", 445, 250),
    ]

    def handle_button_click(name):
        if name == "Character.jpeg":
            subprocess.Popen(["python", "ai_dash.py"])
            root.destroy()
        elif name == "No":
            subprocess.Popen(["python", "no_ai.py"])
            root.destroy()
        # elif name == "Workouts":
        #     subprocess.Popen(["python", "workout_main.py"])
        else:
            print(f"{name} clicked!")

    buttons = []

    for img_path, x, y in button_data:
        # Load and resize original image
        normal_img = Image.open(img_path).resize((472, 500), Image.LANCZOS).convert("RGBA")

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

    root.bg_photo = bg_photo
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
