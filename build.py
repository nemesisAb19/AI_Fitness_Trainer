import tkinter as tk
from PIL import Image, ImageTk, ImageEnhance
import subprocess

def launch_gui():
    root = tk.Tk()
    root.geometry("925x500+300+200")
    root.resizable(False, False)
    root.title("Fitness Dashboard")

    image_path = "plainbackground.jpeg"
    bg_image = Image.open(image_path).resize((925, 500), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Start Button
    button_data2 = [("start_button.jpeg", 720, 450)]

    # def handle_button_click2(name):
    #     if name == "Start":
    #         print("Exercise Order:")
    #         for idx, ex in enumerate(selected_buttons, 1):
    #             print(f"{idx}. {ex}")
    #     else:
    #         print(f"{name} clicked!")
    # def handle_button_click2(name):
    #     if name == "Start":
    #         with open("selected_exercises.txt", "w") as f:
    #             for ex in selected_buttons:
    #                 f.write(f"{ex}\n")
    #         subprocess.Popen(["python", "count.py"])
    #         root.destroy()

    def handle_button_click2(name):
        if name == "Start":
            with open("selected_exercises.txt", "w") as f:
                for ex in selected_buttons:
                    f.write(f"{ex}\n")
            subprocess.Popen(["python", "count.py"])
            root.destroy()



    for img_path, x, y in button_data2:
        normal_img = Image.open(img_path).resize((150, 45), Image.LANCZOS).convert("RGBA")
        bright_img = ImageEnhance.Brightness(normal_img).enhance(1.4)
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
            command=lambda name=button_name: handle_button_click2(name)
        )
        btn.normal_photo = normal_photo
        btn.bright_photo = bright_photo

        def on_enter(e, b=btn):
            b.config(image=b.bright_photo)

        def on_leave(e, b=btn):
            b.config(image=b.normal_photo)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        btn.place(x=x, y=y)

    # Exercise Buttons
    button_data = [
        ("pushups_button.jpeg", 93, 55),
        ("bicepcurls_button.jpeg", 286, 55),
        ("benchpress_button.jpeg", 478, 55),
        ("mountainclimbers_button.jpeg", 670, 55),
        ("shoulder_button.jpeg", 93, 270),
        ("situps_button.jpeg", 286, 270),
        ("squat_button.jpeg", 478, 270),
        ("pullups_button.jpeg", 670, 270),
    ]

    selected_buttons = []
    button_refs = {}
    selection_labels = {}

    def update_selection_labels():
        # Clear all existing labels
        for name, label in selection_labels.items():
            label.destroy()
        selection_labels.clear()

        # Create new numbered labels
        for idx, name in enumerate(selected_buttons, 1):
            btn = button_refs[name]
            label = tk.Label(root, text=str(idx), bg="green", fg="white",
                             font=("Helvetica", 10, "bold"), width=2)
            # Position label relative to button
            x = btn.winfo_x()
            y = btn.winfo_y()
            label.place(x=x+5, y=y+5)
            selection_labels[name] = label

    def handle_button_click(name, btn):
        if name in selected_buttons:
            selected_buttons.remove(name)
            btn.config(image=btn.normal_photo)
            btn.selected = False
        else:
            selected_buttons.append(name)
            btn.config(image=btn.bright_photo)
            btn.selected = True
        update_selection_labels()
        print("Selected Order:", selected_buttons)

    for img_path, x, y in button_data:
        normal_img = Image.open(img_path).resize((160, 160), Image.LANCZOS).convert("RGBA")
        bright_img = ImageEnhance.Brightness(normal_img).enhance(1.4)
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
            command=lambda name=button_name, b=None: handle_button_click(name, button_refs[name])
        )

        btn.normal_photo = normal_photo
        btn.bright_photo = bright_photo
        btn.selected = False

        def on_enter(e, b=btn):
            if not b.selected:
                b.config(image=b.bright_photo)

        def on_leave(e, b=btn):
            if not b.selected:
                b.config(image=b.normal_photo)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        btn.place(x=x, y=y)

        button_refs[button_name] = btn

    # Previous Button
    prev_btn = tk.Button(
        root,
        text="Previous",
        command=lambda: (root.destroy(), subprocess.Popen(["python", "ai_dash.py"])),
        bg="#ecc474",
        fg="#000000",
        font=("Helvetica", 6, "bold"),
        bd=4,
        relief="raised",
        cursor="hand2",
        width=12,
        height=2
    )
    prev_btn.place(x=20, y=430)
    prev_btn.bind("<Enter>", lambda e: e.widget.config(bg="#eccc7c"))
    prev_btn.bind("<Leave>", lambda e: e.widget.config(bg="#ecc474"))

    root.bg_photo = bg_photo
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
