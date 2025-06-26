import tkinter as tk
from PIL import Image, ImageTk, ImageEnhance
import subprocess

def launch_gui():
    root = tk.Tk()
    root.geometry("925x500+300+200")
    root.resizable(False, False)
    root.title("Fitness Dashboard")

    # Load background image
    image_path = "count.jpeg"
    bg_image = Image.open(image_path).resize((925, 500), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # ======================= Spinbox Section =======================
    label_font = ("Helvetica", 22, "bold")
    label_fg = "white"

    form_frame = tk.Frame(root, bg="#f89474")
    form_frame.place(x=320, y=155)

    tk.Label(form_frame, text="Sets", font=label_font, fg=label_fg, bg="#f89474").grid(row=0, column=0, sticky="w", pady=10)
    sets_spinbox = tk.Spinbox(form_frame, from_=1, to=10, width=5)
    sets_spinbox.grid(row=0, column=1, padx=20)

    tk.Label(form_frame, text="Reps", font=label_font, fg=label_fg, bg="#f89474").grid(row=1, column=0, sticky="w", pady=10)
    reps_spinbox = tk.Spinbox(form_frame, from_=1, to=50, width=5)
    reps_spinbox.grid(row=1, column=1, padx=20)

    tk.Label(form_frame, text="Rest Time (s)", font=label_font, fg=label_fg, bg="#f89474").grid(row=2, column=0, sticky="w", pady=10)
    rest_spinbox = tk.Spinbox(form_frame, from_=5, to=300, width=5)
    rest_spinbox.grid(row=2, column=1, padx=20)

    # ======================= Button Section =======================
    # Button config: (image_path, x, y)
    button_data = [
        ("start_button.jpeg", 680, 400),
    ]

    def handle_button_click(name):
        # if name == "Start":
        #     sets = int(sets_spinbox.get())
        #     reps = int(reps_spinbox.get())
        #     rest = int(rest_spinbox.get())
        #     print(f"Sets: {sets}, Reps: {reps}, Rest Time: {rest} seconds")
            
        #     # Pass values to next script if needed (e.g., via environment variables, file, or arguments)
        #     # subprocess.Popen(["python", "pushups_page_ai.py"])
        #     root.destroy()
        if name == "Start":
            sets = int(sets_spinbox.get())
            reps = int(reps_spinbox.get())
            rest = int(rest_spinbox.get())
            print(f"Sets: {sets}, Reps: {reps}, Rest Time: {rest} seconds")

            subprocess.Popen([
                "python", "chestpress_final.py",
                str(sets), str(reps), str(rest)
                ])
            # root.destroy()
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

        btn.normal_photo = normal_photo
        btn.bright_photo = bright_photo

        def on_enter(e, b=btn):
            b.config(image=b.bright_photo)

        def on_leave(e, b=btn):
            b.config(image=b.normal_photo)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

        btn.place(x=x, y=y)
        buttons.append(btn)

    # Previous button
    prev_btn = tk.Button(
        root,
        text="Previous",
        command=lambda: (root.destroy(), subprocess.Popen(["python", "benchpress_page_ai.py"])),
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
