import tkinter as tk
from PIL import Image, ImageTk, ImageEnhance
import subprocess
from tkinter import messagebox

def launch_gui():
    root = tk.Tk()
    root.geometry("925x500+300+200")
    root.resizable(False, False)
    root.title("Fitness Dashboard")

    image_path = "background_nonai.jpeg"
    bg_image = Image.open(image_path).resize((925, 500), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    button_data = [
        ("upper_body.jpeg", 70, 55),
        ("lower_body.jpeg", 530, 55),
    ]

    def handle_button_click(name):
        if name == "Upper":
            subprocess.Popen(["python", "bmi.py"])
        elif name == "Lower":
            subprocess.Popen(["python", "bmi.py"])
        else:
            print(f"{name} clicked!")

    buttons = []
    name_to_button = {}

    for img_path, x, y in button_data:
        normal_img = Image.open(img_path).resize((325, 325), Image.LANCZOS).convert("RGBA")
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
        buttons.append((btn, x, y))
        name_to_button[button_name.lower()] = (btn, button_name)

    suggestion_box = tk.Listbox(root, font=("Helvetica", 10), height=4)
    suggestion_box.place(x=620, y=45, width=185)
    suggestion_box.place_forget()

    def show_buttons_in_queue(matched_names):
        for btn, _, _ in buttons:
            btn.place_forget()

        start_x = 70
        y = 100
        gap = 350

        for index, name in enumerate(matched_names):
            btn, _ = name_to_button[name]
            x = start_x + index * gap
            btn.place(x=x, y=y)

    def show_all_buttons():
        for btn, x, y in buttons:
            btn.place(x=x, y=y)

    def search_button():
        query = search_entry.get().strip().lower()
        matched = [key for key in name_to_button if query in key]

        if matched:
            show_buttons_in_queue(matched)

            # Just highlight the first match (no auto-click)
            first_btn, _ = name_to_button[matched[0]]
            first_btn.focus_set()
            first_btn.config(highlightbackground="yellow", highlightthickness=4)
            root.after(500, lambda: first_btn.config(highlightthickness=0))
        else:
            messagebox.showinfo("Not Found", f"No button found for '{query}'.")
            show_all_buttons()

    def update_suggestions(event=None):
        query = search_entry.get().strip().lower()
        suggestion_box.delete(0, tk.END)

        if query == "":
            suggestion_box.place_forget()
            show_all_buttons()
            return

        matches = [name for name in name_to_button if query in name]
        if matches:
            for name in matches:
                suggestion_box.insert(tk.END, name.capitalize())
            suggestion_box.place(x=620, y=45, width=185)
        else:
            suggestion_box.place_forget()

    def select_suggestion(event):
        if suggestion_box.curselection():
            selected = suggestion_box.get(suggestion_box.curselection())
            search_entry.delete(0, tk.END)
            search_entry.insert(0, selected)
            suggestion_box.place_forget()
            search_button()

    search_entry = tk.Entry(root, font=("Helvetica", 12))
    search_entry.place(x=620, y=20, width=180)
    search_entry.bind("<KeyRelease>", update_suggestions)
    search_entry.bind("<Return>", lambda e: search_button())

    suggestion_box.bind("<<ListboxSelect>>", select_suggestion)

    search_btn = tk.Button(
        root,
        text="Search",
        command=search_button,
        bg="#ecc474",
        fg="black",
        font=("Helvetica", 10, "bold")
    )
    search_btn.place(x=810, y=17)

    prev_btn = tk.Button(
        root,
        text="Previous",
        command=lambda: (root.destroy(), subprocess.Popen(["python", "ai_vs.py"])),
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
