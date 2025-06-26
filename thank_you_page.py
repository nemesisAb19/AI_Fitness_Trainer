import tkinter as tk
from PIL import Image, ImageTk, ImageEnhance
import subprocess
import time
def launch_gui():
    root = tk.Tk()
    root.geometry("925x500+300+200")
    root.resizable(False, False)
    root.title("Fitness Dashboard")

    # Auto-close after 5 seconds (5000 milliseconds)
    root.after(5000, root.destroy)

    # Load background image
    image_path = "thank_you_page.jpeg"
    bg_image = Image.open(image_path).resize((925, 500), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # prev_btn = tk.Button(
    #     root,
    #     text="Previous",
    #     command=lambda: (root.destroy()),
    #     bg="#ecc474",
    #     fg="#000000",
    #     font=("Helvetica", 6, "bold"),
    #     bd=4,
    #     relief="raised",
    #     cursor="hand2",
    #     width=12,
    #     height=2
    # )
    # prev_btn.place(x=20, y=430)

    # prev_btn.bind("<Enter>", lambda e: e.widget.config(bg="#eccc7c"))
    # prev_btn.bind("<Leave>", lambda e: e.widget.config(bg="#ecc474"))

    root.bg_photo = bg_photo
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
