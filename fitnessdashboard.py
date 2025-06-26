from PIL import Image, ImageTk
import tkinter as tk
import os
import tkinter as tk
from PIL import ImageEnhance
import subprocess

class ModernCardButton(tk.Frame):
    def __init__(self, master, title, subtitle, bg_color, command=None, *args, **kwargs):
        super().__init__(master, bg=bg_color, width=250, height=100, *args, **kwargs)
        self.command = command
        self.bg_color = bg_color
        self.hover_color = self._shade_color(bg_color, 0.9)

        self.configure(highlightthickness=0, bd=0)
        self.pack_propagate(False)

        self.title_lbl = tk.Label(self, text=title, font=("Helvetica", 16, "bold"), bg=bg_color, fg="white")
        self.title_lbl.pack(pady=(15, 0))

        self.subtitle_lbl = tk.Label(self, text=subtitle, font=("Helvetica", 10), bg=bg_color, fg="white")
        self.subtitle_lbl.pack()

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        self.title_lbl.bind("<Button-1>", self.on_click)
        self.subtitle_lbl.bind("<Button-1>", self.on_click)

    def _shade_color(self, color, factor):
        color = color.lstrip('#')
        r, g, b = [int(color[i:i+2], 16) for i in (0, 2 ,4)]
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        return f'#{r:02x}{g:02x}{b:02x}'

    def on_enter(self, event):
        self.config(bg=self.hover_color)
        self.title_lbl.config(bg=self.hover_color)
        self.subtitle_lbl.config(bg=self.hover_color)

    def on_leave(self, event):
        self.config(bg=self.bg_color)
        self.title_lbl.config(bg=self.bg_color)
        self.subtitle_lbl.config(bg=self.bg_color)

    def on_click(self, event):
        if self.command:
            self.command()

# # Demo App
# def launch_gui():
#     root = tk.Tk()
#     root.geometry("600x400")
#     root.configure(bg="#1e1e2f")
#     root.title("Modern Button Demo")

#     def dummy_action():
#         print("Card clicked!")

#     cards = [
#         ("BMI", "Calculate your BMI", "#7E57C2"),
#         ("Calorie", "Daily calorie needs", "#42A5F5"),
#         ("Diet Plan", "Recommended diet", "#FFCA28"),
#         ("Workouts", "Fitness routines", "#EF5350"),
#     ]

#     row = 0
#     col = 0
#     for i, (title, subtitle, color) in enumerate(cards):
#         card = ModernCardButton(root, title, subtitle, color, command=dummy_action)
#         card.grid(row=row, column=col, padx=20, pady=20)
#         col += 1
#         if col > 1:
#             col = 0
#             row += 1

#     root.mainloop()
# def launch_gui():
#     root = tk.Tk()
#     root.geometry("925x500+300+200")
#     root.resizable(False,False)
#     root.title("Modern Button Demo")

#     # Load background image
#     image_path = "background.jpeg"
#     bg_image = Image.open(image_path)
#     bg_image = bg_image.resize((925, 500), Image.LANCZOS)

#     bg_photo = ImageTk.PhotoImage(bg_image)

#     # Create a Label to hold the image
#     bg_label = tk.Label(root, image=bg_photo)
#     bg_label.place(x=0, y=0, relwidth=1, relheight=1)

#     def dummy_action():
#         print("Card clicked!")

#     cards = [
#         ("BMI", "Calculate your BMI", "#7E57C2"),
#         ("Calorie", "Daily calorie needs", "#42A5F5"),
#         ("Diet Plan", "Recommended diet", "#FFCA28"),
#         ("Workouts", "Fitness routines", "#EF5350"),
#     ]

#     row = 0
#     col = 0
#     for i, (title, subtitle, color) in enumerate(cards):
#         card = ModernCardButton(root, title, subtitle, color, command=dummy_action)
#         card.grid(row=row, column=col, padx=20, pady=20)
#         col += 1
#         if col > 1:
#             col = 0
#             row += 1

#     # Keep a reference to avoid garbage collection
#     root.bg_photo = bg_photo

#     root.mainloop()
# def launch_gui():
#     root = tk.Tk()
#     root.geometry("925x500+300+200")
#     root.resizable(False, False)
#     root.title("Modern Button Demo")

#     # Load background
#     image_path = "background.jpeg"
#     bg_image = Image.open(image_path).resize((925, 500), Image.LANCZOS)
#     bg_photo = ImageTk.PhotoImage(bg_image)
#     bg_label = tk.Label(root, image=bg_photo)
#     bg_label.place(x=0, y=0, relwidth=1, relheight=1)

#     def dummy_action():
#         print("Card clicked!")

#     # Place first 3 cards
#     cards = [
#         ("BMI", "Calculate your BMI", "#7E57C2"),
#         ("Calorie", "Daily calorie needs", "#42A5F5"),
#         ("Diet Plan", "Recommended diet", "#FFCA28")
#     ]
#     row = 0
#     col = 0
#     for title, subtitle, color in cards:
#         card = ModernCardButton(root, title, subtitle, color, command=dummy_action)
#         card.grid(row=row, column=col, padx=20, pady=20)
#         col += 1
#         if col > 1:
#             col = 0
#             row += 1

#     # === Add image button for "Workouts" ===
#     normal_img = Image.open("workouts_button.jpeg").resize((250, 100), Image.LANCZOS)
#     hover_img = ImageEnhance.Brightness(normal_img).enhance(1.2)

#     normal_photo = ImageTk.PhotoImage(normal_img)
#     hover_photo = ImageTk.PhotoImage(hover_img)

#     img_btn = tk.Label(root, image=normal_photo, bd=0, cursor="hand2")
#     img_btn.image = normal_photo  # prevent garbage collection
#     img_btn.hover = hover_photo

#     def on_img_enter(event):
#         img_btn.configure(image=img_btn.hover)

#     def on_img_leave(event):
#         img_btn.configure(image=img_btn.image)

#     def on_img_click(event):
#         dummy_action()

#     img_btn.bind("<Enter>", on_img_enter)
#     img_btn.bind("<Leave>", on_img_leave)
#     img_btn.bind("<Button-1>", on_img_click)

#     img_btn.grid(row=row, column=col, padx=20, pady=20)

#     root.bg_photo = bg_photo  # preserve background image
#     root.mainloop()


# def launch_gui():
#     root = tk.Tk()
#     root.geometry("925x500+300+200")
#     root.resizable(False, False)
#     root.title("Modern Button Demo")

#     # Load background
#     image_path = "background.jpeg"
#     bg_image = Image.open(image_path).resize((925, 500), Image.LANCZOS)
#     bg_photo = ImageTk.PhotoImage(bg_image)
#     bg_label = tk.Label(root, image=bg_photo)
#     bg_label.place(x=0, y=0, relwidth=1, relheight=1)

#     def dummy_action():
#         print("Card clicked!")

#     # Define button info and manual positions
#     buttons = [
#         ("BMI", "Calculate your BMI", "#7E57C2", 100, 100),
#         ("Calorie", "Daily calorie needs", "#42A5F5", 600, 100),
#         ("Diet Plan", "Recommended diet", "#FFCA28", 100, 300),
#         ("Workouts", "Fitness routines", "#EF5350", 600, 300),
#     ]

#     for title, subtitle, color, x, y in buttons:
#         card = ModernCardButton(root, title, subtitle, color, command=dummy_action)
#         card.place(x=x, y=y)

#     root.bg_photo = bg_photo  # preserve background image
#     root.mainloop()

def launch_gui():
    root = tk.Tk()
    root.geometry("925x500+300+200")
    root.resizable(False, False)
    root.title("Fitness Dashboard")

    # Load background
    image_path = "background.PNG"
    bg_image = Image.open(image_path).resize((925, 500), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Button config: (image_path, x, y)
    button_data = [
        ("bmi_button.jpeg", 168, 142),
        ("dietplan_button.jpeg", 718, 130),
        # ("workouts_button.jpeg", 60, 300),
        ("workouts_button.jpeg", 445, 250),
    ]

    # def dummy_action(name):
    #     print(f"{name} clicked!")
    # def handle_button_click(name):
        
    #     if name == "Bmi":
    #         subprocess.Popen(["python", "bmi.py"])
    #     else:
    #         print(f"{name} clicked!")
    def handle_button_click(name):
        if name == "Bmi":
            subprocess.Popen(["python", "bmi.py"])
        elif name == "Dietplan":
            subprocess.Popen(["python", "diet_update.py"])
        elif name == "Workouts":
            subprocess.Popen(["python", "ai_vs.py"])
            # root.destroy()
        else:
            print(f"{name} clicked!")


    for img_path, x, y in button_data:
        # Load normal and hover image
        normal_img = Image.open(img_path).resize((21, 21), Image.LANCZOS)
        hover_img = ImageEnhance.Brightness(normal_img).enhance(1.2)

        normal_photo = ImageTk.PhotoImage(normal_img)
        hover_photo = ImageTk.PhotoImage(hover_img)

        # Create label as image button
        img_btn = tk.Label(root, image=normal_photo, bd=0, cursor="hand2", bg="black")
        img_btn.image = normal_photo  # prevent garbage collection
        img_btn.hover = hover_photo
        img_btn.normal = normal_photo

        # Bind hover & click
        img_btn.bind("<Enter>", lambda e, b=img_btn: b.config(image=b.hover))
        img_btn.bind("<Leave>", lambda e, b=img_btn: b.config(image=b.normal))
        # img_btn.bind("<Button-1>", lambda e, name=img_path.split("_")[0].capitalize(): dummy_action(name))
        img_btn.bind("<Button-1>", lambda e, name=img_path.split("_")[0].capitalize(): handle_button_click(name))


        # Place it manually
        img_btn.place(x=x, y=y)

    root.bg_photo = bg_photo  # avoid garbage collection
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
