import tkinter as tk
import time

def countdown_timer(seconds):
    def update():
        nonlocal seconds
        if seconds > 0:
            label.config(text=f"Next Exercise>>\n{seconds}",font=("Helvetica", 60, "bold"))
            # label.config(text=f"Starting in\n{seconds}",font=("Helvetica", 60, "bold"))# text=f"Starting in:\n{seconds}",
            seconds -= 1
            root.after(1000, update)
        else:
            root.destroy()

    root = tk.Tk()
    root.title("Rest Timer")
    root.attributes("-fullscreen", True)
    root.configure(bg="black")

    label = tk.Label(root, text="", fg="white", bg="black")
    label.pack(expand=True)

    update()
    root.mainloop()

if __name__ == "__main__":
    countdown_timer(20)
