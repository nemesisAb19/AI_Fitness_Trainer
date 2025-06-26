import tkinter as tk
from PIL import Image, ImageTk

# Initialize the main window
root = tk.Tk()
root.title("Workout Button Preview")
root.geometry("600x400")  # Adjusted to better fit the new image
root.configure(bg="#1e1e1e")

# Load and resize the new image
image_path = "236EEDBB-205B-4E0F-81EE-E402CB1C3056.jpeg"
image = Image.open(image_path)
image = image.resize((500, 200), Image.LANCZOS)  # Resize as needed
photo = ImageTk.PhotoImage(image)

# Create a button with the image
workout_button = tk.Button(
    root,
    image=photo,
    borderwidth=0,
    highlightthickness=0,
    bg="#1e1e1e",
    activebackground="#1e1e1e"
)
workout_button.pack(pady=60)

# Run the application
root.mainloop()
