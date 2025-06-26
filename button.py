import tkinter as tk
from PIL import Image, ImageTk

# Create main window
root = tk.Tk()
root.title("Button with Image and Border")
root.geometry("300x200")

# Load image using PIL
image = Image.open("bmi_button.jpeg")  # Replace with your image file
image = image.resize((30, 30))         # Resize as needed
button_image = ImageTk.PhotoImage(image)

# Create a button with image and border
button_with_border = tk.Button(
    root,
    
    image=button_image,
    # compound="left",  # Show image to the left of text
    # bd=5,              # Border width
    # relief="raised",   # Border style
    font=("Arial", 14)
)
button_with_border.pack(pady=50)

# Keep a reference to the image to avoid garbage collection
button_with_border.image = button_image

# Start the main event loop
root.mainloop()
