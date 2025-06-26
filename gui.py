import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import threading
from ExercisesModule import simulate_target_exercies

class FitnessApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("M.D. AI Fitness Trainer")
        self.geometry("400x300")
        self.resizable(False, False)
        
        self.frames = {}
        for F in (MainMenu, RepsPage):
            frame = F(self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        self.selected_exercise = None
        self.show_frame("MainMenu")

    def show_frame(self, name):
        frame = self.frames[name]
        frame.tkraise()

class MainMenu(tk.Frame):
    def __init__(self, controller):
        super().__init__(controller)
        self.controller = controller
        
        label = tk.Label(self, text="Choose Your Exercise", font=("Helvetica", 16))
        label.pack(pady=20)

        for ex in ["Push-up", "Bicep Curl", "Squat"]:
            btn = tk.Button(self, text=ex, width=20, height=2,
                            command=lambda e=ex: self.select_exercise(e))
            btn.pack(pady=5)

    def select_exercise(self, exercise):
        self.controller.selected_exercise = exercise
        self.controller.show_frame("RepsPage")

# class RepsPage(tk.Frame):
#     def __init__(self, controller):
#         super().__init__(controller)
#         self.controller = controller
        
#         self.label = tk.Label(self, text="", font=("Helvetica", 14))
#         self.label.pack(pady=20)
        
#         self.reps_var = tk.IntVar(value=10)
#         tk.Label(self, text="Select number of reps:").pack()
#         tk.Spinbox(self, from_=1, to=100, textvariable=self.reps_var).pack(pady=10)

#         self.start_btn = tk.Button(self, text="Start Exercise", command=self.start_exercise)
#         self.start_btn.pack(pady=10)

#         self.back_btn = tk.Button(self, text="Back to Main Menu", command=self.back_to_main)
#         self.back_btn.pack()

#     def tkraise(self, *args, **kwargs):
#         exercise = self.controller.selected_exercise
#         self.label.config(text=f"{exercise} - Reps Selection")
#         super().tkraise(*args, **kwargs)

#     def start_exercise(self):
#         reps = self.reps_var.get()
#         exercise = self.controller.selected_exercise
#         messagebox.showinfo("Exercise Started", f"Starting {reps} reps of {exercise}...")

#         # 🔁 Simulate Exercise Here
#         self.after(3000, self.finish_exercise)  # Placeholder for your real exercise function

#     def finish_exercise(self):
#         messagebox.showinfo("Done!", "Exercise completed!")
#         # You can return to RepsPage or go to summary
#         self.tkraise()

#     def back_to_main(self):
#         self.controller.show_frame("MainMenu")




class RepsPage(tk.Frame):
    def __init__(self, controller):
        super().__init__(controller)
        self.controller = controller
        
        self.label = tk.Label(self, text="", font=("Helvetica", 14))
        self.label.pack(pady=20)
        
        self.reps_var = tk.IntVar(value=10)
        tk.Label(self, text="Select number of reps:").pack()
        tk.Spinbox(self, from_=1, to=100, textvariable=self.reps_var).pack(pady=10)

        self.start_btn = tk.Button(self, text="Start Exercise", command=self.start_exercise_thread)
        self.start_btn.pack(pady=10)

        self.back_btn = tk.Button(self, text="Back to Main Menu", command=self.back_to_main)
        self.back_btn.pack()

    def tkraise(self, *args, **kwargs):
        exercise = self.controller.selected_exercise
        self.label.config(text=f"{exercise} - Reps Selection")
        super().tkraise(*args, **kwargs)

    def start_exercise_thread(self):
        self.start_btn.config(state="disabled")
        threading.Thread(target=self.start_exercise).start()

    # def start_exercise(self):
    #     reps = self.reps_var.get()
    #     exercise = self.controller.selected_exercise

    #     # Call appropriate method
    #     trainer = simulate_target_exercies()
    #     if exercise == "Push-up":
    #         trainer.push_ups(reps)
    #     elif exercise == "Bicep Curl":
    #         trainer.bicep(reps)
    #     elif exercise == "Squat":
    #         trainer.squat(reps)

    #     # After completion
    #     self.after(0, self.exercise_complete)
    def start_exercise(self):
        reps = self.reps_var.get()
        exercise = self.controller.selected_exercise

        trainer = simulate_target_exercies()

        if exercise == "Push-up":
            for _ in trainer.push_ups(reps):
                pass
        elif exercise == "Bicep Curl":
            for _ in trainer.bicep(reps):
                pass
        elif exercise == "Squat":
            for _ in trainer.squat(reps):
                pass

        self.after(0, self.exercise_complete)

    def exercise_complete(self):
        self.start_btn.config(state="normal")
        messagebox.showinfo("Done!", "Exercise completed!")
        self.tkraise()

    def back_to_main(self):
        self.controller.show_frame("MainMenu")

if __name__ == "__main__":
    app = FitnessApp()
    app.mainloop()
