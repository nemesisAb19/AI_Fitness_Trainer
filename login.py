from tkinter import *
from tkinter import messagebox
import ast
import subprocess
root=Tk()
root.title("Login")
root.geometry('925x500+300+200')
root.configure(bg="#fff")
root.resizable(False,False)

def signin():
    username = user.get()
    password = code.get()


    file=open('datasheet.txt','r')
    d=file.read()
    r=ast.literal_eval(d)
    file.close()

    # print(r.keys())
    # print(r.values())



    # if username in r.keys() and password==r[username]:
    #     # screen = Toplevel(root)
    #     # screen.title("App")
    #     # screen.geometry('925x500+300+200')
    #     # screen.config(bg="white")
    #     # Label(screen, text='Hello Everyone!', bg='#fff', font=('Calibri (Body)', 50, 'bold')).pack(expand=True)
    #     # screen.mainloop()
    #     # def open_dashboard():
    #     screen = Toplevel(root)
    #     screen.title("Fitness Dashboard")
    #     screen.geometry("1000x600+200+100")
    #     screen.config(bg="#f5f6fa")

    # # --- Welcome Banner ---
    #     welcome_frame = Frame(screen, bg="#ffe5b4", height=100, width=960)
    #     welcome_frame.place(x=20, y=20)
    #     Label(welcome_frame, text=f"Welcome back, {username}!",
    #       bg="#ffe5b4", font=("Helvetica", 20, "bold")).place(x=20, y=20)
    #     Label(welcome_frame, text="You've achieved 80% of your weekly fitness goal. Keep it up!",
    #       bg="#ffe5b4", font=("Arial", 12)).place(x=20, y=60)

    # # --- User Profile Panel ---
    #     profile_frame = Frame(screen, bg="#ffffff", height=200, width=250)
    #     profile_frame.place(x=720, y=140)
    #     Label(profile_frame, text="Anna Morrison", font=("Arial", 14, "bold"), bg="white").pack(pady=10)
    #     Label(profile_frame, text="Fitness Enthusiast", font=("Arial", 10), bg="white").pack()
    #     Label(profile_frame, text="🏋️ B2 Level", bg="white", font=("Arial", 12)).pack(pady=5)

    # # --- Workout Summary (Progress Bars) ---
    #     summary_frame = Frame(screen, bg="white", height=180, width=400)
    #     summary_frame.place(x=20, y=140)
    #     Label(summary_frame, text="Workout Summary", font=("Arial", 14, "bold"), bg="white").pack(pady=10)

    #     workouts = [("Squats", 80), ("Push-ups", 60), ("Planks", 45)]
    #     for i, (workout, percent) in enumerate(workouts):
    #         Label(summary_frame, text=workout, bg="white", anchor="w").place(x=10, y=50+i*40)
    #         Canvas(summary_frame, width=300, height=20, bg="#ddd").place(x=80, y=50+i*40)
    #         Canvas(summary_frame, width=3*percent, height=20, bg="#57a1f8").place(x=80, y=50+i*40)

    # # --- Time Graph (Placeholder) ---
    #     graph_frame = Frame(screen, bg="white", height=200, width=400)
    #     graph_frame.place(x=20, y=340)
    #     Label(graph_frame, text="Time Spent on Workouts", font=("Arial", 14, "bold"), bg="white").pack(pady=10)
    # # Add bars or use matplotlib later

    # # --- Reminders ---
    #     reminders = ["15 Squats", "Drink Water", "Stretch Legs"]
    #     reminder_frame = Frame(screen, bg="white", height=200, width=250)
    #     reminder_frame.place(x=720, y=360)
    #     Label(reminder_frame, text="Reminders", font=("Arial", 14, "bold"), bg="white").pack(pady=10)
    #     for i, item in enumerate(reminders):
    #         Label(reminder_frame, text=f"• {item}", bg="white", anchor="w").pack(padx=10, anchor="w")

    # # --- Start Workout Button ---
    #     Button(screen, text="Start Workout", font=("Arial", 12, "bold"),
    #        bg="#57a1f8", fg="white").place(x=720, y=570)#, command=start_workout
    
    if username in r.keys() and password==r[username]:
        subprocess.Popen(["python", "fitnessdashboard.py"])
    else:
        messagebox.showerror('Invalid', 'invalid username or password')

######################signupststart#################################
def signup_command():
    window=Toplevel(root)
    window.title("SignUp")
    window.geometry('925x500+300+200')
    window.configure(bg='#fff')
    window.resizable(False, False)

    def signup():
        username = user.get()
        password = code.get()
        confirm_password = confirm_code.get()
        if password == confirm_password:
            try:
                file = open('datasheet.txt', 'r+')
                d = file.read()
                r = ast.literal_eval(d)
                dict2 = {username: password}
                r.update(dict2)
                file.truncate(0)
                file.close()
                file = open('datasheet.txt', 'w')
                w = file.write(str(r))
                messagebox.showinfo('Signup', 'Sucessfully sign up')
                window.destroy()
            except:
                file = open('datasheet.txt', 'w')
                pp = str({'Username': 'password'})
                file.write(pp)
                file.close()
        else:
            messagebox.showerror('Invalid', "Both Password should match")



    def sign():
        window.destroy()


    img = PhotoImage(file='login.png')
    Label(window,image=img, border=0,bg='white').place(x=50,y=90)

    frame=Frame(window,width=350,height=390,bg='white')
    frame.place(x=480,y=50)

    heading = Label(frame, text='Sign up', fg="#57a1f8", bg='white', font=('Microsoft Yahei UI Light', 23, 'bold'))
    heading.place(x=100, y=5)

    def on_enter(e):
        user.delete(0, 'end')

    def on_leave(e):
        if user.get() == '':
            user.insert(0, 'Username')

    user = Entry(frame, width=25, fg='black', border=0, bg='white', font=('Microsoft Yahei UI Light', 11))
    user.place(x=30, y=80)
    user.insert(0, 'Username')

    user.bind("<FocusIn>", on_enter)
    user.bind("<FocusOut>", on_leave)

    Frame(frame, width=295, height=2, bg='black').place(x=25, y=107)

############################################

    def on_enter(e):
        code.delete(0, 'end')

    def on_leave(e):
        if code.get() == '':
            code.insert(0, 'Password')

    code = Entry(frame, width=25, fg='black', border=0, bg='white', font=('Microsoft Yahei UI Light', 11))
    code.place(x=30, y=150)
    code.insert(0, 'Password')
    code.bind("<FocusIn>", on_enter)
    code.bind("<FocusOut>", on_leave)

    Frame(frame, width=295, height=2, bg='black').place(x=25, y=177)

##################################
    def on_enter(e):
        confirm_code.delete(0, 'end')

    def on_leave(e):
        if confirm_code.get() == '':
            confirm_code.insert(0, 'Confirm Password')

    confirm_code = Entry(frame, width=25, fg='black', border=0, bg='white', font=('Microsoft Yahei UI Light', 11))
    confirm_code.place(x=30, y=220)
    confirm_code.insert(0, 'Confirm Password')
    confirm_code.bind("<FocusIn>", on_enter)
    confirm_code.bind("<FocusOut>", on_leave)

    Frame(frame, width=295, height=2, bg='black').place(x=25, y=247)

#################################

    Button(frame, width=39, pady=7, text='Sign up', bg='#57a1f8', fg='white', border=0,command=signup).place(x=35, y=280)
    label = Label(frame, text='Already have an account', fg='black', bg='white', font=('Microsoft YaHei UI Light', 9))
    label.place(x=90, y=340)

    signin = Button(frame, width=6, text='Sign in', border=0, bg='white', cursor='hand2', fg='#57a1f8',command=sign)
    signin.place(x=200, y=340)


    window.mainloop()
    
####################signupend#####################################
img=PhotoImage(file='login.png')
Label(root,image=img,bg='white').place(x=50,y=50)

frame=Frame(root,width=350,height=350,bg="white")
frame.place(x=480,y=70)

heading=Label(frame,text='Sign in',fg='#57a1f8',bg='white',font=('Microsoft YaHei UI Light',23,'bold'))
heading.place(x=100,y=5)


def on_enter(e):
    user.delete(0,'end')

def on_leave(e):
    name=user.get()
    if name=='':
        user.insert(0,'Username')


user=Entry(frame,width=25,fg='black',border=0,bg="white",font=('Microsoft YaHei UI Light',11))
user.place(x=30,y=80)
user.insert(0,'Username')
user.bind('<FocusIn>', on_enter)
user.bind('<FocusOut>', on_leave)

Frame(frame,width=295,height=2,bg='black').place(x=25,y=107)


def on_enter(e):
    code.delete(0, 'end')

def on_leave(e):
    name = code.get()
    if name == '':
        code.insert(0, 'Password')

code = Entry(frame, width=25, fg='black', border=0, bg="white", font=('Microsoft YaHei UI Light', 11))
code.place(x=30, y=150)
code.insert(0, 'Password')
code.bind('<FocusIn>', on_enter)
code.bind('<FocusOut>', on_leave)

Frame(frame, width=295, height=2, bg='black').place(x=25, y=177)


Button(frame, width=39, pady=7, text='Sign in', bg='#57a1f8', fg='white',command = signin, border=0).place(x=35, y=204)
label = Label(frame, text="Don't have an account?", fg='black', bg='white', font=('Microsoft YaHei UI Light', 9))
label.place(x=75, y=270)

sign_up = Button(frame, width=6, text='Sign up', border=0, bg='white', cursor='hand2', fg='#57a1f8',command=signup_command)
sign_up.place(x=215, y=270)



root.mainloop()