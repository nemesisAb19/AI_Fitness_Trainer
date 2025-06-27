# AI Fitness Trainer (Tkinter Desktop App)

An AI-powered fitness trainer desktop application built using **Python**, **Tkinter**, **OpenCV**, and **MediaPipe**. This app provides real-time pose detection, guided workouts, voice feedback, and a beautiful GUI for an interactive fitness experience.

---

## 📁 Project Folder Structure

AI_PersonTrainer-main/

├── assets/ # Icons, GIFs, logo images

│ ├── bicep_curl.gif

│ ├── cross-button.png

│ ├── befit-logo-white.png

│ ├── next-icon.png

│ ├── prev-icon.png

│ └── start-button.png

│

├── core/ # Core logic modules

│ ├── pose_detection/

│ │ ├── PoseModule.py

│ │ └── face_detection.py

│ ├── audio/

│ │ └── AudioCommSys.py

│ ├── camera/

│ │ ├── camera.py

│ │ └── camera_check.py

│ ├── exercise_logic/

│ │ ├── ExercisesModule.py

│ │ └── bicep_check.py

│

├── ui/ # User interface components

│ ├── frames/

│ │ ├── BicepCurlSettingsFrame.py

│ │ ├── ExerciseFrame.py

│ │ ├── WorkoutsFrame.py

│ │ ├── NextPageFrame.py

│ │ └── StreamingFrame.py

│ └── components/

│ └── TransitionManager.py

│
├── .venv/   # Python virtual environment (excluded from GitHub)

├── main.py   # Entry point for launching the app

├── requirements.txt   # List of required Python packages

└── README.md   # Project overview and instructions

---

## Tech Stack

**Languages & UI Framework**
- Python 3.x
- Tkinter (for GUI)

**Computer Vision**
- OpenCV
- MediaPipe

**Voice Interaction**
- pyttsx3 (Text-to-Speech)
- SpeechRecognition

**Image & Media Handling**
- Pillow

**Other**
- Virtual Environment (`venv`)
- Git & GitHub for version control

---

## Features

- Real-time pose detection using **MediaPipe**
- Rep counting and accuracy tracking
- Interactive GUI with scrollable workout selection
- Voice feedback (Text-to-Speech)
- Tkinter-based animated transitions
- Clean, beginner-friendly codebase

---

## How to Download & Run the Project
1. **Clone the Repository**
   ```bash
   git clone https://github.com/nemesisAb19/AI_Fitness_Trainer.git
   cd AI_PersonTrainer-main
   
2. **Set Up a Virtual Environment (Recommended)**
   ```bash
   python -m venv .venv
- Activate it:
  - On Windows:
    ```bash
    .venv\Scripts\activate
  - On macOS/Linux:
    ```bash
    source .venv/bin/activate
   
3. **Install the Dependencies**
   ```bash
   pip install -r requirements.txt
   
4. **Run the App**
   ```bash
   python main.py

---

## Requirements

Listed in requirements.txt but key libraries include:

- opencv-python
- mediapipe
- Pillow
- pyttsx3
- SpeechRecognition
- tkinter (built into Python)

---

## Screenshots

![Image](https://github.com/user-attachments/assets/5c324fcb-3534-450b-9af7-6e8335a9246d)

![Image](https://github.com/user-attachments/assets/d2f8cec1-ef82-4ed9-9bd2-12abb9e56537)

![Image](https://github.com/user-attachments/assets/90944f06-1334-44cf-bce2-677a1f662778)

![Image](https://github.com/user-attachments/assets/de6794eb-9a0b-4cdc-a01e-a3bc145a3585)

![Image](https://github.com/user-attachments/assets/187eb040-eb70-431b-a897-c2e24c2d394b)

![Image](https://github.com/user-attachments/assets/cb99a5c5-a722-46e2-89ff-4375297f22c2)

![Image](https://github.com/user-attachments/assets/0c4b7a8d-d2e7-441f-8496-beba9eaddf0b)

![Image](https://github.com/user-attachments/assets/b3b4c5a1-4b63-40e3-8ffa-b28184380f28)

![Image](https://github.com/user-attachments/assets/48ee6b89-e196-4ba6-a9c3-9d1321ed530a)

![Image](https://github.com/user-attachments/assets/8327a41f-ee96-4bdb-b7c7-40a4adfe33ff)

![Image](https://github.com/user-attachments/assets/f5a499e0-4d46-4e8f-b214-7c937d2f77bd)

---

## 📃 License

This project is for educational/demo purposes and is not currently under a formal open-source license.
