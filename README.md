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

### 1. Clone the Repository
```bash
git clone https://github.com/nemesisAb19/AI_Fitness_Trainer.git
cd AI_Fitness_Trainer

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv .venv

- Activate it:
  - On Windows:
    ```bash
    .venv\Scripts\activate
  - On macOS/Linux:
    ```bash
    source .venv/bin/activate

### 3. Install the Dependencies
```bash
pip install -r requirements.txt

### 4. Run the App
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

---

## 📃 License

This project is for educational/demo purposes and is not currently under a formal open-source license.
