import subprocess

def run_exercises():
    with open("config.txt", "r") as f:
        sets, reps, rest = [line.strip() for line in f.readlines()]

    with open("selected_exercises.txt", "r") as f:
        selected = [line.strip() for line in f.readlines()]

    exercise_map = {
        # "Pushups": "pushup.py",
        # "Pullups": "pull_up.py",
        # "Benchpress": "chestpress_final.py",
        "Pushups": "pushup.py",
            "Pullups": "pull_up.py",
            "Benchpress": "chestpress_final.py",
            "Situps": "crunch.py",
            "Squat": "squat.py",
            "Bicepcurls": "bicep_curl.py",
        # Add more here
    }

    for i, ex in enumerate(selected):
        script = exercise_map.get(ex)
        if script:
            subprocess.call(["python", script, sets, reps, rest])
            if i != len(selected) - 1:
                subprocess.call(["python", "rest_timer.py"])

    subprocess.Popen(["python", "thank_you_page.py"])

if __name__ == "__main__":
    run_exercises()
