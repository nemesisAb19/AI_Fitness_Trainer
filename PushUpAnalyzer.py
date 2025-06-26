import cv2
import time
import numpy as np
from ExercisesModule import utilities
import PoseModule as pm
import face_detection as face_det  # Ensure this is your face direction module

class PushUpAnalyzer:
    def __init__(self, reps=10, difficulty_level=1):
        self.target_reps = reps * difficulty_level
        self.count = 0
        self.direction = 0
        self.detector = pm.posture_detector()
        self.utility = utilities()
        self.start_time = time.time()

    def estimate_pose(self, img):
        img = self.detector.find_person(img, draw=False)
        landmarks = self.detector.find_landmarks(img, draw=False)
        return img, landmarks

    def calculate_joint_angles(self, img, lm):
        angles = {}
        try:
            angles['right_elbow'] = self.detector.find_angle(img, 12, 14, 16)  # shoulder, elbow, wrist
            angles['right_hip'] = self.detector.find_angle(img, 24, 26, 28)    # hip, knee, ankle
            angles['right_leg'] = self.detector.find_angle(img, 26, 28, 32)    # knee, ankle, foot
        except:
            angles['right_elbow'] = angles['right_hip'] = angles['right_leg'] = 0
        return angles

    def detect_pushup_progress(self, elbow_angle):
        percent = np.interp(elbow_angle, (60, 160), (0, 100))
        bar = np.interp(elbow_angle, (60, 160), (650, 100))
        return percent, bar

    def count_reps(self, percent):
        data = self.utility.repitition_counter(percent, self.count, self.direction)
        self.count = data['count']
        self.direction = data['direction']

    def validate_direction(self, lm):
        try:
            shoulder_x1 = lm[12][1]
            shoulder_x2 = lm[11][1]
            waist_x = lm[24][1]
            return face_det.is_in_right_direction(None, shoulder_x1, shoulder_x2, waist_x)
        except:
            return False

    def draw_visuals(self, img, angles, percent, bar, direction_valid):
        color = (0, 255, 0) if percent in [0, 100] else self.utility.get_performance_bar_color(percent)

        # Overlay push-up status
        self.utility.draw_performance_bar(img, percent, bar, color, self.count)
        self.utility.display_rep_count(img, self.count, self.target_reps)
        self.utility.position_info_floor_exercise(img, direction_valid)

        # Angle visualization
        cv2.putText(img, f"{int(angles['right_elbow'])} degrees", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return img

    def display_calories(self):
        duration = int(time.time() - self.start_time)
        calories = (duration / 60) * ((4.0 * 3.5 * 64) / 200)
        print(f"Workout Duration: {duration}s | Calories Burned: {calories:.2f}")

    def run(self):
        cap = cv2.VideoCapture(0)

        while self.count < self.target_reps:
            ret, img = cap.read()
            if not ret:
                break

            img, lm = self.estimate_pose(img)
            if not lm:
                continue

            angles = self.calculate_joint_angles(img, lm)
            percent, bar = self.detect_pushup_progress(angles['right_elbow'])
            self.count_reps(percent)
            direction_valid = self.validate_direction(lm)
            img = self.draw_visuals(img, angles, percent, bar, direction_valid)

            cv2.imshow("Push-Up Tracker", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.display_calories()
