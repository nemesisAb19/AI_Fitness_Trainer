## import packages
import pandas as pd
import numpy as np
import cv2
from utils import *
import argparse
import mediapipe as mp

class BodyPartAngle:
    def __init__(self, landmarks):
        self.landmarks = landmarks

    def angle_of_the_left_arm(self):
        l_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        l_elbow = detection_body_part(self.landmarks, "LEFT_ELBOW")
        l_wrist = detection_body_part(self.landmarks, "LEFT_WRIST")
        return calculate_angle(l_shoulder, l_elbow, l_wrist)

    def angle_of_the_right_arm(self):
        r_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
        r_elbow = detection_body_part(self.landmarks, "RIGHT_ELBOW")
        r_wrist = detection_body_part(self.landmarks, "RIGHT_WRIST")
        return calculate_angle(r_shoulder, r_elbow, r_wrist)

    def angle_of_the_left_leg(self):
        l_hip = detection_body_part(self.landmarks, "LEFT_HIP")
        l_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
        l_ankle = detection_body_part(self.landmarks, "LEFT_ANKLE")
        return calculate_angle(l_hip, l_knee, l_ankle)

    def angle_of_the_right_leg(self):
        r_hip = detection_body_part(self.landmarks, "RIGHT_HIP")
        r_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        r_ankle = detection_body_part(self.landmarks, "RIGHT_ANKLE")
        return calculate_angle(r_hip, r_knee, r_ankle)

    def angle_of_the_neck(self):
        r_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
        l_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        r_mouth = detection_body_part(self.landmarks, "MOUTH_RIGHT")
        l_mouth = detection_body_part(self.landmarks, "MOUTH_LEFT")
        r_hip = detection_body_part(self.landmarks, "RIGHT_HIP")
        l_hip = detection_body_part(self.landmarks, "LEFT_HIP")

        shoulder_avg = [(r_shoulder[0] + l_shoulder[0]) / 2,
                        (r_shoulder[1] + l_shoulder[1]) / 2]
        mouth_avg = [(r_mouth[0] + l_mouth[0]) / 2,
                     (r_mouth[1] + l_mouth[1]) / 2]
        hip_avg = [(r_hip[0] + l_hip[0]) / 2, (r_hip[1] + l_hip[1]) / 2]

        return abs(180 - calculate_angle(mouth_avg, shoulder_avg, hip_avg))

    def angle_of_the_abdomen(self):
        # calculate angle of the avg shoulder
        r_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
        l_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        shoulder_avg = [(r_shoulder[0] + l_shoulder[0]) / 2,
                        (r_shoulder[1] + l_shoulder[1]) / 2]

        # calculate angle of the avg hip
        r_hip = detection_body_part(self.landmarks, "RIGHT_HIP")
        l_hip = detection_body_part(self.landmarks, "LEFT_HIP")
        hip_avg = [(r_hip[0] + l_hip[0]) / 2, (r_hip[1] + l_hip[1]) / 2]

        # calculate angle of the avg knee
        r_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        l_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
        knee_avg = [(r_knee[0] + l_knee[0]) / 2, (r_knee[1] + l_knee[1]) / 2]

        return calculate_angle(shoulder_avg, hip_avg, knee_avg)


class TypeOfExercise(BodyPartAngle):
    def __init__(self, landmarks):
        super().__init__(landmarks)

    def push_up(self, counter, status):
        left_arm_angle = self.angle_of_the_left_arm()
        right_arm_angle = self.angle_of_the_left_arm()
        avg_arm_angle = (left_arm_angle + right_arm_angle) // 2

        if status:
            if avg_arm_angle < 70:
                counter += 1
                status = False
        else:
            if avg_arm_angle > 160:
                status = True

        return [counter, status]

    # def push_up_method_2():

    def pull_up(self, counter, status):
        nose = detection_body_part(self.landmarks, "NOSE")
        left_elbow = detection_body_part(self.landmarks, "LEFT_ELBOW")
        right_elbow = detection_body_part(self.landmarks, "RIGHT_ELBOW")
        avg_shoulder_y = (left_elbow[1] + right_elbow[1]) / 2

        if status:
            if nose[1] > avg_shoulder_y:
                counter += 1
                status = False

        else:
            if nose[1] < avg_shoulder_y:
                status = True

        return [counter, status]

    def squat(self, counter, status):
        left_leg_angle = self.angle_of_the_right_leg()
        right_leg_angle = self.angle_of_the_left_leg()
        avg_leg_angle = (left_leg_angle + right_leg_angle) // 2

        if status:
            if avg_leg_angle < 70:
                counter += 1
                status = False
        else:
            if avg_leg_angle > 160:
                status = True

        return [counter, status]

    def walk(self, counter, status):
        right_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        left_knee = detection_body_part(self.landmarks, "LEFT_KNEE")

        if status:
            if left_knee[0] > right_knee[0]:
                counter += 1
                status = False

        else:
            if left_knee[0] < right_knee[0]:
                counter += 1
                status = True

        return [counter, status]

    def sit_up(self, counter, status):
        angle = self.angle_of_the_abdomen()
        if status:
            if angle < 55:
                counter += 1
                status = False
        else:
            if angle > 105:
                status = True

        return [counter, status]

    def curl(self, counter, status):
        left_arm_angle = self.angle_of_the_left_arm()
        right_arm_angle = self.angle_of_the_right_arm()
        avg_arm_angle = (left_arm_angle + right_arm_angle) // 2

        if status:
            if right_arm_angle > 160 or left_arm_angle > 160:
                stage = "down"
            if right_arm_angle < 90 and left_arm_angle < 90 and stage == "down":
                counter += 1
                status = False
                stage = "up"

        else:
            if avg_arm_angle > 30:
                counter += 1
                status = True

        return [counter, status]

    def calculate_exercise(self, exercise_type, counter, status):
        if exercise_type == "push-up":
            counter, status = TypeOfExercise(self.landmarks).push_up(
                counter, status)
        elif exercise_type == "pull-up":
            counter, status = TypeOfExercise(self.landmarks).pull_up(
                counter, status)
        elif exercise_type == "squat":
            counter, status = TypeOfExercise(self.landmarks).squat(
                counter, status)
        elif exercise_type == "walk":
            counter, status = TypeOfExercise(self.landmarks).walk(
                counter, status)
        elif exercise_type == "sit-up":
            counter, status = TypeOfExercise(self.landmarks).sit_up(
                counter, status)
        elif exercise_type == "curl":
            counter, status = TypeOfExercise(self.landmarks).curl(
                counter, status)

        return [counter, status]

def push():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

## setting the video source

    cap = cv2.VideoCapture("push-up.mp4")

#cap = cv2.VideoCapture(0)  # webcam

    cap.set(3, 800)  # width
    cap.set(4, 480)  # height

## setup mediapipe
    with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

        counter = 0  # movement of exercise
        status = True  # state of move
        while cap.isOpened():
            try:
                ret, frame = cap.read()
            # result_screen = np.zeros((250, 400, 3), np.uint8)

                frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_AREA)
            ## recolor frame to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
            ## make detection
                results = pose.process(frame)
            ## recolor back to BGR
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark
                    counter, status = TypeOfExercise(landmarks).calculate_exercise(
                    "push-up", counter, status)
                except:
                    pass

                score_table("push-up", counter, status)

            ## render detections (for landmarks)
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255),
                                    thickness=2,
                                    circle_radius=2),
                    mp_drawing.DrawingSpec(color=(174, 139, 45),
                                    thickness=2,
                                    circle_radius=2),
                )

                cv2.imshow('Video', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("counter: " + str(counter))
                    break
            except:
                print("err counter: " + str(counter))
                break

        cap.release()
        cv2.destroyAllWindows()
