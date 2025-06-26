def dumbbell_row(self):
    cap = cv2.VideoCapture(0)
    detector = posemodule.posture_detector()
    count = 0
    direction = 0

    bar = 650
    per = 0

    utilities().illustrate_exercise("TrainerImages/dumbbell_row.jpg", "DUMBBELL ROW")

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = detector.find_person(img, draw=False)
        lmList = detector.find_landmarks(img, draw=True)

        if lmList:
            # Updated: use hip (24), shoulder (12), and elbow (14) for dumbbell row motion
            angle = detector.find_angle(img, 24, 12, 14)

            # Adjusted angle thresholds for dumbbell row motion
            per = np.interp(angle, (60, 150), (100, 0))
            bar = np.interp(angle, (60, 150), (150, 400))

            result = utilities().repitition_counter(per, count, direction)
            count = result["count"]
            direction = result["direction"]

            # Correct angle range for row posture
            isCorrect = 60 < angle < 150

            color = utilities().get_performance_bar_color(per)
            utilities().draw_performance_bar(img, per, bar, color, count)
            utilities().display_rep_count(img, count, total_reps=10)
            utilities().position_info_standing_exercise(img, isCorrect)

        cv2.imshow("Dumbbell Row", img)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
