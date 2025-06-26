def mountain_climbers(self, sets=3, reps_per_set=10, rest_time=30):
    for i in utilities().illustrate_exercise("TrainerImages/mountain_climber_illustraion.jpeg", "MOUNTAIN CLIMBERS"):
        yield i

    for current_set in range(1, sets + 1):
        print(f"Starting Set {current_set}/{sets}...")

        cap = cv2.VideoCapture(0)
        detector = pm.posture_detector()
        count = 0
        direction = 0
        start = time.process_time()

        while count < reps_per_set:
            success, img = cap.read()
            if not success:
                break

            img = detector.find_person(img, False)
            landmark_list = detector.find_landmarks(img, False)

            if landmark_list:
                left_arm_angle = detector.find_angle(img, 11, 13, 15)
                right_arm_angle = detector.find_angle(img, 12, 14, 16)
                left_leg_angle = detector.find_angle(img, 24, 26, 28)
                right_leg_angle = detector.find_angle(img, 23, 25, 27)

                per = np.interp(right_leg_angle, (220, 280), (0, 100))
                bar_pos = np.interp(right_leg_angle, (220, 280), (img.shape[0] * 0.9, img.shape[0] * 0.15))
                color = utilities().get_performance_bar_color(per)

                if per == 100 or per == 0:
                    color = (0, 255, 0)
                    rep = utilities().repitition_counter(per, count, direction)
                    count = rep["count"]
                    direction = rep["direction"]

                utilities().draw_performance_bar(img, per, bar_pos, color, count)

            utilities().display_rep_count(img, count, reps_per_set)
            cv2.putText(img, f"Set {current_set}/{sets}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
            cv2.putText(img, "Maintain core tightness!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)

            _, jpeg = cv2.imencode('.jpg', img)
            yield b'--frame\r\n\r\n' + jpeg.tobytes() + b'\r\n'

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()

        time_elapsed = int(time.process_time() - start)
        calories_burned = (time_elapsed / 60) * ((4.0 * 3.5 * 64) / 200)
        print(f"Set {current_set} complete. Estimated calories burned: {calories_burned:.2f}")

        if current_set < sets:
            for remaining in range(rest_time, 0, -1):
                rest_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(rest_img, f"Rest: {remaining}s", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
                _, jpeg = cv2.imencode('.jpg', rest_img)
                yield b'--frame\r\n\r\n' + jpeg.tobytes() + b'\r\n'
                time.sleep(1)

    print("Workout Complete!")
