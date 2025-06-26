from ExercisesModule import simulate_target_exercies

exercise = simulate_target_exercies(difficulty_level=1, reps=3)

for frame in exercise.push_ups():
    # This frame is in MJPEG format, decode and display:
    import numpy as np
    import cv2
    jpg_bytes = frame.split(b'\r\n\r\n')[1].split(b'\r\n')[0]
    jpg_array = np.frombuffer(jpg_bytes, dtype=np.uint8)
    img = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
    cv2.imshow("Push Ups Live Test", img)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
