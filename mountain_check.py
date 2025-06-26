from ExercisesModule import simulate_target_exercies
import numpy as np
import cv2

# Initialize the exercise with desired difficulty and reps
exercise = simulate_target_exercies(difficulty_level=1, reps=3)

# Start the generator for mountain climbers
mountain_climber_gen = exercise.mountain_climbers()

# Skip illustration frames (optional)
for frame in mountain_climber_gen:
    img = cv2.imdecode(np.frombuffer(frame.split(b'\r\n\r\n')[1].split(b'\r\n')[0], np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow("Mountain Climbers - Illustration", img)
    if cv2.waitKey(1000) & 0xFF == ord('s'):  # Press 's' to skip early
        break
cv2.destroyAllWindows()

# Re-run the generator to get real-time frames now
mountain_climber_gen = exercise.mountain_climbers()

# Skip illustration frames again (this time for real-time part)
for frame in mountain_climber_gen:
    # Only show frames that contain JPEG images
    if isinstance(frame, bytes) and frame.startswith(b'--frame'):
        jpg_bytes = frame.split(b'\r\n\r\n')[1].split(b'\r\n')[0]
        jpg_array = np.frombuffer(jpg_bytes, dtype=np.uint8)
        img = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
        cv2.imshow("Mountain Climbers - Live", img)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
