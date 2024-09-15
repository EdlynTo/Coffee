import cv2
import dlib
import numpy as np
import time
import pyttsx3

def eye_aspect_ratio(eye):
    # Eye Aspect Ratio (EAR)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_eye_closure():
    engine = pyttsx3.init() # text to speech engine
    engine.setProperty('volume', 1)
    # Models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('/Users/edlynto/Documents/GitHub/coffee/shape_predictor_68_face_landmarks.dat')

    # Start webcam
    cap = cv2.VideoCapture(0)
    
    # Calculate num of frames in 1.5 seconds
    fps = cap.get(cv2.CAP_PROP_FPS)
    closure_frames = int(fps * 1.5)

    # Initialize variables for eye closure detection
    consecutive_closed_frames = 0
    EYE_CLOSED_FRAMES_THRESHOLD = closure_frames  # Eyes must be closed for this many frames
    EAR_THRESHOLD = 0.15  # EAR threshold to consider eyes closed
    sleepy_detected = False

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = detector(gray)

        if len(faces) > 0:  # If one face is detected
            face = faces[0]  # Use first detected face

            # Get face landmarks
            landmarks = predictor(gray, face)
            landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])

            # Get eye landmarks
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            # Average EAR for both eyes
            ear = (left_ear + right_ear) / 2.0

            # Check if EAR is below the threshold
            if ear < EAR_THRESHOLD:
                consecutive_closed_frames += 1
            else:
                consecutive_closed_frames = 0  # Reset the counter if eyes open
                sleepy_detected = False

            # If eyes are closed for more than the threshold duration (1.5 seconds)
            if consecutive_closed_frames >= EYE_CLOSED_FRAMES_THRESHOLD:
                cv2.putText(frame, 'Sleepy', (face.left(), face.top() - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                engine.say("Sleeper detected")
                engine.runAndWait()
                sleepy_detected = True
                print("WAKE UP")

            # Draw rectangles around the face
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

            # Draw landmarks
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Display the frame
        cv2.imshow('Eye Closure Detection', frame)

        # Exit video capture when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_eye_closure()