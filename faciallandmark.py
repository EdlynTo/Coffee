import cv2
import dlib
import numpy as np
import time
import pygame

import os

# Initialize pygame
pygame.init()

# Get screen dimensions
screen_info = pygame.display.Info()
screen_width = screen_info.current_w
screen_height = screen_info.current_h

# Calculate position for bottom right corner
window_width, window_height = 100, 100
x_position = screen_width - window_width
y_position = screen_height - window_height

# Set the environment variable to position the window
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x_position},{y_position}"

# Create the window
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('Bottom Right Window')

# Run the Pygame window loop
running = True



def eye_aspect_ratio(eye):
    # Eye Aspect Ratio (EAR)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def detect_eye_closure():
    # Load face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('/Users/edlynto/Documents/GitHub/coffee/shape_predictor_68_face_landmarks.dat')

    # Start webcam
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    closure_time_threshold = 1 # Eye closure duration in seconds

    # Thresholds and variables
    EAR_THRESHOLD = 0.15
    eyes_closed_start_time = None  # Track the time when eyes are first closed
    wake_up_printed = False 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(gray, face)
            landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])

            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            if ear < EAR_THRESHOLD:
                # Eyes are closed
                if eyes_closed_start_time is None:
                    eyes_closed_start_time = time.time()  # Record the time when eyes first closed
                elapsed_time = time.time() - eyes_closed_start_time

                if elapsed_time >= closure_time_threshold and not wake_up_printed:
                    print("WAKE UP")
                    window.fill((255,255,255))
                    pygame.display.update() 
                    wake_up_printed = True  # Ensure "WAKE UP" is printed only once
            else:
                eyes_closed_start_time = None  # Reset the closed time when eyes open
                wake_up_printed = False  # Reset the flag to allow for future "WAKE UP" detection
                # print("Eyes opened")
                window.fill((0,0,0))
                pygame.display.update() 

            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

        cv2.imshow('Eye Closure Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while running:
        # Quit pygame
        detect_eye_closure()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()



