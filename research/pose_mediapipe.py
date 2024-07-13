import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize the VideoCapture
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Perform pose detection
        results = pose.process(image)
    
        # Convert the image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw the pose annotations on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        # Check if pose landmarks are detected
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Landmark indices according to the MediaPipe pose landmark map
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
            
            # Print landmark coordinates
            print("Left Shoulder: (x:{:.2f}, y:{:.2f}, z:{:.2f})".format(left_shoulder.x, left_shoulder.y, left_shoulder.z))
            print("Right Shoulder: (x:{:.2f}, y:{:.2f}, z:{:.2f})".format(right_shoulder.x, right_shoulder.y, right_shoulder.z))
            print("Left Eye: (x:{:.2f}, y:{:.2f}, z:{:.2f})".format(left_eye.x, left_eye.y, left_eye.z))
            print("Right Eye: (x:{:.2f}, y:{:.2f}, z:{:.2f})".format(right_eye.x, right_eye.y, right_eye.z))

        # Display the resulting frame
        cv2.imshow('Mediapipe Feed', image)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
