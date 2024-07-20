import math
import mediapipe as mp
import sys

mp_pose = mp.solutions.pose

class SpeedJab:
    def non_telegraphing(self, landmarks, stability_threshold=5, line_threshold=10):
        shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
        elbow = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y)
        wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y)

        initial_distance = math.hypot(wrist[0] - shoulder[0], wrist[1] - shoulder[1])
        final_distance = initial_distance  # Adjust based on your needs

        distance_change = abs(initial_distance - final_distance)

        A = wrist[1] - shoulder[1]
        B = shoulder[0] - wrist[0]
        C = wrist[0] * shoulder[1] - shoulder[0] * wrist[1]

        elbow_distance_to_line = abs(A * elbow[0] + B * elbow[1] + C) / math.sqrt(A**2 + B**2)

        # Clearing previous output
        sys.stdout.flush()
        
        print(f"Initial Distance: {initial_distance:.3f}")
        print(f"Distance Change: {distance_change:.3f} (Threshold: {stability_threshold})")
        print(f"Elbow Distance to Line: {elbow_distance_to_line:.3f} (Threshold: {line_threshold})")

        if distance_change > stability_threshold or elbow_distance_to_line > line_threshold:
            return False

        return True
    
    def check(self, landmarks):
        telegraph_bool = self.non_telegraphing(landmarks)
        print(f"Collinearity: {telegraph_bool}")
    
    def check(self, landmarks):
        telegraph_bool = self.non_telegraphing(landmarks)
        print(f"Collinearity: {telegraph_bool}")