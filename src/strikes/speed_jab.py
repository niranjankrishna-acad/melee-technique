import math
import mediapipe as mp
import sys
from collections import deque

mp_pose = mp.solutions.pose

class SpeedJab:
    def __init__(self):
        self.angle_history = deque(maxlen=10)  # Store last 10 angles
        self.perpendicular_history = deque(maxlen=10)  # Store last 10 angles


    def non_telegraphing(self, landmarks):
        shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
        elbow = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y)
        wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y)

        # 1. x axis - check angle
        vector_se = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
        vector_ew = (wrist[0] - elbow[0], wrist[1] - elbow[1])

        dot_product = vector_se[0] * vector_ew[0] + vector_se[1] * vector_ew[1]
        mag_se = math.sqrt(vector_se[0]**2 + vector_se[1]**2)
        mag_ew = math.sqrt(vector_ew[0]**2 + vector_ew[1]**2)
        cosine_angle = dot_product / (mag_se * mag_ew)
        angle = math.acos(cosine_angle)
        angle_degrees = math.degrees(angle)

        self.angle_history.append(angle_degrees)
        
        sys.stdout.flush()
        print("Last 10 Angles:", " ".join(f"{ang:.2f}" for ang in self.angle_history))

        #2. y axis - check perpendicular plane
        if wrist[0] != shoulder[0]:  # Avoid division by zero
            slope = (wrist[1] - shoulder[1]) / (wrist[0] - shoulder[0])
            intercept = shoulder[1] - slope * shoulder[0]
            
            # Calculate the perpendicular distance from elbow to the line
            perpendicular_distance = abs(slope * elbow[0] - elbow[1] + intercept) / math.sqrt(slope**2 + 1)
        else:  # Perfect vertical line
            perpendicular_distance = abs(elbow[0] - shoulder[0])

        self.perpendicular_history.append(perpendicular_distance)
        sys.stdout.flush()
        print("Last 10 Perpendicular:", " ".join(f"{per:.2f}" for per in self.perpendicular_history))
        
        return angle_degrees, perpendicular_distance

    def check(self, landmarks):
        angle, perpendicular = self.non_telegraphing(landmarks)
        print(f"Current Angle: {angle:.2f}, Current Perpendicular: {perpendicular:.2f}")