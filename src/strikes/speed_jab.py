import math
import mediapipe as mp
import sys
from collections import deque
from src.strikes.utils import Landmark
mp_pose = mp.solutions.pose

class SpeedJab:
    def __init__(self):
        self.angle_history = deque(maxlen=10)  # Store last 10 angles
        self.perpendicular_history = deque(maxlen=10)  # Store last 10 angles

    def non_telegraphing(self, landmarks):
        shoulder = Landmark.from_landmark(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        elbow = Landmark.from_landmark(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
        wrist = Landmark.from_landmark(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

        # 1. x axis - check angle
        vector_se = (elbow.x - shoulder.x, elbow.y - shoulder.y)
        vector_ew = (wrist.x - elbow.x, wrist.y - elbow.y)

        dot_product = vector_se[0] * vector_ew[0] + vector_se[1] * vector_ew[1]
        mag_se = math.sqrt(vector_se[0]**2 + vector_se[1]**2)
        mag_ew = math.sqrt(vector_ew[0]**2 + vector_ew[1]**2)
        cosine_angle = dot_product / (mag_se * mag_ew)
        angle = math.acos(cosine_angle)
        angle_degrees = math.degrees(angle)

        self.angle_history.append(angle_degrees)
        
        sys.stdout.flush()
        print("Last 10 Angles:", " ".join(f"{ang:.2f}" for ang in self.angle_history))

       # 2. y axis - check perpendicular plane
        shoulder_xz = (shoulder.x, shoulder.z)
        elbow_xz = (elbow.x, elbow.z)
        wrist_xz = (wrist.x, wrist.z)

        # Slope and intercept of the line in XZ plane
        if wrist_xz[0] == shoulder_xz[0]:  # Avoid division by zero
            y_displacement = abs(elbow_xz[0] - shoulder_xz[0])
        else:
            slope = (wrist_xz[1] - shoulder_xz[1]) / (wrist_xz[0] - shoulder_xz[0])
            intercept = shoulder_xz[1] - slope * shoulder_xz[0]

            # Perpendicular distance from the point to the line
            y_displacement = abs(slope * elbow_xz[0] - elbow_xz[1] + intercept) / math.sqrt(slope**2 + 1)

        self.perpendicular_history.append(y_displacement)
        sys.stdout.flush()
        print("Last 10 Perpendicular:", " ".join(f"{per:.2f}" for per in self.perpendicular_history))
        
        return angle_degrees, y_displacement

    def check(self, landmarks):
        angle, perpendicular = self.non_telegraphing(landmarks)
        print(f"Current Angle: {angle:.2f}, Current Perpendicular: {perpendicular:.2f}")