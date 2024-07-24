import math
import mediapipe as mp
from collections import deque
from src.strikes.utils import Landmark

mp_pose = mp.solutions.pose

class Punch:
    def __init__(self, punch_threshold=1, delta_threshold=0.2, window_size=10):
        self.punch_threshold = punch_threshold
        self.delta_threshold = delta_threshold
        self.current_threshold = 0
        self.wrist_positions = deque(maxlen=window_size)

    def check(self, landmarks):
        shoulder = Landmark.from_landmark(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        wrist = Landmark.from_landmark(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

        relative_wrist = Landmark.sub(shoulder, wrist)
        
        if len(self.wrist_positions) == 0:
            self.wrist_positions.append(relative_wrist)
        else:
            prev_relative_wrist = self.wrist_positions[-1]
            relative_wrist_movement = Landmark.distz(prev_relative_wrist, relative_wrist)

            print(relative_wrist_movement)
            if relative_wrist_movement > self.delta_threshold:
                self.current_threshold += relative_wrist_movement
            elif relative_wrist_movement < -self.delta_threshold:
                self.current_threshold = 0

        print("Threshold", self.current_threshold)
        # Check if the current threshold exceeds the punch threshold
        if self.current_threshold >= self.punch_threshold:
            print("Punch detected!")
            self.current_threshold = 0
            return True

        return False
