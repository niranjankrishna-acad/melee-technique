import math
import mediapipe as mp
from collections import deque
from src.strikes.utils import Landmark

mp_pose = mp.solutions.pose

class Punch:
    def __init__(self, punch_threshold=0.8, delta_threshold=0.1, window_size=10):
        self.punch_threshold = punch_threshold
        self.delta_threshold = delta_threshold
        self.current_threshold = 0
        self.wrist_positions = deque(maxlen=window_size)
        self.punch_count = 0

    def check(self, landmarks):
        shoulder = Landmark.from_landmark(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        wrist = Landmark.from_landmark(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

        relative_wrist = Landmark.sub(shoulder, wrist)
        
        if len(self.wrist_positions) == 0:
            self.wrist_positions.append(relative_wrist)
        else:
            prev_relative_wrist = self.wrist_positions[-1]
            relative_wrist_movement = Landmark.distz(prev_relative_wrist, relative_wrist)

            if relative_wrist_movement > self.delta_threshold:
                self.current_threshold += relative_wrist_movement
            elif relative_wrist_movement < -self.delta_threshold:
                self.current_threshold = 0

        # Check if the current threshold exceeds the punch threshold
        if self.current_threshold >= self.punch_threshold:
            self.punch_count +=1
            print("Punch detected!", self.punch_count)
            self.current_threshold = 0
            return True

        return False
