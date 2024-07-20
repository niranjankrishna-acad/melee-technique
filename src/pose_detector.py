import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=min_detection_confidence, 
                                      min_tracking_confidence=min_tracking_confidence)

    def process_landmarks(self, frame):
        results = self._process(frame)
        landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
        return landmarks

    def process_frame(self, frame):
        results = self._process(frame)
        image = self._draw_landmarks(frame, results.pose_landmarks)
        return image

    def process_frame_and_landmarks(self, frame):
        results = self._process(frame)
        image = self._draw_landmarks(frame, results.pose_landmarks)

        landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
        return image, landmarks

    def _process(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        return results

    def _draw_landmarks(self, frame, landmarks):
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image.flags.writeable = True
        self.mp_drawing.draw_landmarks(
            image,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        return image

    def release_resources(self):
        self.pose.close()
