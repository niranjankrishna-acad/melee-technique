from src.pose_detector import PoseDetector
from src.cv2_utils import live_feed
from src.strikes.speed_jab import SpeedJab

def render_pose(detector: PoseDetector, frame):
    return detector.process_frame_and_landmarks(frame)

def main():
    detector = PoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    jab_detector = SpeedJab()
    technique_list = [jab_detector]
    
    live_feed(render_pose, detector, technique_list, "Mediapipe Feed")

if __name__ == "__main__":
    main()