from src.pose_detector import PoseDetector
from src.cv2_utils import live_feed
from src.strikes.speed_jab import SpeedJab
from src.strikes.punch import Punch

def main():
    detector = PoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    jab_detector = SpeedJab()
    punch = Punch()
    technique_list = [punch]
    
    live_feed(detector, technique_list, "Mediapipe Feed", enable_3d_view=False)

if __name__ == "__main__":
    main()