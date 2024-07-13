from src.pose_detector import PoseDetector
from src.cv2_utils import live_feed

def render_pose(detector, frame):
    return detector.process_frame(frame)

def main():
    detector = PoseDetector()
    live_feed(render_pose, detector, "Mediapipe Feed")

if __name__ == "__main__":
    main()
