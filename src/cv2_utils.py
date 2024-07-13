import cv2
from src.pose_detector import PoseDetector

def live_feed(process_frame, detector:PoseDetector, window_name="Video Feed"):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        processed_frame = process_frame(detector, frame)
        cv2.imshow(window_name, processed_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if hasattr(detector, 'release_resources'):
        detector.release_resources()
