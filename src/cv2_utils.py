import cv2
from src.pose_detector import PoseDetector

def live_feed(render_pose, detector: PoseDetector, technique_list, window_name="Video Feed"):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        processed_frame, landmarks = render_pose(detector, frame)
        if landmarks:
            for technique in technique_list:
                technique.check(landmarks)

        cv2.imshow(window_name, processed_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if hasattr(detector, 'release_resources'):
        detector.release_resources()
