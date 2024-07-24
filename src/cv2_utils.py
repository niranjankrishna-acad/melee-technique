import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.pose_detector import PoseDetector
import mediapipe as mp

mp_pose = mp.solutions.pose

def live_feed(render_pose, detector: PoseDetector, technique_list, window_name="Video Feed", enable_3d_view=False):
    cap = cv2.VideoCapture(0)
    fig, ax = None, None

    if enable_3d_view:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=20, azim=-45)  # Set the view angle
        plt.ion()  # Turn on interactive mode
        plt.show()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        processed_frame, landmarks = render_pose(detector, frame)
        if landmarks:
            for technique in technique_list:
                technique.check(landmarks)

            if enable_3d_view:
                update_3d_view(ax, landmarks)

        cv2.imshow(window_name, processed_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if enable_3d_view:
        plt.ioff()  # Turn off interactive mode
    if hasattr(detector, 'release_resources'):
        detector.release_resources()

def update_3d_view(ax, landmarks):
    shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    elbow_left = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    wrist_left = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

    # Anchor around the shoulder and flip y and z coordinates for inversion
    xs = [0, elbow_left.x - shoulder_left.x, wrist_left.x - shoulder_left.x]
    ys = [0, -(elbow_left.y - shoulder_left.y), -(wrist_left.y - shoulder_left.y)]
    zs = [0, -(elbow_left.z - shoulder_left.z), -(wrist_left.z - shoulder_left.z)]

    ax.clear()
    ax.scatter(xs, ys, zs, c='r', marker='o')
    ax.plot(xs, ys, zs, label='Left Arm')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.draw()
    plt.pause(0.001)  # Pause for a brief moment to update the plot
