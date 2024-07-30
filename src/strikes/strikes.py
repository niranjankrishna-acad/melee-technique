import numpy as np
import mediapipe as mp
from sklearn.cluster import KMeans
import pickle

PARENT_DIR = "src/strikes"

mp_pose = mp.solutions.pose

class Jab:
    def __init__(self):
        self.keypoints = []
        self.labels = ["Non-Jab", "Jab"]

    def select_landmarks(self, landmarks):
        # Select left shoulder, left elbow, and left wrist landmarks
        selected_landmarks = {
            'LEFT_SHOULDER': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            'LEFT_ELBOW': landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            'LEFT_WRIST': landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        }
        return selected_landmarks

    def normalize_landmarks(self, landmarks):
        # Normalize landmarks with respect to the left shoulder using unit normalization
        shoulder = landmarks['LEFT_SHOULDER']
        normalized = {}
        
        for keypoint, point in landmarks.items():
            vector = np.array([point.x - shoulder.x, point.y - shoulder.y, point.z - shoulder.z])
            magnitude = np.linalg.norm(vector)
            if magnitude == 0:  # To handle division by zero
                magnitude = 1
            normalized[keypoint] = {
                'x': vector[0] / magnitude,
                'y': vector[1] / magnitude,
                'z': vector[2] / magnitude
            }
        
        return normalized

    def record_landmarks(self, landmarks):
        selected = self.select_landmarks(landmarks)
        normalized = self.normalize_landmarks(selected)
        self.keypoints.append(normalized)

    def save_keypoints(self, filename=f'{PARENT_DIR}/data/jab_strikes.npy'):
        np.save(filename, self.keypoints)

    def extract_features(self):
        # Extract features (normalized coordinates of left shoulder, left elbow, and left wrist)
        features = []
        for data in self.keypoints:
            shoulder = data['LEFT_SHOULDER']
            elbow = data['LEFT_ELBOW']
            wrist = data['LEFT_WRIST']
            features.append([shoulder['x'], shoulder['y'], shoulder['z'],
                             elbow['x'], elbow['y'], elbow['z'],
                             wrist['x'], wrist['y'], wrist['z']])
        return np.array(features)

    def train_kmeans(self, n_clusters=2):
        features = self.extract_features()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        with open(f'{PARENT_DIR}/data/jab_kmeans.pkl', 'wb') as f:
            pickle.dump(kmeans, f)
        print("K-Means model trained and saved.")

    def load_kmeans(self):
        with open(f'{PARENT_DIR}/data/jab_kmeans.pkl', 'rb') as f:
            self.kmeans = pickle.load(f)

    def infer(self, landmarks):
        selected = self.select_landmarks(landmarks)
        normalized = self.normalize_landmarks(selected)
        features = np.array([[normalized['LEFT_SHOULDER']['x'], normalized['LEFT_SHOULDER']['y'], normalized['LEFT_SHOULDER']['z'],
                              normalized['LEFT_ELBOW']['x'], normalized['LEFT_ELBOW']['y'], normalized['LEFT_ELBOW']['z'],
                              normalized['LEFT_WRIST']['x'], normalized['LEFT_WRIST']['y'], normalized['LEFT_WRIST']['z']]])
        label = self.kmeans.predict(features)[0]
        return label

    def is_strike(self, label_window):
        prev_label, cur_label = label_window[-2], label_window[-1]
        if prev_label == "Non-Jab" and cur_label == "Jab":
            return True
        
        return False
