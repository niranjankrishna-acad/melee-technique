import numpy as np
import mediapipe as mp
from sklearn.cluster import KMeans
import pickle

mp_pose = mp.solutions.pose

class Jab:
    def __init__(self):
        self.keypoints = []

    def select_landmarks(self, landmarks):
        # Select left shoulder, left elbow, and left wrist landmarks
        selected_landmarks = {
            'LEFT_SHOULDER': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            'LEFT_ELBOW': landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            'LEFT_WRIST': landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        }
        return selected_landmarks

    def normalize_landmarks(self, landmarks):
        # Normalize landmarks with respect to the left shoulder
        shoulder = landmarks['LEFT_SHOULDER']
        normalized = {}
        
        for keypoint, point in landmarks.items():
            normalized[keypoint] = {
                'x': (point.x - shoulder.x) / shoulder.x,
                'y': (point.y - shoulder.y) / shoulder.y,
                'z': (point.z - shoulder.z) / shoulder.z
            }
        
        return normalized

    def record_landmarks(self, landmarks):
        selected = self.select_landmarks(landmarks)
        normalized = self.normalize_landmarks(selected)
        self.keypoints.append(normalized)

    def save_keypoints(self, filename='src/recorder/data/jab_strikes.npy'):
        np.save(filename, self.keypoints)

    def extract_features(self):
        # Extract features (normalized coordinates of left elbow and left wrist)
        features = []
        for data in self.keypoints:
            elbow = data['LEFT_ELBOW']
            wrist = data['LEFT_WRIST']
            features.append([elbow['x'], elbow['y'], elbow['z'], wrist['x'], wrist['y'], wrist['z']])
        return np.array(features)

    def train_kmeans(self, n_clusters=2):
        features = self.extract_features()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        with open('src/recorder/data/jab_kmeans.pkl', 'wb') as f:
            pickle.dump(kmeans, f)
        print("K-Means model trained and saved.")

    def load_kmeans(self):
        with open('src/recorder/data/jab_kmeans.pkl', 'rb') as f:
            self.kmeans = pickle.load(f)

    def infer(self, landmarks):
        selected = self.select_landmarks(landmarks)
        normalized = self.normalize_landmarks(selected)
        features = np.array([[normalized['LEFT_ELBOW']['x'], normalized['LEFT_ELBOW']['y'], normalized['LEFT_ELBOW']['z'],
                              normalized['LEFT_WRIST']['x'], normalized['LEFT_WRIST']['y'], normalized['LEFT_WRIST']['z']]])
        label = self.kmeans.predict(features)[0]
        return label
