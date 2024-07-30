import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

# Load the recorded data and K-Means model
data = np.load('src/recorder/data/jab_strikes.npy', allow_pickle=True)
with open('src/recorder/data/jab_kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Extract features (normalized coordinates of left elbow and left wrist)
features = []
for record in data:
    elbow = record['LEFT_ELBOW']
    wrist = record['LEFT_WRIST']
    features.append([elbow['x'], elbow['y'], elbow['z'], wrist['x'], wrist['y'], wrist['z']])
features = np.array(features)

# Predict the cluster labels
labels = kmeans.predict(features)

# Plot the clusters in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define colors for the clusters
colors = ['r', 'g']
cluster_names = ["Non-Jab", "Jab"]

# Plot the elbow keypoints
for i in range(len(features)):
    elbow = features[i][:3]
    cluster_label = labels[i]
    ax.scatter(elbow[0], elbow[1], elbow[2], c=colors[cluster_label], label=cluster_names[cluster_label])

# Setting labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('K-Means Clustering of Jab Data')

# Create a legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label=cluster_names[i]) for i in range(len(cluster_names))]
ax.legend(handles=handles)

plt.show()
