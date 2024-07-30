import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import numpy as np
import mediapipe as mp
from src.pose_detector import PoseDetector
from src.strikes import technique_dict

mp_pose = mp.solutions.pose

class App:
    def __init__(self, window, window_title, detector, technique_list, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.detector: PoseDetector = detector
        self.technique_list = technique_list

        self.vid = cv2.VideoCapture(self.video_source)
        self.recording = False
        self.recording_time = 30  # 30 seconds
        self.countdown = 3
        self.punch_count = 0
        self.prev_label = None  # To store the previous label for transition detection

        # Recorder for the selected technique
        self.technique= None
        self.current_label = None

        # Create a canvas for the video feed
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.grid(row=0, column=0)

        # GUI controls on the right
        self.controls_frame = tk.Frame(window)
        self.controls_frame.grid(row=0, column=1, padx=10, pady=10)

        # Dropdown list
        self.technique_list_var = tk.StringVar()
        self.dropdown = ttk.Combobox(self.controls_frame, textvariable=self.technique_list_var)
        self.dropdown['values'] = list(technique_dict.keys())
        self.dropdown.current(0)
        self.dropdown.grid(row=0, column=0, pady=5)
        self.technique_list_var.trace('w', self.update_technique)

        # Record button
        self.record_button = tk.Button(self.controls_frame, text="Start 30 sec Record", command=self.start_recording)
        self.record_button.grid(row=1, column=0, pady=5)

        # Button to train K-Means model
        self.train_button = tk.Button(self.controls_frame, text="Train K-Means", command=self.train_kmeans)
        self.train_button.grid(row=2, column=0, pady=5)

        # Button to perform inference
        self.infer_button = tk.Button(self.controls_frame, text="Start Inference", command=self.start_inference)
        self.infer_button.grid(row=3, column=0, pady=5)
        self.inference_mode = False

        # Recording status
        self.status_label = tk.Label(self.controls_frame, text="Status: Idle")
        self.status_label.grid(row=4, column=0, pady=5)

        # Punch counter
        self.punch_counter_label = tk.Label(self.controls_frame, text="Punch Count: 0")
        self.punch_counter_label.grid(row=5, column=0, pady=5)

        self.update_technique()
        self.update()
        self.window.mainloop()

    def update_technique(self, *args):
        selected_technique = self.technique_list_var.get()
        self.technique = technique_dict[selected_technique]()

    def start_recording(self):
        self.recording = True
        self.status_label.config(text="Recording in...")
        self.window.after(1000, self.countdown_timer)

    def countdown_timer(self):
        if self.countdown > 0:
            self.status_label.config(text=f"Recording in... {self.countdown}")
            self.countdown -= 1
            self.window.after(1000, self.countdown_timer)
        else:
            self.status_label.config(text="Recording...")
            threading.Thread(target=self.record).start()

    def record(self):
        self.status_label.config(text=f"Recording... {self.recording_time}")
        start_time = time.time()

        while time.time() - start_time < self.recording_time:
            ret, frame = self.vid.read()
            if not ret:
                break
            frame, landmarks = self.detector.process_frame_and_landmarks(frame)
            if landmarks:
                self.technique.record_landmarks(landmarks)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.status_label.config(text=f"Recording... {int(self.recording_time - (time.time() - start_time))}")
            self.window.update_idletasks()
            self.window.update()

        self.technique.save_keypoints()
        self.status_label.config(text="Status: Recording Completed")
        self.recording = False
        self.countdown = 3

    def train_kmeans(self):
        if self.technique:
            self.status_label.config(text="Training...")
            threading.Thread(target=self.run_kmeans_training).start()
        else:
            print("No technique recorder selected.")

    def run_kmeans_training(self):
        self.technique.train_kmeans()
        self.status_label.config(text="Training Complete")

    def start_inference(self):
        if self.technique:
            self.status_label.config(text="Loading K-Means Model...")
            self.technique.load_kmeans()
            self.status_label.config(text="Inference Mode")
            self.inference_mode = True
        else:
            print("No technique recorder selected.")

    def update(self):
        if not self.recording:
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.inference_mode:
                    _, landmarks = self.detector.process_frame_and_landmarks(frame)
                    if landmarks:
                        label = self.technique.infer(landmarks)
                        self.current_label = self.technique.labels[label]
                        print(f"Inference label: {self.current_label}")  # Debug print
                        cv2.putText(frame, self.current_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        is_strike = self.technique.is_strike([self.prev_label, self.current_label])
                        # Punch counter logic: count a punch when transitioning from "Non-Jab" to "Jab"
                        if is_strike:
                            self.punch_count += 1
                            self.punch_counter_label.config(text=f"Punch Count: {self.punch_count}")

                        self.prev_label = self.current_label

                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.window.after(10, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

def main():
    detector = PoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    
    root = tk.Tk()
    App(root, "Pose Detection with GUI", detector, [])

if __name__ == "__main__":
    main()
