import cv2
import time
import threading
import torch
import os
from IPython.display import display
import mediapipe as mp
from base64 import b64encode
from IPython.display import HTML
from Mivolo_Sys import RecommendationSystem
from utils import compare_face_distances, calculate_face_angle
import subprocess
import argparse

args = argparse.ArgumentParser()
args.add_argument("--model_path", type=str, default="weights/best_model_weights_10.pth")
args.add_argument("--detection_path", type=str, default="weights/yolov8x_person_face.pt")
args.add_argument("--save_path", type=str, default="image_database/captured_face.jpg")
args = args.parse_args()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class FaceMeshRecommendationSystem:
    def __init__(self, model_path, detection_path, save_path, 
                 sim_threshold = {}, process_every_n_frames=1, delay=4, threshold=70,
                 device="cuda", distance = ['cosine', 'manhattan'], straight_threshold=0.35):
        self.model_path = model_path
        self.detection_path = detection_path
        self.save_path = save_path
        self.defaut_sim_threshold = sim_threshold
        self.sim_threshold = {key: sim_threshold[key] for key in distance}
        self.process_every_n_frames = process_every_n_frames
        self.delay = delay
        self.threshold = threshold

        self.device = device
        self.face_mesh_lost = False
        self.last_time_face_lost = 0
        self.face_captured = False
        self.face_count = 0
        self.distance = distance

        self.system = RecommendationSystem(
            model_path=self.model_path,
            detection_path=self.detection_path,
            device=self.device
        )
        self.straight_threshold = straight_threshold
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh

        print(f"System initialized. Running model on {self.device} system.")

    def infer_recommendation(self, face_image):
        start_time = time.time()
        if os.path.exists(self.save_path):
            previous_face = cv2.imread(self.save_path)
            similarity_score = compare_face_distances(previous_face, face_image, distance=self.distance, hamming_threshold=self.defaut_sim_threshold["hamming"])
            print(f"Threshold       : {[f'{key}: {self.sim_threshold[key]:.4f}' for key in self.sim_threshold]}")
            print(f"Similarity Score: {[f'{key}: {similarity_score[key]:.4f}' for key in similarity_score]}")
            if all([similarity_score[key] > self.sim_threshold[key] for key in similarity_score]):
                print("Similarity Scores > Thresholds. Capturing object.....")
                cv2.imwrite(self.save_path, face_image)
                gender, age = self.system.infer_image(self.save_path)
                print(f"Predicted Gender: {gender}, Predicted Age: {age}")
                print("----------------------------------------------------------------------------------")
            else:
                print("Similarity Scores < Thresholds. Ignoring object.....")
                print("----------------------------------------------------------------------------------")
        else:
            cv2.imwrite(self.save_path, face_image)
            gender, age = self.system.infer_image(self.save_path)
            print(f"Predicted Gender: {gender}, Predicted Age: {age}")
        end_time = time.time()
        print(f"Recommendation inference time: {end_time - start_time:.4f} seconds")

    def run(self):
        cap = cv2.VideoCapture(0)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('demos/output.avi', fourcc, 20.0, (frame_width, frame_height))

        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            
            while cap.isOpened():
                iteration_start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    print("Ignoring empty camera frame.")
                    continue

                self.face_count += 1

                if self.face_mesh_lost and time.time() - self.last_time_face_lost < self.delay:
                    cv2.imshow('Camera', cv2.flip(frame, 1))
                    out.write(frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                if self.face_count % self.process_every_n_frames != 0:
                    continue

                original_frame = frame.copy()
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = face_mesh.process(frame)
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if results.multi_face_landmarks:
                    self.face_mesh_lost = False
                    for face_landmarks in results.multi_face_landmarks:


                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_iris_connections_style())

                        h, w, _ = frame.shape
                        x_min, y_min = w, h
                        x_max, y_max = 0, 0

                        is_face_straight = calculate_face_angle(face_landmarks, w, h, threshold=self.straight_threshold)
                        if not is_face_straight:
                            print("The person is facing at an angle.")
                        else:
                            print("The person is facing straight.")

                        for landmark in face_landmarks.landmark:
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)

                            if x < x_min: x_min = x
                            if y < y_min: y_min = y
                            if x > x_max: x_max = x
                            if y > y_max: y_max = y

                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        if not self.face_captured and self.face_count % self.threshold == 0:
                            self.face_captured = True
                            face_image = original_frame

                            threading.Thread(target=self.infer_recommendation, args=(face_image,)).start()

                else:
                    self.face_mesh_lost = True
                    self.last_time_face_lost = time.time()
                    self.face_captured = False
                    self.face_count = 0

                out.write(frame)
                cv2.imshow('Camera', cv2.flip(frame, 1))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                iteration_end_time = time.time()
                total_iteration_time = iteration_end_time - iteration_start_time

                if total_iteration_time > self.delay:
                    print(f"WARNING: Total iteration time ({total_iteration_time:.4f}s) exceeds DELAY ({self.delay}s)")

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        subprocess.run(["ffmpeg", "-i", "demos/output.avi", "-i", "demos/output.avi", "-filter_complex", "hstack", "output.mp4"])

        mp4 = open('output.mp4', 'rb').read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()


        display(HTML(f"""
        <video controls>
        <source src="{data_url}" type="video/mp4">
        </video>
        """))

# Configuration and execution
config = {
    "model_path": f"{args.model_path}",
    "detection_path": f"{args.detection_path}",
    "sim_threshold": {
        "euclidean": 0.45,
        "cosine": 0.08,
        "manhattan": 2.5,
        "minkowski": 0.15,
        "chebyshev": 0.01,
        "hamming": 0.5
    },
    "save_path": f"{args.save_path}",
    "process_every_n_frames": 1,
    "delay": 4,
    "threshold": 70,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "distance": ["euclidean", "cosine", "manhattan"],
    "straight_threshold": 0.35
}

system = FaceMeshRecommendationSystem(**config)
system.run()
