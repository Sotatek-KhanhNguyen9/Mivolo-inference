import torchvision.transforms as transforms
import face_recognition
import numpy as np
import requests
import torch
import os



def calculate_face_angle(face_landmarks, image_width, image_height, threshold=0.95, x_max=0, y_max=0):
    nose_landmark = face_landmarks.landmark[1]  
    x_min, y_min = image_width, image_height
    nose_x = int(nose_landmark.x * image_width)
    nose_y = int(nose_landmark.y * image_height)

    left_distance = nose_x - x_min
    right_distance = x_max - nose_x
    top_distance = nose_y - y_min
    bottom_distance = y_max - nose_y

    horizontal_ratio = left_distance / right_distance if right_distance != 0 else 0
    vertical_ratio = top_distance / bottom_distance if bottom_distance != 0 else 0

    print(f"Horizontal ratio: {abs(horizontal_ratio - 1):.4f}, Vertical ratio: {abs(vertical_ratio - 1):.4f}")

    if abs(horizontal_ratio - 1) > threshold or abs(vertical_ratio - 1) > threshold:
        return False  
    return True  

def compare_face_distances(image1, image2, distance=["euclidean", "cosine", "manhattan", "minkowski", "chebyshev", "hamming"], p = 3, hamming_threshold = 0.5):
    print("Comparing faces...")
    image1_encoding = face_recognition.face_encodings(image1)
    image2_encoding = face_recognition.face_encodings(image2)
    
    if len(image1_encoding) == 0 or len(image2_encoding) == 0:
        return np.empty((0))
    
    encoding1 = image1_encoding[0]
    encoding2 = image2_encoding[0]
    
    encoding1_binary = np.where(encoding1 > hamming_threshold, 1, 0)
    encoding2_binary = np.where(encoding2 > hamming_threshold, 1, 0)

    results = {
        "euclidean": np.linalg.norm(encoding1 - encoding2),
        "cosine":  1 - np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2)),
        "manhattan": np.sum(np.abs(encoding1 - encoding2)),
        "minkowski": np.sum(np.abs(encoding1 - encoding2) ** p) ** (1 / p),
        "chebyshev": np.max(np.abs(encoding1 - encoding2)),
        "hamming": np.sum(encoding1_binary != encoding2_binary) / len(encoding1_binary)
    }    
    # print(f"Compare Similarity Score: {results}")

    return {key: results[key] for key in distance}

def download_files_to_cache(urls, file_names, cache_dir_name="age_estimation"):
    def download_file(url, save_path):
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the download was successful

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"File downloaded and saved to {save_path}")

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", cache_dir_name)

    os.makedirs(cache_dir, exist_ok=True)
    paths = []
    for url, file_name in zip(urls, file_names):
        save_path = os.path.join(cache_dir, file_name)
        paths.append(save_path)
        if not os.path.exists(save_path):
            print(f"File {file_name} does not exist. Downloading...")
            download_file(url, save_path)
        else:
            print(f"File {file_name} already exists at {save_path}")
    return paths

def chunk_then_stack(image, detector):
    image_np = np.array(image)
    results = detector.predict(image_np, conf=0.35)
    for result in results:
        boxes = result.boxes
        face_coords = [None, None, None, None]
        person_coords = [None, None, None, None]
        for i, box in enumerate(boxes.xyxy):
            cls = int(boxes.cls[i].item())
            x_min, y_min, x_max, y_max = map(int, box.tolist())  # Chuyển tọa độ sang int
            if cls == 1:  # Face
                face_coords = [x_min, y_min, x_max, y_max]
            elif cls == 0:  # Person
                person_coords = [x_min, y_min, x_max, y_max]

    return face_coords, person_coords

def tranfer_image(image, transform_infer, detector):
    face_coords, person_coords = chunk_then_stack(image, detector)
    face_image = image.crop((int(face_coords[0]), int(face_coords[1]), int(face_coords[2]), int(face_coords[3])))

    person_image = image.crop((int(person_coords[0]), int(person_coords[1]), int(person_coords[2]), int(person_coords[3])))
    
    face_image = face_image.resize((224, 224))
    person_image = person_image.resize((224, 224))
    face_image = transform_infer(face_image)
    person_image = transform_infer(person_image)


    image_ = torch.cat((face_image, person_image), dim=0)
    return image_.unsqueeze(0)
