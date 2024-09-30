import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch.nn.functional as F
import time
from mivolo_model import MiVOLOModel_1
from utils import download_files_to_cache

urls = [
    "https://huggingface.co/hungdang1610/estimate_age/resolve/main/models/best_model_weights_10.pth?download=true",
 
    "https://huggingface.co/hungdang1610/estimate_age/resolve/main/models/yolov8x_person_face.pt?download=true"
]

file_names = [
    "best_model_weights_10.pth",
    "yolov8x_person_face.pt"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RecommendationSystem:
    def __init__(self, model_path=None, detection_path=None, urls=None, file_names=None, device="cuda"):
        self.DEVICE = device
        if urls != None and file_names != None:
            self.model_path, self.detection_path = download_files_to_cache(urls, file_names)
        elif model_path != None and detection_path != None:
            self.model_path = model_path
            self.detection_path = detection_path
        
        self.model = MiVOLOModel_1(
            layers=(4, 4, 8, 2),
            img_size=224,
            in_chans=6,
            num_classes=3,
            patch_size=8,
            stem_hidden_dim=64,
            embed_dims=(192, 384, 384, 384),
            num_heads=(6, 12, 12, 12),
        ).to(self.DEVICE)

        state = torch.load(self.model_path, map_location=self.DEVICE)
        try:
            self.model.load_state_dict(state, strict=True)
        except:
            raise Exception("Model weights are not compatible with the model architecture.")

        self.detector = YOLO(self.detection_path)

        self.transform_infer = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.MEAN_TRAIN = 36.64
        self.STD_TRAIN = 21.74

    def chunk_then_stack(self, image):
        image_np = np.array(image)
        results = self.detector.predict(image_np, conf=0.35)
        for result in results:
            boxes = result.boxes

            face_coords = [None, None, None, None]
            person_coords = [None, None, None, None]

            for i, box in enumerate(boxes.xyxy):
                cls = int(boxes.cls[i].item())
                x_min, y_min, x_max, y_max = map(int, box.tolist())  # Convert coordinates to int

                if cls == 1:  
                    face_coords = [x_min, y_min, x_max, y_max]
                elif cls == 0:  
                    person_coords = [x_min, y_min, x_max, y_max]

        return face_coords, person_coords

    def transfer_image(self, image):
        face_coords, person_coords = self.chunk_then_stack(image)

        face_image = image.crop((int(face_coords[0]), int(face_coords[1]), int(face_coords[2]), int(face_coords[3])))
        person_image = image.crop((int(person_coords[0]), int(person_coords[1]), int(person_coords[2]), int(person_coords[3])))

        face_image = self.transform_infer(face_image)
        person_image = self.transform_infer(person_image)

        image_ = torch.cat((face_image, person_image), dim=0)
        return image_.unsqueeze(0)

    def infer_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        image_ = self.transfer_image(image)
        image_ = image_.to(self.DEVICE)

        start_time = time.time()
        output = self.model(image_)

        age_tensor = output[1]
        output_mse = age_tensor[:, 0]
        predicted_age = output_mse.item() * self.STD_TRAIN + self.MEAN_TRAIN

        gender_logits = output[0]
        gender_probs = F.softmax(gender_logits, dim=1)
        predicted_gender_index = torch.argmax(gender_probs, dim=1).item()

        predicted_gender = 'Female' if predicted_gender_index == 0 else 'Male'

        print(f"Inference time: {time.time() - start_time} seconds")
        return predicted_gender, predicted_age


if __name__ == "__main__":
    system = RecommendationSystem(
        model_path    ="weights/best_model_weights_10.pth",
        detection_path="weights/yolov8x_person_face.pt"
    )
    # gender, age = system.infer_image("captured_face.jpg")
    # print(f"Predicted Gender: {gender}, Predicted Age: {age}")
    print("Model config 1 loaded successfully!")

    system = RecommendationSystem(
        urls=urls,
        file_names=file_names
    )
    print("Model config 2 loaded successfully!")
