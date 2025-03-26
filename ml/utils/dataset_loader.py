import os
import torch
from torch.utils.data import Dataset
import cv2
import xml.etree.ElementTree as ET
import albumentations as A
import torchvision.transforms as T

class PCBDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, augmentation_transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.augmentation_transform = augmentation_transform

        # Lista plików .jpg/.png
        all_files = os.listdir(image_dir)
        self.image_filenames = [f for f in all_files if f.lower().endswith(('.jpg', '.png'))]

        # Tylko te, dla których istnieją adnotacje XML
        self.image_filenames = [
            f for f in self.image_filenames 
            if os.path.exists(os.path.join(self.annotation_dir, f.rsplit('.', 1)[0] + '.xml'))
        ]

        print(f"[INIT] Znalazłem {len(self.image_filenames)} plików w '{image_dir}' z odpowiadającymi adnotacjami w '{annotation_dir}'.")

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        xml_path = os.path.join(self.annotation_dir, img_name.rsplit('.', 1)[0] + '.xml')

        # Wczytanie obrazu
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"[ERROR] Nie można wczytać obrazu: {img_path}")

        if len(img.shape) < 3 or img.shape[2] != 3:
            raise RuntimeError(f"[ERROR] Obraz nie ma 3 kanałów: {img_path}, shape={img.shape}")

        height, width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Log
        # print(f"[INFO] idx={idx}: '{img_name}' ma rozmiar {width}x{height}")

        # Parsowanie XML
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            raise RuntimeError(f"[ERROR] Nie można sparsować pliku XML '{xml_path}': {e}")

        boxes = []
        labels = []

        objects = root.findall("object")
        # print(f"[INFO] idx={idx}: Plik XML '{xml_path}' zawiera {len(objects)} obiektów")

        for obj in objects:
            labels.append(1)  # Jedna klasa (np. kondensator)
            bbox = obj.find("bndbox")
            x_min = int(bbox.find("xmin").text)
            y_min = int(bbox.find("ymin").text)
            x_max = int(bbox.find("xmax").text)
            y_max = int(bbox.find("ymax").text)

            # Przycinanie do rozmiaru obrazu
            x_min = max(0, min(x_min, width - 1))
            y_min = max(0, min(y_min, height - 1))
            x_max = max(0, min(x_max, width - 1))
            y_max = max(0, min(y_max, height - 1))

            w = x_max - x_min
            h = y_max - y_min

            # Pomijamy box o niepoprawnym rozmiarze
            if w <= 0 or h <= 0:
                print(f"[WARN] Pomijam box o niepoprawnym rozmiarze: {img_name}, box={(x_min, y_min, x_max, y_max)}")
                continue

            boxes.append([x_min, y_min, x_max, y_max])

        # Augmentacja (lub ToTensor)
        if self.augmentation_transform:
            augmented = self.augmentation_transform(
                image=img,
                bboxes=boxes,
                labels=labels
            )
            img = augmented['image']
            boxes = augmented['bboxes']
            labels = augmented['labels']
        else:
            img = T.ToTensor()(img)

        # Upewnij się, że obraz jest float w [0,1]
        if not torch.is_floating_point(img):
            img = img.float() / 255.0

        # Tworzenie tensora boxów i etykiet
        if len(boxes) == 0:
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            labels_tensor = torch.empty((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor
        }

        return img, target

    def __len__(self):
        return len(self.image_filenames)
