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
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

        # Używamy tylko obrazów, dla których istnieją adnotacje
        self.image_filenames = [
            f for f in self.image_filenames 
            if os.path.exists(os.path.join(self.annotation_dir, f.rsplit('.', 1)[0] + '.xml'))
        ]

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        xml_path = os.path.join(self.annotation_dir, img_name.rsplit('.', 1)[0] + '.xml')

        # Wczytanie obrazu i konwersja do formatu RGB
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Parsowanie adnotacji XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.findall("object"):
            labels.append(1)  # Zakładamy jedną klasę (kondensatory)
            bbox = obj.find("bndbox")
            x_min = int(bbox.find("xmin").text)
            y_min = int(bbox.find("ymin").text)
            x_max = int(bbox.find("xmax").text)
            y_max = int(bbox.find("ymax").text)
            boxes.append([x_min, y_min, x_max, y_max])

        # Jeśli pipeline augmentacji został przekazany, zastosuj go
        if self.augmentation_transform:
            augmented = self.augmentation_transform(image=img, bboxes=boxes, labels=labels)
            img = augmented['image']
            boxes = augmented['bboxes']
            labels = augmented['labels']
        else:
            img = T.ToTensor()(img)
        
        # Upewnij się, że obraz jest w formacie float (zakres [0, 1])
        if not torch.is_floating_point(img):
            img = img.float() / 255.0

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32), 
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        return img, target

    def __len__(self):
        return len(self.image_filenames)
