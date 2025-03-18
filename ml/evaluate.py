import os
import json
import torch
import cv2
import xml.etree.ElementTree as ET
from torch.utils.data import DataLoader
from models.faster_rcnn import get_model
from utils.dataset_loader import PCBDataset
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def create_coco_ground_truth(image_dir, annotation_dir):
    """
    Konwertuje adnotacje w formacie VOC do formatu COCO.
    Zwraca słownik zawierający klucze: "images", "annotations", "categories".
    """
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "capacitor"}
        ]
    }
    ann_id = 1
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    # Uwzględniamy tylko obrazy, dla których istnieje adnotacja XML
    image_files = [f for f in image_files if os.path.exists(os.path.join(annotation_dir, f.rsplit('.', 1)[0] + '.xml'))]
    
    for img_id, file_name in enumerate(image_files):
        img_path = os.path.join(image_dir, file_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        height, width = img.shape[:2]
        coco_gt["images"].append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })
        
        xml_file = os.path.join(annotation_dir, file_name.rsplit('.', 1)[0] + '.xml')
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            box_width = xmax - xmin
            box_height = ymax - ymin
            
            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,  # Zakładamy jedną klasę: kondensator
                "bbox": [xmin, ymin, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0
            })
            ann_id += 1
    return coco_gt

def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)

# Definicja globalnej funkcji collate_fn
def collate_fn(batch):
    return tuple(zip(*batch))

# Aby móc przypisać image_id w wynikach inferencji, rozszerzamy nasz dataset
class EvalDataset(PCBDataset):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        # Dodajemy image_id do target, które odpowiada indeksowi obrazu
        target["image_id"] = idx
        return img, target

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = "dataset"
    annotation_dir = os.path.join("dataset", "voc_annotations")
    
    # 1. Konwersja ground truth do formatu COCO
    coco_gt = create_coco_ground_truth(image_dir, annotation_dir)
    gt_filename = "gt_coco.json"
    save_json(coco_gt, gt_filename)
    print(f"Ground truth zapisane w {gt_filename}")
    
    # 2. Przygotowanie zbioru walidacyjnego (bez augmentacji)
    eval_dataset = EvalDataset(image_dir, annotation_dir, augmentation_transform=None)
    data_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # 3. Ładowanie modelu
    num_classes = 2
    model = get_model(num_classes)
    model.load_state_dict(torch.load("faster_rcnn_pcb.pth", map_location=device))
    model.to(device)
    model.eval()
    
    # 4. Przeprowadzenie inferencji i zbieranie predykcji
    predictions = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Przeprowadzanie inferencji"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for idx, output in enumerate(outputs):
                image_id = targets[idx]["image_id"]
                boxes = output["boxes"].cpu().numpy().tolist()
                scores = output["scores"].cpu().numpy().tolist()
                labels = output["labels"].cpu().numpy().tolist()
                for box, score, label in zip(boxes, scores, labels):
                    if score < 0.05:
                        continue
                    # Konwersja z [xmin, ymin, xmax, ymax] do [xmin, ymin, width, height]
                    xmin, ymin, xmax, ymax = box
                    width = xmax - xmin
                    height = ymax - ymin
                    predictions.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [xmin, ymin, width, height],
                        "score": score
                    })
    pred_filename = "predictions.json"
    save_json(predictions, pred_filename)
    print(f"Predykcje zapisane w {pred_filename}")
    
    # 5. Ocena modelu za pomocą pycocotools
    coco_gt_api = COCO(gt_filename)
    coco_dt = coco_gt_api.loadRes(pred_filename)
    coco_eval = COCOeval(coco_gt_api, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    main()
