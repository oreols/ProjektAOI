# evaluate.py

import os
import json
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Import Twojego datasetu i ewentualnej funkcji collate_fn
from utils.dataset_loader import PCBDataset  # Dostosuj ścieżkę importu do swojego projektu

###############################################################################
# Funkcja, która tworzy domyślny model Faster R-CNN z ResNet50 FPN
###############################################################################
def get_default_model(num_classes: int):
    import torchvision
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

###############################################################################
# Funkcja do wczytania wytrenowanych wag modelu
###############################################################################
def load_trained_model(model_path: str, num_classes: int, device: str = "cpu"):
    model = get_default_model(num_classes)
    model.to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model

###############################################################################
# Funkcja collate_fn (jeśli jej potrzebujesz przy DataLoaderze)
###############################################################################
def collate_fn(batch):
    return tuple(zip(*batch))

###############################################################################
# Funkcje pomocnicze do konwersji VOC -> COCO
###############################################################################
def voc_to_coco_format(all_targets, all_predictions, image_ids):
    """
    Konwertuje listy ground truth i predykcji do formatu COCO.
    - all_targets: lista słowników z kluczami {"boxes": np.ndarray, "labels": np.ndarray}
    - all_predictions: lista słowników {"boxes": np.ndarray, "scores": np.ndarray, "labels": np.ndarray}
    - image_ids: lista identyfikatorów obrazu (np. idx lub inna numeracja)
    
    Zwraca (coco_gt, coco_dt):
    - coco_gt: słownik w formacie COCO z kluczami "images", "annotations", "categories"
    - coco_dt: lista predykcji w formacie COCO (dict per detection)
    """
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    # Załóżmy, że masz jedną klasę obiektu (oprócz tła) => category_id=1
    # Jeśli masz więcej klas, zdefiniuj je odpowiednio
    coco_gt["categories"] = [{"id": 1, "name": "object"}]  # Dostosuj do liczby klas
    
    ann_id = 1
    for img_id, (tgt, image_id) in enumerate(zip(all_targets, image_ids)):
        # Dodajemy informację o obrazie
        coco_gt["images"].append({
            "id": image_id,
            "file_name": f"image_{image_id}.jpg",  # lub inna nazwa
            "width": 0,   # można pominąć lub podać rzeczywiste wymiary
            "height": 0
        })

        boxes = tgt["boxes"]  # shape (N, 4) [xmin, ymin, xmax, ymax]
        labels = tgt["labels"]  # shape (N,)
        
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            w = xmax - xmin
            h = ymax - ymin
            cat_id = int(labels[i])  # jeśli masz 1 klasę, to pewnie 1

            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [float(xmin), float(ymin), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0
            })
            ann_id += 1

    # Predykcje w formacie COCO
    coco_dt = []
    for img_id, (pred, image_id) in enumerate(zip(all_predictions, image_ids)):
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]
        
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            w = xmax - xmin
            h = ymax - ymin
            cat_id = int(labels[i])  # Dostosuj do liczby klas

            coco_dt.append({
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [float(xmin), float(ymin), float(w), float(h)],
                "score": float(scores[i])
            })
    return coco_gt, coco_dt

###############################################################################
# GŁÓWNA CZĘŚĆ – INFERENCJA NA ZBIORZE WALIDACYJNYM + obliczanie mAP
###############################################################################
# Ścieżka do wytrenowanego modelu
MODEL_PATH = "final_model.pth"

# Liczba klas (np. 1 klasa obiektów + tło => num_classes=2)
NUM_CLASSES = 2

# Urządzenie (CPU / CUDA)
DEVICE = "cuda"

# Katalog z obrazami walidacyjnymi
VAL_IMAGES_DIR = "dataset"
# Katalog z plikami XML (VOC) walidacyjnymi
VAL_ANNOTATIONS_DIR = "dataset/voc_annotations/val_voc"

if __name__ == "__main__":
    # 1. Ładujemy wytrenowany model
    model = load_trained_model(MODEL_PATH, NUM_CLASSES, DEVICE)

    # 2. Tworzymy dataset walidacyjny
    val_dataset = PCBDataset(
        image_dir=VAL_IMAGES_DIR,
        annotation_dir=VAL_ANNOTATIONS_DIR,
        augmentation_transform=None  # Walidacja bez augmentacji
    )

    # 3. DataLoader dla zbioru walidacyjnego
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,  # lub inna liczba wątków
        collate_fn=collate_fn
    )

    model.eval()
    all_predictions = []
    all_targets = []
    image_ids = []

    with torch.no_grad():
        for idx, (images, targets) in enumerate(val_loader):
            # Przykład: image_id = idx * batch_size + i
            # Lepiej, jeśli Twój dataset ma stałe ID dla każdego obrazka
            # W dataset_loader możesz dodać np. target["image_id"] = idx
            # i tu go pobierać
            batch_size_here = len(images)
            for i in range(batch_size_here):
                # image_id to unikalny identyfikator dla COCO
                image_id = idx * batch_size_here + i
                image_ids.append(image_id)

            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            # Zapisujemy wyniki
            for out, tgt in zip(outputs, targets):
                pred_boxes = out["boxes"].cpu().numpy()
                pred_scores = out["scores"].cpu().numpy()
                pred_labels = out["labels"].cpu().numpy()

                true_boxes = tgt["boxes"].cpu().numpy()
                true_labels = tgt["labels"].cpu().numpy()

                all_predictions.append({
                    "boxes": pred_boxes,
                    "scores": pred_scores,
                    "labels": pred_labels
                })
                all_targets.append({
                    "boxes": true_boxes,
                    "labels": true_labels
                })

    print(f"Liczba przetworzonych obrazów: {len(all_predictions)}")

    # 6. (NOWOŚĆ) Oblicz metryki (mAP) z użyciem pycocotools
    #    Konwertujemy VOC -> COCO, a potem wywołujemy COCOeval
    coco_gt_dict, coco_dt_list = voc_to_coco_format(all_targets, all_predictions, image_ids)

    # Zapisujemy do plików .json
    with open("gt_coco.json", "w") as f:
        json.dump(coco_gt_dict, f)
    with open("pred_coco.json", "w") as f:
        json.dump(coco_dt_list, f)

    # Ładujemy ground truth
    coco_gt_api = COCO("gt_coco.json")
    # Ładujemy predykcje
    coco_dt_api = coco_gt_api.loadRes("pred_coco.json")

    # Tworzymy obiekt COCOeval
    coco_eval = COCOeval(coco_gt_api, coco_dt_api, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Wyświetlane zostaną m.in. AP@[0.5:0.95], AP@0.5, AP@0.75, AR itd.
    print("Ewaluacja (mAP) zakończona!")
