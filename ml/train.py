import os
import json
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from models.faster_rcnn import get_model
from utils.dataset_loader import PCBDataset

# PyCOCOTools do obliczania mAP
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def collate_fn(batch):
    return tuple(zip(*batch))

def voc_to_coco_format(all_targets, all_predictions, image_ids):
    """
    Konwertuje listy ground truth i predykcji do formatu COCO.
    Zakładamy jedną klasę (ID=1) + tło.
    """
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}]
    }
    ann_id = 1
    for img_id, (tgt, image_id) in enumerate(zip(all_targets, image_ids)):
        coco_gt["images"].append({
            "id": image_id,
            "file_name": f"image_{image_id}.jpg",  # nazwa pliku (umowna)
            "width": 0,
            "height": 0
        })
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            w = xmax - xmin
            h = ymax - ymin
            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(labels[i]),
                "bbox": [float(xmin), float(ymin), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0
            })
            ann_id += 1

    # Tworzymy listę predykcji w formacie COCO
    coco_dt = []
    for img_id, (pred, image_id) in enumerate(zip(all_predictions, image_ids)):
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            w = xmax - xmin
            h = ymax - ymin
            coco_dt.append({
                "image_id": image_id,
                "category_id": int(labels[i]),
                "bbox": [float(xmin), float(ymin), float(w), float(h)],
                "score": float(scores[i])
            })
    return coco_gt, coco_dt

def evaluate_model(model, val_loader, device):
    """
    Wykonuje inferencję na zbiorze walidacyjnym, oblicza mAP przy użyciu pycocotools.
    Zwraca wartość AP@[IoU=0.50:0.95].
    """
    model.eval()
    all_predictions = []
    all_targets = []
    image_ids = []

    with torch.no_grad():
        img_counter = 0
        for images, targets in val_loader:
            batch_size_here = len(images)
            for i in range(batch_size_here):
                image_id = img_counter + i
                image_ids.append(image_id)
            images = [img.to(device) for img in images]
            outputs = model(images)
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
            img_counter += batch_size_here

    # Konwersja do formatu COCO
    coco_gt_dict, coco_dt_list = voc_to_coco_format(all_targets, all_predictions, image_ids)

    # Zapis do plików JSON (tymczasowo)
    with open("gt_coco_temp.json", "w") as f:
        json.dump(coco_gt_dict, f)
    with open("pred_coco_temp.json", "w") as f:
        json.dump(coco_dt_list, f)

    # Obliczenie mAP
    coco_gt_api = COCO("gt_coco_temp.json")
    coco_dt_api = coco_gt_api.loadRes("pred_coco_temp.json")
    coco_eval = COCOeval(coco_gt_api, coco_dt_api, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = coco_eval.stats[0]  # AP@[IoU=0.50:0.95]
    model.train()  # powrót do trybu treningowego
    return mAP

if __name__ == "__main__":
    # Parametry treningu
    num_classes = 2  
    num_epochs = 50
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ścieżki do treningu i walidacji
    train_image_dir = "dataset/nieprzyciete"
    train_annotation_dir = "dataset/voc_annotations-usb/train_voc"
    val_image_dir = "dataset/nieprzyciete"
    val_annotation_dir = "dataset/voc_annotations-usb/val_voc"

    # Zwiększamy intensywność augmentacji: RandomScale, Rotate
    train_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=1024),
            A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=0),
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.2   # Usuwa boxy o znikomej widoczności
        )
    )

    # Walidacja - mniejsza augmentacja
    val_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=1024),
            A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=0),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        )
    )

    # Dataset treningowy
    train_dataset = PCBDataset(train_image_dir, train_annotation_dir, augmentation_transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    # Dataset walidacyjny
    val_dataset = PCBDataset(val_image_dir, val_annotation_dir, augmentation_transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    # Inicjalizacja modelu
    model = get_model(num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    # Zamiast SGD, używamy Adam z mniejszym LR
    # optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    # Scheduler - zmniejszamy LR co 10 epok
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    best_map = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, targets in loop:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with autocast(device_type='cuda' if device.type=='cuda' else 'cpu'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += losses.item()
            loop.set_postfix(loss=f"{losses.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

        # Ewaluacja mAP
        mAP = evaluate_model(model, val_loader, device)
        print(f"[INFO] mAP (IoU=0.50:0.95) after epoch {epoch+1}: {mAP:.3f}")

        # Zapis modelu, jeśli lepszy
        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), f"models/trained_components/usb/usb_resnet50v2_model_epoch_{epoch+1}_mAP_{mAP:.3f}.pth")
            print(f"[INFO] Nowy najlepszy model zapisany (mAP={mAP:.3f})!")

    print("[INFO] Trening zakończony!")