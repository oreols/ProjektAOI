import os
import yaml
import torch
import torchvision
import albumentations as A
from tqdm import tqdm
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from torch.amp import GradScaler, autocast
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils.dataset_loader import PCBDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from logger import CSVLogger
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def collate_fn(batch):
    return tuple(zip(*batch))
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def evaluate_model(model, val_loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            metric.update(outputs, targets)
    mAP = metric.compute()["map"].item()
    model.train()
    return mAP

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    


def create_transforms(cfg, is_train=True):
    base = [
        A.LongestMaxSize(max_size=cfg["image_size"]),
        A.PadIfNeeded(min_height=cfg["image_size"], min_width=cfg["image_size"], border_mode=0),
    ]
    if is_train:
        base += [
            A.RandomScale(scale_limit=cfg["scale_limit"], p=0.5),
            A.Rotate(limit=cfg["rotate_limit"], p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.1),
        ]
    base.append(ToTensorV2())
    return A.Compose(base, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=cfg["min_visibility"] if is_train else 0.0))

def main():
    loss_history = []
    mAP_history = []
    
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir=os.path.join("runs", config["project"]["name"], config["project"]["component"]))
    # Logger
    logger = CSVLogger(config["logging"]["metrics_file"], ["epoch", "avg_loss", "mAP"])

    # Transforms
    train_transform = create_transforms(config["augmentation"], is_train=True)
    val_transform = create_transforms(config["augmentation"], is_train=False)

    # Datasets
    train_dataset = PCBDataset(config["data"]["train_images"], config["data"]["train_annotations"], augmentation_transform=train_transform)
    val_dataset = PCBDataset(config["data"]["val_images"], config["data"]["val_annotations"], augmentation_transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=config["training"]["num_workers"], collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config["training"]["num_workers"], collate_fn=collate_fn)

    # Model
    model = get_model(config["model"]["num_classes"]).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["t_max"], eta_min=config["training"]["eta_min"])
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    best_map = 0.0
    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")

        for images, targets in loop:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += losses.item()
            loop.set_postfix(loss=f"{losses.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        mAP = evaluate_model(model, val_loader, device)
        print(f"[INFO] Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, mAP: {mAP:.3f}")
        logger.log([epoch + 1, avg_loss, mAP])
        
        loss_history.append(avg_loss)
        mAP_history.append(mAP)

        
        writer.add_scalar("loss/train", avg_loss, epoch + 1)
        writer.add_scalar("mAP/val", mAP, epoch + 1)


        if config["logging"]["save_best_model"] and mAP > best_map:
            best_map = mAP
            save_path = os.path.join(config["logging"]["model_save_dir"], f"best_model_epoch_{epoch+1}_mAP_{mAP:.3f}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Zapisano nowy najlepszy model: {save_path}")

    # TorchScript Export
    dummy_input = torch.randn(1, 3, config["augmentation"]["image_size"], config["augmentation"]["image_size"]).to(device)
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, config["logging"]["scripted_model_path"])
    print(f"[INFO] TorchScript model saved to: {config['logging']['scripted_model_path']}")
    writer.close()
    
    # Zapis wykresów
    os.makedirs("logs/plots", exist_ok=True)

    # Loss
    plt.figure()
    plt.plot(range(1, len(loss_history)+1), loss_history, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("logs/plots/loss_vs_epoch.png")
    plt.close()

    # mAP
    plt.figure()
    plt.plot(range(1, len(mAP_history)+1), mAP_history, label="Validation mAP")
    plt.xlabel("Epoch")
    plt.ylabel("mAP@[.5:.95]")
    plt.title("Validation mAP")
    plt.grid(True)
    plt.legend()
    plt.savefig("logs/plots/mAP_vs_epoch.png")
    plt.close()

    print("[INFO] Wykresy zapisane do logs/plots/")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')  # dla bezpieczeństwa na Windows
    main()
