import torch
from torch.utils.data import DataLoader
from models.faster_rcnn import get_model
from utils.dataset_loader import PCBDataset
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Globalna funkcja collate_fn, aby uniknąć problemów z picklingiem
def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    num_classes = 2  
    num_epochs = 20
    batch_size = 4  # Upewnij się, że GPU pozwala na taki batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dir = "dataset"
    annotation_dir = "dataset/voc_annotations-jumpers"

    # Rozszerzony pipeline augmentacji – usunięto parametry powodujące ostrzeżenia.
    augmentation_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),                # Flip poziomy
            A.RandomRotate90(p=0.5),                # Losowa rotacja o 90 stopni
            A.Rotate(limit=15, p=0.5),               # Obrót o losowy kąt w zakresie ±15 stopni
            A.RandomBrightnessContrast(p=0.5),      # Losowa zmiana jasności i kontrastu
            A.CoarseDropout(p=0.5),                 # Losowe "wycinanie" fragmentów obrazu z domyślnymi ustawieniami
            A.GaussNoise(p=0.5),                    # Dodanie szumu Gaussowskiego z domyślnymi ustawieniami
            ToTensorV2()                           # Konwersja obrazu do tensora (skalowanie do [0,1])
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )

    dataset = PCBDataset(image_dir, annotation_dir, augmentation_transform=augmentation_transform)
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,  # równoległe ładowanie danych
        collate_fn=collate_fn
    )

    model = get_model(num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    scaler = GradScaler(enabled=(device.type == 'cuda'))

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        loop = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
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
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "models/trained_components/jumpers_faster_rcnn_pcb.pth")
    print("Model zapisany!")

if __name__ == '__main__':
    main()
