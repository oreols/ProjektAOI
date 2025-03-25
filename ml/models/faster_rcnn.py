import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Ustawienia „na sztywno”
MODEL_PATH = "models/trained_components/final_capacitor_faster_rcnn_pcb.pth"  # Ścieżka do pliku z wagami
NUM_CLASSES = 2  # Liczba klas (1 klasa obiektów + tło)
DEVICE = "cuda"   # Lub "cuda" jeśli masz GPU z zainstalowanym PyTorch z obsługą CUDA

def get_model(num_classes: int):
    """
    Tworzy i zwraca domyślny model Faster R-CNN (ResNet50 FPN)
    z wstępnie wytrenowanymi wagami na COCO.
    Modyfikuje warstwę końcową (box_predictor) pod liczbę klas.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_trained_model(model_path: str, num_classes: int, device: str = "cpu"):
    """
    Ładuje wytrenowany model Faster R-CNN z pliku `model_path`.
    Zakłada, że liczba klas to `num_classes`.
    Zwraca model w trybie ewaluacji (model.eval()).
    """
    model = get_model(num_classes)
    model.to(device)

    # Wczytanie wag (state_dict) z pliku
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Ustawienie modelu w tryb ewaluacji
    model.eval()
    return model
