import torch
import cv2
import matplotlib.pyplot as plt
from models.faster_rcnn import get_model

# Poprawione ścieżki do katalogów
image_path = "test.png"  
model_path = "models/trained_components/jumpers/jumpers_resnet50v2_model_epoch_49_mAP_0.469.pth"

num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Załadowanie modelu
model = get_model(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Wczytanie obrazu
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Nie znaleziono pliku: {image_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0  # Konwersja do tensoru
img_tensor = img_tensor.unsqueeze(0).to(device)  # Dodanie wymiaru batch i przeniesienie na GPU/CPU

# Predykcja
with torch.no_grad():
    prediction = model(img_tensor)[0]

def visualize(image, prediction, threshold=0.8):
    """
    Wizualizacja wykrytych obiektów.
    - image: obraz w formacie tensorowym
    - prediction: słownik z przewidywaniami modelu
    - threshold: minimalna pewność predykcji do wyświetlenia obiektu
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(image.squeeze(0).permute(1, 2, 0).cpu().numpy())
    count = 0  # Licznik obiektów

    for box, score in zip(prediction["boxes"], prediction["scores"]):
        if score > threshold:  # Tylko pewne detekcje
            x1, y1, x2, y2 = map(int, box.cpu().numpy())  # Konwersja na int
            count += 1  # Zliczanie obiektów
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', linewidth=2, fill=False))
            plt.text(x1, y1 - 5, f'IC: {score:.2f}', color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    print(f"Liczba wykrytych obiektów: {count}")
    plt.axis("off")
    plt.show()

# Wizualizacja wyników
visualize(img_tensor, prediction)
