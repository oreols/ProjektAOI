"# ProjektAOI" 

python -m venv myenv

myenv\Scripts\activate

pip install -r requirements.txt

###

1. Gotowe labele convertujemy na voc plikiem conver-pascal-cov.py
2. Sprawdzamy poprawnosc sciezek w data_loader.py oraz train.py 
3. Uruchamiamy python train.py
4. Przetrenowany model zapisuje się w folderze ml 
5. Uruchamiamy testowanie plikiem test.py 


---------------------------------------------------------------
# 🧠 PCB AOI – Wykrywanie komponentów na płytkach PCB

Projekt systemu automatycznej kontroli optycznej (AOI) do wykrywania komponentów SMD na obrazach PCB, oparty o **PyTorch**, **Faster R-CNN** oraz metrykę mAP z **torchmetrics**.

---

## 🔧 Technologie
- 🐍 Python 3.10+
- 🔥 PyTorch + torchvision
- 📊 TorchMetrics (mAP IoU=0.50:0.95)
- 🧪 Albumentations (augmentacja danych)
- 📈 TensorBoard (monitoring metryk)
- 🗃️ Format danych: VOC XML

---

## 🗂️ Struktura projektu

```
ProjektAOI/
│
├── main.py                      # Główna pętla treningowa + ewaluacja
├── config.yaml                  # Plik konfiguracyjny projektu
├── logger.py                    # Logger CSV
├── README.md                    # Dokumentacja (ten plik)
│
├── utils/
│   └── dataset_loader.py        # Klasa PCBDataset (VOC + augmentacja)
│
├── logs/
│   ├── training_metrics.csv     # Historia strat i mAP
│   └── plots/                   # Wykresy .png (loss, mAP)
│
├── runs/                        # Logi TensorBoard
│
├── models/
│   ├── trained_components/
│   │   └── capacitors/          # Zapisane modele .pth
│   └── exported/
│       └── scripted_model.pt    # TorchScript (do wdrożenia)
```

---

## ⚙️ Uruchomienie

### 1. Instalacja zależności

```bash
pip install -r requirements.txt
```

Albo ręcznie:

```bash
pip install torch torchvision albumentations torchmetrics tensorboard matplotlib tqdm pyyaml
```

---

### 2. Start treningu

```bash
python main.py
```

---

### 3. Podgląd metryk (TensorBoard)

```bash
tensorboard --logdir=runs
```

Otwórz przeglądarkę i przejdź do: [http://localhost:6006](http://localhost:6006)

---

## 🧠 Informacje o treningu

- Model: `fasterrcnn_resnet50_fpn_v2` (torchvision)
- Liczba klas: 1 (np. kondensatory) + tło
- Augmentacja: obrót, skalowanie, jasność, padding, itp.
- Format danych: Pascal VOC (XML)
- Ewaluacja: mAP@[IoU=0.50:0.95] (`torchmetrics`)

---

## ⚙️ Konfiguracja (`config.yaml`)

Opis kluczowych sekcji:

```yaml
project:
  name: "pcb_aoi_detection"
  component: "capacitor"

data:
  train_images: "ścieżka_do_obrazów_treningowych"
  train_annotations: "ścieżka_do_plików_XML"
  val_images: "ścieżka_do_obrazów_walidacyjnych"
  val_annotations: "ścieżka_do_XML_walidacyjnych"

model:
  architecture: "fasterrcnn_resnet50_fpn_v2"
  num_classes: 2  # 1 klasa + tło

training:
  epochs: 100
  batch_size: 4
  learning_rate: 0.0001
  scheduler: "cosine"

augmentation:
  image_size: 1024
  rotate_limit: 15
  scale_limit: 0.2
  min_visibility: 0.2

logging:
  metrics_file: "logs/training_metrics.csv"
  model_save_dir: "models/trained_components/capacitors"
  scripted_model_path: "models/exported/scripted_model.pt"
```

---

## 📦 Eksport modelu

Po zakończeniu treningu:
- Najlepszy model `.pth` zapisywany w `models/trained_components/...`
- Model TorchScript (`.pt`) eksportowany do `models/exported/` → gotowy do wdrożenia

---

## 📈 Wizualizacja wyników

Automatycznie tworzone wykresy w `logs/plots/`:
- `loss_vs_epoch.png`
- `mAP_vs_epoch.png`

Dodatkowo:
- `logs/training_metrics.csv` — zapis historii metryk do CSV
- `runs/` — TensorBoard logi

---
