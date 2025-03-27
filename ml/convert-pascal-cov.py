import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image

def labelme_to_voc(labelme_folder, image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for json_file in tqdm(os.listdir(labelme_folder)):
        if not json_file.endswith(".json"):
            continue

        json_path = os.path.join(labelme_folder, json_file)
        with open(json_path, "r") as f:
            data = json.load(f)

        # Pobranie informacji o obrazie
        image_name = data["imagePath"]
        image_name = os.path.basename(image_name)  # Usuwa błędne ścieżki
        image_path = os.path.join(image_folder, image_name)  # Ścieżka do obrazów w folderze 'inductor-jpg'

        try:
            image = Image.open(image_path)
            width, height = image.size
        except FileNotFoundError:
            print(f"Nie znaleziono obrazu: {image_path}")
            continue

        # Tworzenie struktury XML
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = os.path.basename(image_folder)
        ET.SubElement(annotation, "filename").text = image_name
        ET.SubElement(annotation, "path").text = image_path

        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"  # Zakładając, że obrazy są kolorowe (RGB)

        for shape in data["shapes"]:
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = shape["label"]
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            # Konwersja punktów do bounding boxa
            points = shape["points"]
            x_min, y_min = int(min(p[0] for p in points)), int(min(p[1] for p in points))
            x_max, y_max = int(max(p[0] for p in points)), int(max(p[1] for p in points))

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(x_min)
            ET.SubElement(bndbox, "ymin").text = str(y_min)
            ET.SubElement(bndbox, "xmax").text = str(x_max)
            ET.SubElement(bndbox, "ymax").text = str(y_max)

        # Zapis do pliku XML
        xml_str = ET.tostring(annotation, encoding="utf-8").decode()
        xml_path = os.path.join(output_folder, json_file.replace(".json", ".xml"))
        with open(xml_path, "w") as xml_file:
            xml_file.write(xml_str)

    print(f"Konwersja zakończona! Pliki zapisane w {output_folder}")

# Przykładowe użycie
labelme_folder = "dataset/labels-inductor/"   # Folder z plikami JSON
image_folder = "dataset/inductor-jpg/"        # Folder z obrazami
output_folder = "dataset/voc_annotations-inductor/"  # Folder do zapisania wyników

labelme_to_voc(labelme_folder, image_folder, output_folder)
