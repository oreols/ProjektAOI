import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image

def labelme_to_voc(labelme_folder, output_folder, image_base_dir):
    os.makedirs(output_folder, exist_ok=True)
    
    for json_file in tqdm(os.listdir(labelme_folder), desc="Konwersja"):
        if not json_file.endswith(".json"):
            continue
        
        json_path = os.path.join(labelme_folder, json_file)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Pobieramy tylko nazwę pliku (ignorujemy oryginalną ścieżkę) 
        # i wymuszamy rozszerzenie .jpg, bo wszystkie obrazy są jpg.
        image_name = os.path.basename(data.get("imagePath", ""))
        image_name = os.path.splitext(image_name)[0] + ".jpg"
        
        if not image_name:
            print(f"Ostrzeżenie: Plik {json_file} nie zawiera pola 'imagePath'. Pomijam.")
            continue
        
        # Budujemy ścieżkę do obrazu, korzystając z katalogu bazowego
        image_path = os.path.join(image_base_dir, image_name)
        try:
            image = Image.open(image_path)
            width, height = image.size
        except Exception as e:
            print(f"Nie udało się wczytać obrazu {image_path}: {e}")
            continue
        
        # Tworzenie struktury XML zgodnie z formatem VOC
        annotation = ET.Element("annotation")
        
        folder_el = ET.SubElement(annotation, "folder")
        folder_el.text = os.path.basename(image_base_dir)
        
        filename_el = ET.SubElement(annotation, "filename")
        filename_el.text = image_name
        
        path_el = ET.SubElement(annotation, "path")
        path_el.text = image_path
        
        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"
        
        size_el = ET.SubElement(annotation, "size")
        ET.SubElement(size_el, "width").text = str(width)
        ET.SubElement(size_el, "height").text = str(height)
        ET.SubElement(size_el, "depth").text = "3"  # zakładamy RGB
        
        segmented = ET.SubElement(annotation, "segmented")
        segmented.text = "0"
        
        # Iteracja po adnotacjach w polu "shapes"
        for shape in data.get("shapes", []):
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = shape.get("label", "unknown")
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            
            points = shape.get("points", [])
            if not points:
                continue
            
            x_min = int(min(p[0] for p in points))
            y_min = int(min(p[1] for p in points))
            x_max = int(max(p[0] for p in points))
            y_max = int(max(p[1] for p in points))
            
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(x_min)
            ET.SubElement(bndbox, "ymin").text = str(y_min)
            ET.SubElement(bndbox, "xmax").text = str(x_max)
            ET.SubElement(bndbox, "ymax").text = str(y_max)
        
        # Zapisujemy drzewo XML do pliku
        xml_filename = os.path.splitext(json_file)[0] + ".xml"
        xml_path = os.path.join(output_folder, xml_filename)
        tree = ET.ElementTree(annotation)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        print(f"Created VOC xml: {xml_path}")
    
    print(f"Konwersja zakończona! Pliki zapisane w katalogu: {output_folder}")

# Przykładowe użycie
if __name__ == "__main__":
    labelme_folder = "dataset/labels-transistors-tactswitches/"  # Folder z plikami JSON
    output_folder = "dataset/voc_annotations-transistors-tactswitches/"  # Folder, gdzie zapisane zostaną pliki XML
    image_base_dir = "dataset"  # Wszystkie obrazy są bezpośrednio w katalogu 'dataset'
    labelme_to_voc(labelme_folder, output_folder, image_base_dir)
