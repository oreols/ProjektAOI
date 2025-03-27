import os
import xml.etree.ElementTree as ET

def update_voc_xml(xml_path, new_folder):
    """
    Aktualizuje plik XML w formacie VOC, zmieniając:
      - <folder> na new_folder
      - <path> na new_folder + <filename>
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Aktualizacja tagu <folder>
    folder_el = root.find("folder")
    if folder_el is not None:
        folder_el.text = new_folder
    
    # Aktualizacja tagu <path>
    filename_el = root.find("filename")
    if filename_el is not None:
        filename = filename_el.text
        new_path = os.path.join(new_folder, filename)
        path_el = root.find("path")
        if path_el is not None:
            path_el.text = new_path
    
    # Zapisz zmodyfikowany plik XML
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

# Przykładowe użycie:
if __name__ == "__main__":
    # Katalog, w którym znajdują się pliki XML (VOC)
    xml_folder = "dataset/voc_annotations-resistor/train_voc"
    # Nowa wartość dla tagu <folder> i ścieżki, np. katalog z obrazami
    new_folder = "dataset"
    
    for file in os.listdir(xml_folder):
        if file.endswith(".xml"):
            xml_path = os.path.join(xml_folder, file)
            update_voc_xml(xml_path, new_folder)
            print(f"Zaktualizowano: {xml_path}")
