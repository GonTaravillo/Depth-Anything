import fiftyone as fo
import fiftyone.zoo as foz
import os
from pathlib import Path

def download_open_images_subset(max_samples=50000, dataset_dir="../data/open_images_v7"):
    """
    Descarga un subconjunto de imágenes de Open Images V7.
    """
    # Intentar encontrar si ya existe en el disco para saltar el loading pesado
    try:
        dataset_path = foz.find_zoo_dataset("open-images-v7")
        print(f"Dataset ya encontrado en disco: {dataset_path}")
        print("Saltando descarga/loading pesado...")
    except:
        print(f"Descargando {max_samples} imágenes de Open Images V7...")
        # Solo necesitamos las imágenes, sin las anotaciones
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            max_samples=max_samples,
            seed=42,
            shuffle=True
        )
        dataset_path = foz.find_zoo_dataset("open-images-v7")
    print(f"Dataset descargado en {dataset_path}")
    
    # Crear link simbólico para que train.py lo encuentre en ./data
    default_source = os.path.join(dataset_path, "train", "data")
    target_link = os.path.join(dataset_dir, "train", "data")
    
    if os.path.exists(default_source):
        os.makedirs(os.path.dirname(target_link), exist_ok=True)
        if not os.path.exists(target_link):
            print(f"Creando enlace simbólico: {target_link} -> {default_source}")
            os.symlink(default_source, target_link)
        else:
            print(f"El enlace o directorio {target_link} ya existe.")
    
    
    # print(dataset) # Eliminamos esto para evitar errores si no se instancia

if __name__ == "__main__":
    download_open_images_subset()
