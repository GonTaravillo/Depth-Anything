import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class OpenImagesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        data_dir: Directorio raíz donde están las imágenes descargadas
                  (ej: './data/open_images_v7/train/data')
        """
        self.data_dir = data_dir
        # Obtener todas las imágenes jpg en el directorio
        self.image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
        
        if transform is None:
            # Transformaciones estándar esperadas por DINOv2 / DepthAnything
            self.transform = transforms.Compose([
                transforms.Resize((518, 518)), # Resolución esperada múltiple de 14
                transforms.RandomHorizontalFlip(p=0.5), # Aumentación: duplica variedad de datos
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error cargando la imagen {img_path}: {e}")
            # Si hay un error, retorna otra imagen aleatoria
            return self.__getitem__((idx + 1) % len(self))
