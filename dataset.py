import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize, ConvertImageDtype


class PizzaDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_paths = [
            f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.resizer = Resize((64, 64), antialias=True)
        self.converter = ConvertImageDtype(torch.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = read_image(img_path, mode=ImageReadMode.RGB) / 255.0
        image = self.converter(image)
        image = self.resizer(image)

        image = image.view(64, -1)

        if "not_pizza" in img_name.lower():
            label = torch.tensor([0], dtype=torch.float32)
        elif "pizza" in img_name.lower():
            label = torch.tensor([1], dtype=torch.float32)
        else:
            raise ValueError(f"Не удалось определить класс для файла: {img_name}")

        return image, label
