import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

class CMYKDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.image_dir, row["filename"])
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Image not found: {img_path}")

        image = cv2.resize(image, (128, 128))
        image = image / 255.0

        label = torch.tensor([
            row["C"],
            row["M"],
            row["Y"],
            row["K"]
        ], dtype=torch.float32)

        image = torch.tensor(image).permute(2, 0, 1).float()

        return image, label