"""MultiCropBiometryDataset -- wraps the base dataset to yield 2 global + N local
crops per image for DINO-style Phase-1 self-distillation. Ported from the original.
"""

import torch
from torch.utils.data import Dataset

from .dataset import read_rgb


class MultiCropBiometryDataset(Dataset):
    def __init__(self, base_dataset, transforms_dict, n_local_crops=6):
        self.base_dataset = base_dataset
        self.global_transform = transforms_dict["global"]
        self.local_transform = transforms_dict["local"]
        self.n_local_crops = n_local_crops

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        info = self.base_dataset.samples[idx]
        image = read_rgb(info["abs_path"])
        if image is None:
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())

        crops = []
        for _ in range(2):
            crops.append(self.global_transform(image=image)["image"])
        for _ in range(self.n_local_crops):
            crops.append(self.local_transform(image=image)["image"])
        return {"crops": crops, "task_id": info["task_id"]}
