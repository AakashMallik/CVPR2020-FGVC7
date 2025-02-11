import torch
import pandas as pd
from skimage import io
from os import path
from torch.utils.data import Dataset
from dataset.utils import (fold_creator)

NUMBER_OF_FOLDS = 5
DATASET_NAME = 'fgvc7'


class FGVC7_Dataset(Dataset):
    def __init__(self, mode, dataset_path, transformer=None, fold_number=None):
        if transformer is None:
            print("[ No Transformer passed in - ", DATASET_NAME, " ]")
            exit()

        if fold_number is not None and fold_number >= NUMBER_OF_FOLDS:
            print("[ Fold limit exceeded in - ", DATASET_NAME, " ]")
            exit()

        self.mode = mode
        self.transformer = transformer
        self.dataset_path = dataset_path

        if fold_number is None:
            # If fold not selected
            self.csv_path = path.join(dataset_path, mode + ".csv")
            self.image_dir = path.join(dataset_path, "images")
        else:
            # if fold selected
            self.create_folds()
            self.csv_path = path.join(
                "folds", DATASET_NAME, str(fold_number), mode + ".csv")
            self.image_dir = path.join(
                "folds", DATASET_NAME, str(fold_number), mode)

        self.data_frame = pd.read_csv(self.csv_path)

    def create_folds(self):
        fold_creator(
            self.dataset_path,
            path.join("folds", DATASET_NAME),
            NUMBER_OF_FOLDS
        )

    def get_csv_path(self):
        return self.csv_path

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = str(self.data_frame.iloc[idx, 0]) + ".jpg"
        image_path = path.join(self.image_dir, image_name)
        image = io.imread(image_path)

        if self.mode == "test":
            return self.transformer(image)
        else:
            label = torch.tensor(
                self.data_frame.iloc[idx, 1:].to_numpy(dtype=float)
            )
            return self.transformer(image), label
            
