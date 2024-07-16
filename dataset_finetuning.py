from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from augmentations import Transformer, Crop, Cutout, Noise, Normalize, Blur, Flip

class FineTuneMRIDataset(Dataset):

    def __init__(self, config, training=False, validation=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert training != validation  # Ensure dataset is either training or validation, not both

        self.config = config

        # Define transformation for training and validation
        if training:
            self.transforms = Transformer()
            self.transforms.register(Normalize(), probability=1.0)
            if config.tf == "all_tf":
                self.transforms.register(Flip(), probability=0.5)
                self.transforms.register(Blur(sigma=(0.1, 1)), probability=0.5)
                self.transforms.register(Noise(sigma=(0.1, 1)), probability=0.5)
                self.transforms.register(Cutout(patch_size=np.ceil(np.array(config.input_size) / 4)), probability=0.5)
                self.transforms.register(Crop(np.ceil(0.75 * np.array(config.input_size)), "random", resize=True), probability=0.5)
            elif config.tf == "cutout":
                self.transforms.register(Cutout(patch_size=np.ceil(np.array(config.input_size) / 4)), probability=1)
            elif config.tf == "crop":
                self.transforms.register(Crop(np.ceil(0.75 * np.array(config.input_size)), "random", resize=True), probability=1)
        elif validation:
            self.transforms = Transformer()
            self.transforms.register(Normalize(), probability=1.0)

        # Load data and labels
        if training:
            self.data = np.load(config.data_train)
            self.labels = pd.read_csv(config.label_train)
        elif validation:
            self.data = np.load(config.data_val)
            self.labels = pd.read_csv(config.label_val)

        assert self.data.shape[1:] == tuple(config.input_size), "3D images must have shape {}".format(config.input_size)

    def collate_fn(self, list_samples):
        list_x = torch.stack([torch.as_tensor(x, dtype=torch.float) for (x, y, dx) in list_samples], dim=0)
        list_y = torch.stack([torch.as_tensor(y, dtype=torch.float) for (x, y, dx) in list_samples], dim=0)
        list_dx = torch.stack([torch.as_tensor(dx, dtype=torch.long) for (x, y, dx) in list_samples], dim=0)

        return (list_x, list_y, list_dx)

    def __getitem__(self, idx):
        np.random.seed()
        x = self.transforms(self.data[idx])  # Apply normalization or transformations
        labels = self.labels[self.config.label_name].values[idx]
        dx = self.labels.loc[idx, 'stroke']

        return (x, labels, dx)

    def __len__(self):
        return len(self.data)
