from loaders.loaders import KspSensImgLoader
from torch.utils.data import DataLoader
import torch

class FastMRIDataset:
    """FastMRI dataset loader with configurable parameters."""

    def __init__(self, config):
        """
        Initialize FastMRI dataset loader.

        Args:
            config (dict): Dataset configuration containing:
                - data_path (str): Path to the dataset
                - device (str): Device for data loading
                - batch_size (int): Batch size for DataLoader
                - shuffle (bool): Whether to shuffle the data
                - num_workers (int): Number of workers for DataLoader
                - pin_memory (bool): Whether to pin memory
        """
        self.config = config
        self.data_path = config.get('data_path', '')
        self.device = config.get('device', 'cuda:0')
        self.batch_size = config.get('batch_size', 1)
        self.shuffle = config.get('shuffle', False)
        self.num_workers = config.get('num_workers', 0)
        self.pin_memory = config.get('pin_memory', False)

        # Initialize dataset
        self.dataset = KspSensImgLoader(self.data_path, device=self.device)

        # Get shape information from first sample
        if len(self.dataset) > 0:
            kspace, coils, img, fname = self.dataset[0]
            _, C, M, N = kspace.shape
            self.shape = [1, M, N]
            self.C, self.M, self.N = C, M, N
        else:
            self.shape = None
            self.C, self.M, self.N = None, None, None

        # Initialize DataLoader
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def __iter__(self):
        """Return iterator for the DataLoader."""
        return iter(self.loader)

    def __len__(self):
        """Return length of the dataset."""
        return len(self.dataset)

    def get_shape(self):
        """Get the shape of the data [batch_size, M, N]."""
        return self.shape

    def get_dimensions(self):
        """Get dimensions (C, M, N) of the data."""
        return self.C, self.M, self.N


def create_fastmri_loader(config):
    """
    Factory function to create FastMRI dataset loader.

    Args:
        config (dict): Dataset configuration

    Returns:
        FastMRIDataset: Configured dataset loader
    """
    return FastMRIDataset(config)
