import torch
from torch.utils.data import Dataset
from scripts.utils import read_and_preprocess

class SiameseDataset(Dataset):
    def __init__(self, anchor_files, siamese_files, type_siamese):
        """
        Initializes the Siamese Dataset.

        Args:
        - anchor_files (list): List of file paths for anchor images.
        - siamese_files (list): List of file paths for siamese images.
        - type_siamese (str): Type of siamese relationship ('POS' for positive, 'NEG' for negative).

        Raises:
        - ValueError: If type_siamese is not 'POS' or 'NEG'.
        """
        self.anchor_files = anchor_files
        self.siamese_files = siamese_files

        # Assign labels based on siamese relationship type
        if type_siamese == "POS":
            self.labels = torch.ones(len(anchor_files))
        elif type_siamese == "NEG":
            self.labels = torch.zeros(len(anchor_files))
        else:
            raise ValueError("Unsupported type_siamese. Supported types are 'POS' and 'NEG'")
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
        - int: Total number of samples.
        """
        return len(self.anchor_files)
    
    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset by index.

        Args:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - tuple: A tuple containing the anchor image, siamese image, and label.
        """
        anchor_path  = self.anchor_files[idx]
        siamese_path = self.siamese_files[idx]

        # Read and preprocess anchor and siamese images
        anchor_img = read_and_preprocess(anchor_path)
        siamese_img = read_and_preprocess(siamese_path)

        # Get the label for the sample
        label = self.labels[idx]

        return (anchor_img, siamese_img, label)
