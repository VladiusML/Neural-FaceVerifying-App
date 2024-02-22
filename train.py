from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import train_test_split

from model.SNN import SiameseNN
from Dataset import SiameseDataset
from scripts.utils import get_files_in_directory
from scripts.train_utils import train_model_with_early_stopping, test_model
from scripts.constants import POS_PATH, NEG_PATH, ANC_PATH, NUM_FILES, EPOCHS, DEVICE

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Get file paths for anchor, positive, and negative images
anchor_files = get_files_in_directory(ANC_PATH, NUM_FILES)
positive_files = get_files_in_directory(POS_PATH, NUM_FILES)
negative_files = get_files_in_directory(NEG_PATH, NUM_FILES)

# Create datasets for positive and negative pairs
positive_dataset = SiameseDataset(anchor_files, positive_files, "POS")
negative_dataset = SiameseDataset(anchor_files, negative_files, "NEG")

# Concatenate positive and negative datasets
dataset = ConcatDataset([positive_dataset, negative_dataset])

# Split dataset into train and test sets
indices = list(range(len(dataset)))
train_indices, test_indices = train_test_split(indices, test_size=0.25, random_state=42)

# Create train and test subsets
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Create data loaders for train and test sets
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Instantiate Siamese Neural Network model
model = SiameseNN()
model.to(DEVICE)  # Move model to the specified device (e.g., GPU)

# Define optimizer and loss function
optimizer = optim.Adagrad(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()  # Binary Cross Entropy Loss

# Specify checkpoint path for saving model checkpoints during training
checkpoint_path = r"D:\Facial Recognition App\models"

# Train the model with early stopping and get the best training loss
train_loss = train_model_with_early_stopping(model, train_loader, optimizer, loss_fn, DEVICE, EPOCHS, checkpoint_path)

# Test the trained model and get the mean test loss
test_loss = test_model(model, test_loader, loss_fn, DEVICE)

# Print the best training loss and mean test loss
print(f"Best Train Loss: {train_loss:.4f}")
print(f"Mean Test Loss: {test_loss:.4f}")
