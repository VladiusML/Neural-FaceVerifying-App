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

anchor_files = get_files_in_directory(ANC_PATH, NUM_FILES)
positive_files = get_files_in_directory(POS_PATH, NUM_FILES)
negative_files = get_files_in_directory(NEG_PATH, NUM_FILES)

positive_dataset = SiameseDataset(anchor_files, positive_files, "POS")
negative_dataset = SiameseDataset(anchor_files, negative_files, "NEG")

dataset = ConcatDataset([positive_dataset, negative_dataset])
indices = list(range(len(dataset)))
train_indices, test_indices = train_test_split(indices, test_size = 0.25, random_state=42)

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False)

model = SiameseNN()
model.to(DEVICE)

optimizer = optim.Adagrad(model.parameters(), lr=0.001)   
loss_fn = nn.BCELoss() 

checkpoint_path = r"D:\Facial Recognition App\models"
train_loss = train_model_with_early_stopping(model, train_loader, optimizer, loss_fn, DEVICE, EPOCHS, checkpoint_path)
test_loss = test_model(model, test_loader, loss_fn, DEVICE)

print(f"Best Train Loss: {train_loss:.4f}")
print(f"Mean Test Loss: {test_loss:.4f}")
