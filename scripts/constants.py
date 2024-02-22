import os
import torch

# Define various constants for the project (the most important thing here are the constants of data paths)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_FILES = 3500
EPOCHS = 20

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')
INPUT_PATH = os.path.join("application_data", "input_image")
VERIFIY_PATH = os.path.join("application_data", "verification_images")
WEIGHTS_PATH =  os.path.join("model_weghts")