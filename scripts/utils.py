import os
import torchvision.transforms as transforms
import random
import numpy as np
import cv2
from .constants import INPUT_PATH, VERIFIY_PATH 

def read_and_preprocess(file_path):
    """
    Reads an image from the given file path, preprocesses it, and converts it to a PyTorch tensor.

    Args:
    - file_path (str): The path to the image file.

    Returns:
    - img (torch.Tensor): The preprocessed image as a PyTorch tensor.
    """
        
    img = cv2.imread(file_path)
    img = cv2.resize(img, (105,105))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img /255.0
    
    transform = transforms.Compose([
    transforms.ToTensor()
    ])

    img = transform(img)

    return img

def get_files_in_directory(directory, num_files):
    """
    Retrieves a random subset of files from the given directory.

    Args:
    - directory (str): The path to the directory containing the files.
    - num_files (int): The number of files to retrieve.

    Returns:
    - file_paths (list): A list of file paths.
    """
        
    file_paths = []
    if os.path.exists(directory) and os.path.isdir(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_paths.append(os.path.join(root, file))
    
    return random.sample(file_paths, min(num_files, len(file_paths)))

def verify_img(model, detection_threshold, verification_threshold):
    """
    Verifies an image using a pre-trained model.

    Args:
    - model: The pre-trained model.
    - detection_threshold (float): The detection threshold for considering a detection.
    - verification_threshold (float): The verification threshold for considering the verification successful.

    Returns:
    - results (list): A list of results from the model.
    - verified (bool): True if verification is successful, False otherwise.
    """
     
    results = []
    # Iterate over images in the verification directory
    for image in os.listdir(VERIFIY_PATH):
        input_img = read_and_preprocess(os.path.join(INPUT_PATH, "input_image.jpg"))
        validation_img = read_and_preprocess(os.path.join(VERIFIY_PATH, image))

        input_img, validation_img = input_img.float(), validation_img.float()
        
        model.eval()
        result = model(input_img, validation_img)
        results.append(result)

    # Count the number of detections above the detection threshold
    detection = np.sum((np.array([res.detach().numpy() for res in results]) > detection_threshold))
    # Calculate verification ratio
    verification = detection / len(os.listdir(os.path.join("application_data", "verification_images")))
    # Determine if the verification is successful based on the verification threshold
    verified = verification > verification_threshold
    
    return results, verified

def verify_webcam(model):
    """
    Captures images from a webcam and performs real-time verification using a pre-trained model.

    Args:
    - model: The pre-trained model.
    """

    cap = cv2.VideoCapture(0)

    while True: 
        ret, frame = cap.read() 
        frame = frame[120:120+250, 200:200+250, :]

        cv2.imshow("Verification", frame)

        if cv2.waitKey(10) & 0xFF == ord("v"):
            cv2.imwrite(os.path.join(INPUT_PATH, "input_image.jpg"), frame)
            results, verified = verify_img(model, 0.9, 0.7)
            print(verified)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
