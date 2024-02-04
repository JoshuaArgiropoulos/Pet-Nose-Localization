import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

class CustomDataset(Dataset):
    # Initialization method for the dataset
    def __init__(self, directory, is_train=True, transform=None):
        self.directory = directory
        self.is_train = is_train
        self.dataset_mode = "train"
        self.annotation_file = "labels.txt"

        # Switch to evaluation mode if is_train is False
        if not self.is_train:
            self.dataset_mode = "eval"
            self.annotation_file = "labels.txt"

        self.transform = transform
        self.image_folder = os.path.join(directory)
        self.label_folder = os.path.join(directory)
        self.total_images = []
        self.image_labels = {}

        # Read labels from the file and store them
        with open(os.path.join(self.label_folder, self.annotation_file), "r") as file:
            for line in file:
                parts = line.strip().split(",", 1)
                image_file = os.path.join(os.path.abspath(self.image_folder), parts[0])

                # Check if image file exists and add it to the list
                if os.path.isfile(image_file):
                    self.total_images.append(parts[0])
                    self.image_labels[parts[0]] = parts[1]

        self.dataset_length = len(self.total_images)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        image_file_name = ""
        try:
            image_path = os.path.join(self.image_folder, self.total_images[index])
            image_file_name = image_path
            image_data = cv2.imread(image_path)

            # Get image dimensions
            image_width, image_height = image_data.shape[1], image_data.shape[0]


             # Apply transformations
            if self.transform:
                image_data = self.transform(image_data)

            # Process the labels
            label_info = eval(self.image_labels[self.total_images[index]])
            label_parts = label_info[1:-1].split(", ")

            processed_labels = [
                float(label_parts[0]) / image_width,
                float(label_parts[1]) / image_height,
            ]
            
            # Convert label to tensor
            label_tensor = torch.tensor(processed_labels, dtype=torch.float32)
            return image_data, label_tensor
        except Exception as error:
            print(f"Error in __getitem__: {image_file_name} : {error}")
            return None, None

    # Iterator method to iterate through the dataset
    def __iter__(self):
        self.current_index = 0
        return self

    # Next method for the iterator
    def __next__(self):
        if self.current_index >= self.dataset_length:
            raise StopIteration
        self.current_index += 1
        return self.__getitem__(self.current_index - 1)
