import numpy as np
import pandas as pd
import os
import config
import matplotlib.pyplot as plt
from PIL import Image


def analyse_and_plot_dataset(dataset_folder_path):

    print(f"Analysing dataset in folder: {dataset_folder_path}")
   
    # Define the path to your dataset folder in Google Drive

    exists = check_dataset_folder_exists(dataset_folder_path)
    if not exists:
        return
    
    images_counts , sizes , modes , bad_images = analysize_dataset(dataset_folder_path)

    # DataFrame of the dataset

    data = pd.DataFrame(list(images_counts.items()), columns=['Class', 'Count'])

    print("Images counts: ", images_counts)
    print("Sizes: ",[sizes[cls][:1] for cls in sizes])  # Show first 5 sizes for each class
    print("Modes: ", modes)
    print("Bad images: ", bad_images)
    print(data)

    # Data Visualization
    plt.figure(figsize=(10, 6))
    plt.bar(data['Class'], data['Count'])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.show()



# Check if the folder exists

def check_dataset_folder_exists(folder_path):
    if os.path.exists(folder_path):
        print(f'Accessing folder: {folder_path}')
        # List the contents of the folder
        print('Contents of the dataset-resized folder:')
        print(os.listdir(folder_path))

        return True
    else:
        print(f'The folder {folder_path} does not exist. Please check the path and try again.')
        return False




# Analysing the Data


def analysize_dataset(dataset_folder_path):
    classes = os.listdir(dataset_folder_path)
    print("Classes are: ", classes)

    # Initialize dictionaries to hold counts, sizes, and modes
    images_counts = {}
    sizes = {}
    modes = {}
    bad_images = []

    for cls in classes:
        size = []
        folder = os.path.join(dataset_folder_path, cls)
        count = 0
        for f in os.listdir(folder):
            if f.lower().endswith((".jpeg", ".jpg", ".png")):
                img_path = os.path.join(folder, f)
                try:
                    img = Image.open(img_path)
                    img.verify()  # Verify that the image is not corrupted
                    mode = img.mode
                    modes[mode] = modes.get(mode, 0) + 1
                    size.append(img.size)
                    count += 1

                except Exception as e:
                    bad_images.append(img_path)

        sizes[cls] = size
        images_counts[cls] = count

    return images_counts, sizes, modes, bad_images

