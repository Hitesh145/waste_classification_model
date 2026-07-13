import pandas as pd
from PIL import Image 
from torch.utils.data import DataLoader, Dataset
from preprocessing.tranform import train_transform, test_transform

import config

class wasteDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index]).convert("RGB")

        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label


from sklearn.model_selection import train_test_split

def split_dataset(dataset):
    image_paths = [sample[0] for sample in dataset.samples]
    labels = [sample[1] for sample in dataset.samples]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    stratify=temp_labels,
    random_state=42
    )
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


from collections import Counter

BATCH_SIZE = config.BATCH_SIZE

def get_data_loaders(dataset, batch_size=BATCH_SIZE):
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_dataset(dataset)

    train_dataset = wasteDataset(train_paths, train_labels, train_transform)
    val_dataset = wasteDataset(val_paths, val_labels, test_transform)
    test_dataset = wasteDataset(test_paths, test_labels, test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader



# print(train_loader.dataset.__getitem__(0))

def show_class_distribution(dataset , train_labels, val_labels, test_labels):
    

    train_count = Counter(train_labels)
    val_count = Counter(val_labels)
    test_count = Counter(test_labels)

    # Corrected: Use dataset.classes to align with the labels
    df = pd.DataFrame({
        "Class": dataset.classes,
        "Train": [train_count[i] for i in range(len(dataset.classes))],
        "Validation": [val_count[i] for i in range(len(dataset.classes))],
        "Test": [test_count[i] for i in range(len(dataset.classes))]
    })

    df['Total'] = df['Train'] + df['Validation'] + df['Test']

    print(df)
