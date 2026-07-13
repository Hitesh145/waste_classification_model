from torchvision.datasets import ImageFolder
import config
from utils.loader import get_data_loaders, show_class_distribution
from preprocessing.eda import analyse_and_plot_dataset
from models.train import train_model
from models.test import test_model


def main():
    dataset = ImageFolder(root=config.dataset_folder_path)
    num_classes = len(dataset.classes)

    train_loader, val_loader, test_loader = get_data_loaders(dataset)

    # show_class_distribution(
    #     dataset,
    #     train_loader.dataset.labels,
    #     val_loader.dataset.labels,
    #     test_loader.dataset.labels
    # )

    running = True

    while running:
        print(f"Current model being used: {config.MODEL_NAME}")
        print("Do you want to train or test the model? (train/test/analyze/quit)")

        choice = input().lower()

        if choice == "train" or choice == "1":
            print("You chose train.")
            train_model(config.MODEL_NAME, train_loader, val_loader, num_classes, dataset.classes)

        elif choice == "test" or choice == "2":
            print("You chose test.")
            test_model(config.MODEL_NAME, test_loader, config.get_model_path(config.MODEL_NAME), num_classes)

        elif choice == "analyze" or choice == "3":
            print("You chose analyze.")
            analyse_and_plot_dataset(config.dataset_folder_path)

        else:
            print("Quitting the program.")
            running = False


if __name__ == "__main__":
    main()
