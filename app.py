from torchvision.datasets import ImageFolder
import config
from utils.loader import get_data_loaders, show_class_distribution
from preprocessing.eda import analyse_and_plot_dataset
from models.train import train_model
from models.test import test_model , test_single_image
# from utils.metrics import evaluation_model
from evaluation import eval_metrics


def main():
    print("Loading dataset...")
    dataset = ImageFolder(root=config.dataset_folder_path)
    num_classes = len(dataset.classes)

    train_loader, val_loader, test_loader = get_data_loaders(dataset)
    print(f"Loaded {len(dataset)} images across {num_classes} classes.")
    print(f"Name of the Classes in dataset: {dataset.classes}")

    # show_class_distribution(
    #     dataset,
    #     train_loader.dataset.labels,
    #     val_loader.dataset.labels,
    #     test_loader.dataset.labels
    # )

    running = True

    while running:
        print(f"Current model being used: {config.MODEL_NAME}")
        print("Choose an option: \n1. train \n2. test \n3. analyze \n4. test_single_image \n5. evaluate \n6. quit")

        try:
            choice = input("> ").strip().lower()
        except EOFError:
            print("No input received. Quitting the program.")
            break

        if choice == "train" or choice == "1":
            print("You chose train.")
            train_model(config.MODEL_NAME, train_loader, val_loader, num_classes, dataset.classes)

        elif choice == "test" or choice == "2":
            print("You chose test.")
            test_model(config.MODEL_NAME, test_loader, config.get_model_path(config.MODEL_NAME), num_classes)

        elif choice == "analyze" or choice == "3":
            print("You chose analyze.")
            analyse_and_plot_dataset(config.dataset_folder_path)

        elif choice == "test_single_image" or choice == "4":
            print("You chose test single image.")
            test_single_image()

        elif choice == "evaluate" or choice == "5":
            print("You chose evaluate.")
            eval_metrics(config.MODEL_NAME, test_loader, config.get_model_path(config.MODEL_NAME), num_classes , dataset.classes)
        

        elif choice == "quit" or choice == "q" or choice == "6":
            print("Quitting the program.")
            running = False

        else:
            print("Invalid choice. Please type train, test, analyze, or quit.")
            running = False


if __name__ == "__main__":
    main()

    
