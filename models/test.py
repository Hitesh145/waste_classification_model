
import torch
from models.model import get_model
from preprocessing import tranform
import config

def test_model(model_name, test_loader, modelpath, num_classes):

    model = get_model(model_name, num_classes, pretrained=False)
    model = model.to(config.device)

    model.load_state_dict(torch.load(modelpath, map_location=config.device))

    model.eval()
    correct = 0
    total = 0
    print(f"Testing model: {model_name} using model file: {modelpath}")
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print(f"Predicted: {predicted}, Actual: {labels}")

    print("Testing completed.")
    accuracy = 100 * correct / total
    print(f"Test Accuracy of the model on the {total} test images: {accuracy}%")
    return accuracy


def test_single_image():
    from PIL import Image
    from torchvision.datasets import ImageFolder

    # Load the image
    image_path = input("Enter the path of the image to test (relative to the current directory): ")
    image_path = image_path.strip()  # Remove any leading/trailing whitespace
    image = Image.open("test_image/"+image_path)

    # Apply the transformations to the image
    image = tranform.test_transform(image).unsqueeze(0)  # Add batch dimension

    # Load the model
    model_name = config.MODEL_NAME
    num_classes = len(ImageFolder(root=config.dataset_folder_path).classes)
    model = get_model(model_name, num_classes, pretrained=False)
    model = model.to(config.device)
    model.load_state_dict(torch.load(config.get_model_path(model_name), map_location=config.device))
    model.eval()

    # Make prediction
    with torch.no_grad():
        image = image.to(config.device)
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_class_index = predicted.item()
        predicted_class_name = ImageFolder(root=config.dataset_folder_path).classes[predicted_class_index]

    print(f"Predicted class index: {predicted_class_name} (Class Index: {predicted_class_index})")
