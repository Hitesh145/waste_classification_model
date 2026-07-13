
import torch
from models.model import get_model
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
