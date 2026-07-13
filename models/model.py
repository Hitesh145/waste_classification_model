import torch.nn as nn
import torchvision.models as models

# Custom CNN Model
class WasteClassifierCustom(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 12 * 12, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ResNet50 CNN Model

class WasteClassifierResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet50(weights=weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
    
# ResNet34 CNN Model

class WasteClassifierResNet34(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet34(weights=weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# ResNet18 CNN Model

class WasteClassifierResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x):
        return self.resnet(x)
    

def get_model(model_name, num_classes, pretrained=True):
    if model_name == "custom_cnn":
        return WasteClassifierCustom(num_classes)
    
    elif model_name == "resnet34":
        return WasteClassifierResNet34(num_classes, pretrained=pretrained)
    
    elif model_name == "resnet50":
        return WasteClassifierResNet50(num_classes, pretrained=pretrained)
    
    elif model_name == "resnet18":
        return WasteClassifierResNet18(num_classes, pretrained=pretrained)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# model18 = models.resnet18(weights='IMAGENET1K_V1')
# model18.fc = nn.Linear(model18.fc.in_features, num_classes)
# model18 = model18.to(device)


# model34 = models.resnet34(weights='IMAGENET1K_V1')
# model34.fc = nn.Linear(model34.fc.in_features, num_classes)
# model34 = model34.to(device)

