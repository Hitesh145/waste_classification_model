import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from models.model import get_model
import config
from sklearn.utils.class_weight import compute_class_weight

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

def train_model(
    model_name,
    train_loader,
    val_loader,
    num_classes,
    class_names=None,
    epochs=config.NUM_EPOCHS,
    learning_rate=config.LEARNING_RATE,
    momentum=config.MOMENTUM
):

    print (f"Training model: {model_name} for {epochs} epochs with learning rate: {learning_rate} and momentum: {momentum}")
    model = get_model(model_name, num_classes)
    train_labels = train_loader.dataset.labels

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=train_labels
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(config.device)
    
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    print(f'Using device: {config.device}')

    # Move the model to the detected device
    model.to(config.device)

    class_names = class_names or [f"class_{i}" for i in range(num_classes)]
    for class_name, weight in zip(class_names, class_weights):
        print(f"{class_name:15} : {weight:.3f}")

    best_accuracy = 0.0

    for epoch in range(epochs):
        print(f"Training epocs {epoch}...")

        model.train()  # Set the model to training mode
        correct = 0
        total = 0

        running_loss = 0.0
        val_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data

            # Move inputs and labels to the device
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)

            optimizer.zero_grad()
            outputs = model(inputs)

            

            loss = loss_fn(outputs, labels)

            loss.backward()

            optimizer.step()

            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
        total_train_loss = running_loss / len(train_loader)
        train_losses.append(total_train_loss)
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)

        print(f"Training_Loss : {total_train_loss:.4f}" + f" Accuracy: {100 * correct / total}%")

        model.eval()
        print(f"Validating epoch {epoch}...")
        val_loss = 0.0          # reset each epoch
        with torch.no_grad():
            correct = 0
            total = 0
            for data in val_loader:
                inputs, labels = data
                inputs = inputs.to(config.device)
                labels = labels.to(config.device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = loss_fn(outputs, labels)
                val_loss += loss.item()          # accumulate, don't overwrite

            total_val_loss = val_loss / len(val_loader)
            val_losses.append(total_val_loss)
            val_accuracy = 100 * correct / total
            val_accuracies.append(val_accuracy)
            print(f"Validation Accuracy: {val_accuracy}% and Validation Loss: {total_val_loss:.4f}")

            # Save the model if it has the best accuracy so far
            if val_accuracy > best_accuracy and val_accuracy > 80 :  # Save only if accuracy is greater than 50%
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), config.get_model_path(model_name))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, config.get_checkpoint_path(model_name))

            elif val_accuracy < best_accuracy:
                print(f"Validation accuracy decreased from {best_accuracy}% to {val_accuracy}%. Stopping training.")
                break


