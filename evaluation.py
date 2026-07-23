import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import config
from models.train import train_losses, val_losses, train_accuracies, val_accuracies
from models.model import get_model

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
# train_losses = [loss.to("cpu") for loss in train_losses] if train_losses is not None else []
# val_losses = [loss.to("cpu") for loss in val_losses] if val_losses is not None else []
# print("type of train_losses:", type(train_losses[0]))
# print("type of val_losses:", type(val_losses[0]))
# print("type of train_accuracies:", type(train_accuracies[0]))
# print("type of val_accuracies:", type(val_accuracies[0]))
def plot_loss(train_loss, val_loss):

    plt.figure(figsize=(8,5))

    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig("outputs/loss_curve.png")

    plt.show()


def plot_accuracy(train_acc, val_acc):

    plt.figure(figsize=(8,5))

    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title("Training vs Validation Accuracy")

    plt.legend()

    plt.grid(True)

    plt.savefig("outputs/accuracy_curve.png")

    plt.show()


def evaluate_model(
        model,
        test_loader,
        device,
        class_names,
        num_classes,
        modelpath
):
    model = get_model(model, num_classes, pretrained=False)
    model = model.to(config.device)
    
    model.load_state_dict(torch.load(modelpath, map_location=config.device))

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs,1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true,y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(y_true,y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    return accuracy, precision, recall, f1, cm, report



def plot_confusion_matrix(cm, class_names):

    plt.figure(figsize=(10,8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("outputs/confusion_matrix.png")

    plt.show()


def save_classification_report(report):

    with open(
        "outputs/classification_report.txt",
        "w"
    ) as f:

        f.write(report)


def print_metrics(
        accuracy,
        precision,
        recall,
        f1
):

    print("="*50)

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print("="*50)


def eval_metrics(
        model,
        test_loader,
        modelpath,
        num_classes,
        class_names,
        device = config.device,
        t_losses = train_losses,
        v_losses = val_losses,
        t_accuracies = train_accuracies,
        v_accuracies = val_accuracies
):
    print("type of train_losses:", type(train_losses[0]))
    print("type of val_losses:", type(val_losses[0]))
    print("type of train_accuracies:", type(train_accuracies[0]))
    print("type of val_accuracies:", type(val_accuracies[0]))
    # t_losses = [loss.item() for loss in train_losses] if train_losses is not None else []
    v_losses =v_losses = [v if isinstance(v, float) else v.cpu().item() for v in val_losses]
    print("train_losses:", t_losses)
    print("val_losses:", v_losses)
    print("train_accuracies:", t_accuracies)
    print("val_accuracies:", v_accuracies)
    
    accuracy, precision, recall, f1, cm, report = evaluate_model(
        model,
        test_loader,
        device,
        class_names,
        num_classes,
        modelpath
    )

    print_metrics(
        accuracy,
        precision,
        recall,
        f1
    )

    plot_loss(
        t_losses,
        v_losses
    )

    plot_accuracy(
        t_accuracies,
        v_accuracies
    )

    plot_confusion_matrix(
        cm,
        class_names
    )
    # print("train_losses:", t_losses)
    # print("val_losses:", v_losses)

    save_classification_report(report)