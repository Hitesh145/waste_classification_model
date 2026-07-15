from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "Data"
dataset_folder_path = str(DATA_DIR / "dataset-resized")

MODEL_DIR = ROOT_DIR / "models" / "trained_model"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
LEARNING_RATE = 0.001
MOMENTUM = 0.8
NUM_EPOCHS = 30
IMAGE_SIZE = 224

MODEL_NAME = "custom_cnn"  # Options: "custom_cnn", "resnet18", "resnet34", "resnet50"


def get_model_path(model_name):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR / f"{model_name}.pth"


def get_checkpoint_path(model_name):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"checkpoint_{model_name}.pth"
