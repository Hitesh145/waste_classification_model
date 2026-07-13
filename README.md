# Simple Waste Management Image Classifier

This project trains and evaluates image classification models for waste categories such as cardboard, glass, metal, paper, plastic, and trash.

## Features

- Custom CNN model
- ResNet18, ResNet34, and ResNet50 transfer-learning models
- Train/validation/test split with stratification
- Class-weighted loss for imbalanced datasets
- Dataset analysis and class distribution plotting
- CLI menu for training, testing, and analysis

## Project Structure

```text
Simple_Waste_Mgt_sys/
├── app.py
├── config.py
├── models/
│   ├── model.py
│   ├── train.py
│   └── test.py
├── preprocessing/
│   ├── eda.py
│   └── tranform.py
├── utils/
│   └── loader.py
└── requirements.txt
```

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Add your dataset at:

```text
Data/dataset-resized/
```

The dataset should be arranged in class folders:

```text
Data/dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

## Usage

Run the main menu:

```bash
python app.py
```

Then choose:

- `train` to train the selected model
- `test` to evaluate a saved model
- `analyze` to inspect dataset statistics
- `quit` to exit

The active model is set in `config.py`:

```python
MODEL_NAME = "custom_cnn"
```

Supported values:

- `custom_cnn`
- `resnet18`
- `resnet34`
- `resnet50`

## Notes

The dataset and trained model files are ignored by Git because they are large and machine-specific. If you want to share trained weights, upload them separately as a release asset, cloud drive link, or model registry artifact.
