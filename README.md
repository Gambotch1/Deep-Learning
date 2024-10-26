# Deep Learning

## Overview
This repository provides implementations of neural network models for deep learning tasks such as image classification using the MNIST dataset. The package includes pre-built models, training scripts, evaluation utilities and visualisation tools to help users understand core concepts of deep learning.

## Features
- Pre-built models: MLP, CNN, etc.
- Dataset preparation and handling (e.g., MNIST).
- Training and evaluation scripts.
- Visualization utilities for results.

## Installation
### Prerequisites
- Python 38+
- `pip` for managing Python packages


## Setting ip the Project
1. Clone the Repository
```
git clone <repository-url>
```

2. Editable Mode Installation:
To install the package in editable mode for development purposes:
```
pip install -e .
```

## Usage
You can train the model using the provided script or run the `main.py` module directly
```
python src/train/main.py
```

## Testing
To run the tests, you can you:
```
pytest -v
```
## Visualisations
The repository includes pre-generated plots for easy interpretation of the results:
- Loss Cruve: These illustrate how the training and validation losses evlove over epochs
![alt text](https://github.com/Gambotch1/Deep-Learning/blob/main/images/losses.png)
- Accuracy Curves: These plots show how the model's accuracy improve with training
![alt text](https://github.com/Gambotch1/Deep-Learning/blob/main/images/accuracies.png)
