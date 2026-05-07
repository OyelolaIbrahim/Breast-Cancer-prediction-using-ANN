# Breast-Cancer-prediction-using-ANN

![Python](https://img.shields.io/badge/Python-3.x-blue)  ![Framework](https://img.shields.io/badge/Framework-NumPy%20only-blue)  ![Task](https://img.shields.io/badge/Task-Binary%20Classification-blue)  
## Overview
A complete Artificial Neural Network (ANN) built entirely from scratch using only NumPy — no TensorFlow, no PyTorch, no Scikit-Learn model. This project demonstrates deep understanding of neural network internals by implementing every component manually: weight initialisation, forward propagation, sigmoid activation, binary cross-entropy loss, and gradient-based backpropagation.
## Problem Statement
Classify breast tumours as Malignant (M) or Benign (B) using clinical diagnostic measurements. The goal is to build and understand a neural network from the ground up, without relying on high-level ML frameworks.
## Dataset

- **Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** [Download from Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Size:** 569 patient records
- **Features:** 30 numeric diagnostic measurements including radius, 
  texture, perimeter, area, smoothness, compactness, concavity, 
  symmetry, and fractal dimension
- **Classes:** Malignant — M (1), Benign — B (0)
- **Format:** CSV file with an `id` column, a `diagnosis` column, 
  and 30 numeric feature columns
- **Instructions:** Download `data.csv` from the link above and 
  place it in the root folder of this project before running 
  the notebook.

## What Was Built from Scratch
-	Weight initialisation using np.random.rand()
-	Sigmoid activation function: 1 / (1 + exp(-x))
-	Binary cross-entropy loss function: -log(yhat) for y=1, -log(1-yhat) for y=0
-	Forward propagation: yhat = sigmoid(X @ weights)
-	Backpropagation: gradient = X.T @ (-2 * (y - yhat) * sigmoid(yhat))
-	Weight update: weights -= gradient
-	Mean Squared Error (MSE) tracking over 1,000 training iterations
-	Training convergence visualisation using Matplotlib loss curves
## Results

| Property            | Details                                               |
| ------------------- | ----------------------------------------------------- |
| Architecture        | Single-layer Perceptron (custom NumPy implementation) |
| Activation Function | Sigmoid                                               |
| Loss Function       | Binary Cross-Entropy (with MSE tracking)              |
| Training Iterations | 1,000                                                 |
| Evaluation Method   | MSE Loss Curve Visualization                          |


## Technologies
Python, NumPy, Pandas, Matplotlib, Scikit-Learn (data loading only)
## How to Run

```bash
git clone https://github.com/OyelolaIbrahim/breast-cancer-ann-from-scratch.git
cd breast-cancer-ann-from-scratch
pip install numpy pandas matplotlib scikit-learn
jupyter notebook breast_cancer_ann.ipynb
```

