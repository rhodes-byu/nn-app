# MLP Playground

This is a small Streamlit app for exploring how a multilayer perceptron (MLP) learns on simple 2D classification datasets.

The app lets you change the dataset, network architecture, activation function, optimizer, regularization, and training settings, then inspect the learned decision boundary and training diagnostics.

## Features

- Choose from several toy datasets: Moons, Circles, Gaussian Blobs, XOR, and Spiral
- Adjust hidden layers, units, activation, learning rate, batch size, epochs, and regularization
- View dataset splits, learned decision boundaries, loss curves, accuracy curves, and a test confusion matrix
- Save training runs and compare them inside the app
- Use the built-in concept guide for simple MLP experiments

## Requirements

- Python 3.10+
- Packages listed in `requirements.txt`

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run nn-app.py
```

Then open the local Streamlit URL shown in your terminal.

## Notes

- The app uses `sklearn.neural_network.MLPClassifier` for training
- The "feature dropout" control is a teaching approximation, since scikit-learn does not provide hidden-layer dropout in `MLPClassifier`
