# gon: CIFAR-10 Neural Network



## Hyperparameter Tuning (Grid Search Suggestion)

To find optimal hyperparameters, a grid search could be implemented. Here's a suggested set of parameters and values targeting ~96 combinations:

- **Learning Rate (`-lr`)**: `[0.05, 0.01, 0.005, 0.001]` (4 values)
- **L2 Regularization** (Requires adding a flag, e.g., `-l2`): `[0, 1e-5, 1e-4, 1e-3]` (4 values)
- **Momentum Coefficient** (Requires adding a flag, e.g., `-momentum`): `[0.9, 0.95]` (2 values)
- **Mini-batch Size (`-batch`)**: `[32, 64, 128]` (3 values)

Note: Implementing grid search would involve modifying `load.go` to iterate through these combinations, potentially running multiple training sessions and logging results. Flags would need to be added for L2 and Momentum.
[end of README.md]



## Overview
This Go project implements a feedforward neural network for CIFAR-10 image classification. It supports multi-threaded mini-batch training, momentum SGD, and model save/load via JSON, using standard Go slices for internal calculations.

## Installation
```bash
go build -o gon
```

## Usage
```bash
# Build the application
go build -o gon

# Run with default settings
./gon

# Run with specific hyperparameters and profiling enabled
./gon -lr=0.005 -decay=0.98 -epochs=20 -batch=128 -workers=8 -cpuprofile=cpu.prof

# Load a saved best model and evaluate
./gon -load=best_model.json
```
Flags:
- `-lr`: Learning rate (default: 0.01)
- `-decay`: Learning rate decay factor per epoch (default: 0.95)
- `-epochs`: Number of training epochs (default: 10)
- `-batch`: Mini-batch size (default: 64)
- `-workers`: Number of parallel workers for training/accuracy (default: number of CPU cores)
- `-cpuprofile`: File path to write CPU profile data (e.g., cpu.prof)
- `-load`: Path to a saved model JSON file to load and run predictions

## Project Structure
- load.go: Loads and preprocesses the CIFAR-10 dataset.
- neuralnet/: Core neural network implementation, activation functions, training routines.
- run.sh: Helper script to build and run the application.

## Features
- Thread-safe mini-batch training with worker cloning  
- Momentum-based stochastic gradient descent  
- Model persistence (save/load) via JSON  
- Configurable worker count with MAX_WORKERS cap
- Training/validation split
- Command-line flags for LR, decay, epochs, batch size, workers
- Intermediate calculation precision standardized (using float64 internally)
- Performance timing output

## Future Work / TODOs
- Use color image data instead of converting to grayscale
- Implement dropout regularization
- Implement batch normalization
- Explore image augmentation techniques
- Explore other optimizers (e.g., Adam) or 2nd order methods
- Preprocess input data to [-1, 1] range or standardize
- Add more comprehensive unit and integration tests
- Consider manual garbage collection tuning (if necessary)
- Implement Grid Search for hyperparameter tuning
- Dockerization and CI/CD integration

## Code Description

### Overview

This code implements a simple feedforward neural network for image classification using the CIFAR-10 dataset. The network is written in Go using standard slices for numerical operations.

### Key Components

-   **Neural Network (`neuralnet/neuralnet.go`):**
    -   The core of the project. Implements a feedforward neural network.
    -   Supports various activation functions (`neuralnet/activations.go`) like ReLU, LeakyReLU, Sigmoid, and Tanh.
    -   Includes training algorithms like SGD and Mini-Batch.
    -   Supports saving/loading the trained model via JSON.
    -   Implements momentum for SGD.
- **Data Loading (`load.go`):**
    -   Handles loading the CIFAR-10 dataset from the specific binary batch format (1 label byte + 3072 pixel bytes per image).
    -   Includes functions to convert RGB images to grayscale and flatten them for the MLP input layer.
- **Activation Functions (`neuralnet/activations.go`):**
    -   Defines various activation functions and their derivatives.
### Usage

The project provides a runnable application (`load.go`) for training and evaluating the network on CIFAR-10. The `neuralnet` package can also potentially be used as a library in other Go projects.

### Improvements

The following improvements have been implemented:

- **Thread Safety**: Race conditions are handled in Mini-Batch training and accuracy calculation using network cloning.
- **Momentum SGD**: Momentum has been added.
- **Model Saving/Loading**: The model can be saved and loaded.
- **Parallelism**: Mini-batch training and accuracy calculation are parallelized.
- **Configuration**: Key hyperparameters are configurable via flags.
- **Dependency Removal**: Removed dependency on Gonum.

### Future Work
-   **CIFAR color support**: Using color information should improve accuracy.
-   **Dropout/Batch Norm**: Implement other regularization techniques.

### Installation

Build the executable using Go:
```bash
go build -o gon
```
