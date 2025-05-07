# gon: CIFAR-10 Neural Network

## Overview
This Go project implements a feedforward neural network for CIFAR-10 image classification. It supports multi-threaded mini-batch training, momentum SGD, and model save/load via JSON, using standard Go slices for internal calculations.

## Installation
```bash
go build -o gon
```

## Usage
```bash
./gon -cpuprofile=cpu.prof
```

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
    -   Handles loading the CIFAR-10 dataset (binary format).
    -   Includes functions to convert RGB images to grayscale and flatten them.
- **Activation Functions (`neuralnet/activations.go`):**
    -   Defines various activation functions and their derivatives.
### Usage

The code is intended to be used as a library for learning and experimenting with neural networks.

### Improvements

The following improvements have been implemented:

- **Error Handling**: Proper error handling is implemented instead of panics.
- **Thread Safety**: Race conditions are now handled in Mini-Batch.
- **Momentum SGD**: Momentum has been added.
- **Model Saving/Loading**: The model can be saved and loaded.
- **Parallelism**: Mini-batch training and accuracy calculation are parallelized.
- **Configuration**: Key hyperparameters are configurable via flags.
- **Dependency Removal**: Removed dependency on Gonum.

### Future Work
-   **CIFAR color support**: Using color information should improve accuracy.
-   **Dropout/Batch Norm**: Implement other regularization techniques.

### Installation

To use this code you must copy and paste:

Copy and paste: Copy the contents of `neuralnet` directory into your own project.

Import as a module: Place `neuralnet` directory within your project and use it as a module:
[end of README.md]
