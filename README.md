# Improvement Suggestions

## Critical Fixes
1. ✅ **Fix regularization bug in UpdateWeights**: Line 313 uses incorrect index (i instead of k) - FIXED
2. ✅ **Fix NewParamsFull function**: Function signature has 6 parameters but only 5 are used - FIXED
3. ✅ **Fix NaN issues**: Handle potential NaN values in backpropagation - FIXED
4. ✅ **Fix thread safety issues**: Current concurrent backpropagation has race conditions - FIXED

## Functional Improvements
1. **Improve activation function selection**: Set better default activations for each layer type
2. **Implement momentum SGD**: Would improve convergence as noted in TODOs
3. **Implement batch normalization**: Would stabilize learning and prevent NaNs
4. **Add model saving/loading**: Important for reusing trained models
5. **Fix random seed issue**: Current seeding method isn't reliable; use time-based seeds
6. **Implement CIFAR color support**: Using color information would improve accuracy
7. **Optimize matrix operations**: Reuse allocated memory with SubVec, MulVec, etc.
8. **Improve error handling**: Replace silent failures and panics with proper error handling

## Current Architecture:
Input layer: 1024 neurons (for 32x32 grayscale image)
Hidden layer 1: 256 neurons, ReLU activation
Hidden layer 2: 128 neurons, ReLU activation
Hidden layer 3: 10 neurons, Leaky ReLU activation
Output layer: softmax activation

"10 to 50 epochs for medium-sized datasets.
50 to 200 epochs for very large datasets."

example of CIFAR:
https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/cifar10_tutorial.ipynb#scrollTo=smL1ykHVXO75

        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

		Pytorch result

		batch size used by default is 4.

		40 batch Accuracy of the network on the 10000 test images: 40 %


Greyscale: Accuracy of the network on the 10000 test images: 30 %


<!-- [1,  2000] loss: 1.913
[1,  4000] loss: 1.843
[1,  6000] loss: 1.822
[1,  8000] loss: 1.805
[1, 10000] loss: 1.793
[1, 12000] loss: 1.770
[2,  2000] loss: 1.773
[2,  4000] loss: 1.775
[2,  6000] loss: 1.757
[2,  8000] loss: 1.759
[2, 10000] loss: 1.763
[2, 12000] loss: 1.754 -->
<!-- Accuracy of plane : 26 %
Accuracy of   car : 41 %
Accuracy of  bird : 22 %
Accuracy of   cat : 18 %
Accuracy of  deer : 48 %
Accuracy of   dog : 35 %
Accuracy of  frog : 32 %
Accuracy of horse : 34 %
Accuracy of  ship : 56 %
Accuracy of truck : 60 % -->

Grey scale
L = 0.2989 * R + 0.5870 * G + 0.1140 * B.

results:
Loss SGD 0 = 728.76
train 0.29666665
validation 0.2

TODO: learn more on Stochastic Gradient Descent (SGD) ?

Profile:
go build -o load  
./load -cpuprofile=cpu.prof
go tool pprof cpu.prof
>top10

 <!-- func ClipGradient(gradients []float32, clipValue float32) []float32 {
        norm := float32(0)
        for _, g := range gradients {
            norm += g * g
        }
        norm = float32(math.Sqrt(float64(norm)))
        
        if norm > clipValue {
            scale := clipValue / norm
            for i := range gradients {
                gradients[i] *= scale
            }
        }
        return gradients
    } -->

TODO: add multythreading guards since number of workes affects result

TODO: preprocess to [-1, 1] range
TODO: better gradient descent (momentum)
TODO: drop neurons instead of L2
TODO: use colors of image

TODO: 2nd order gradient descent ?
TODO: save/load model

TODO: measure perfomance/time
TODO: compare float32 vs float64 perfomance
TODO: consider "manual" GC
TODO: put it into Docker
TODO: since this images augumentation may help

TODO?:
func init() {
	mat64.Register(goblas.Blas{})
}

## Code Description

### Overview

This code implements a simple neural network for image classification using the CIFAR-10 dataset. The network is written in Go and utilizes the Gonum library for matrix operations.

### Key Components

-   **Neural Network (`neuralnet/neuralnet.go`):**
    -   The core of the project. Implements a feedforward neural network.
    -   Supports various activation functions (`neuralnet/activations.go`) like ReLU, LeakyReLU, Sigmoid, and Tanh.
    -   Includes training algorithms like SGD and Mini-Batch.
    -   Now supports saving/loading.
    - Includes momentum and batch normalisation.
- **Data Loading (`load.go`):**
    -   Handles loading and preprocessing the CIFAR-10 dataset.
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
- **Batch Normalization**: Batch normalization has been added.
- **Model Saving/Loading**: Now the model can be saved and loaded.

### Future Work
-   **CIFAR color support**: Using color information would improve accuracy
-   **Optimize matrix operations**: Reuse allocated memory with SubVec, MulVec, etc.

### Installation

To use this code you must copy and paste:

Copy and paste: Copy the contents of `neuralnet` directory into your own project.

Import as a module: Place `neuralnet` directory within your project and use it as a module:
[end of README.md]
