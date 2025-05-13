package neuralnet

import "math"

type ActivationFunction interface {
	Activate(x float32) float32
	Derivative(x float32) float32
}

// ReLU (Rectified Linear Unit) activation function.
// Returns max(0, x). Commonly used in hidden layers for its simplicity and effectiveness.
type ReLU struct{}

func (r ReLU) Activate(x float32) float32 {
	return float32(math.Max(float64(x), 0.0))
}

func (r ReLU) Derivative(x float32) float32 {
	if x > 0 {
		return 1
	}
	return 0
}

// LeakyReLU is a variant of ReLU that allows a small, non-zero gradient (slope `alpha`) when the unit is not active (x < 0).
// Helps mitigate the "dying ReLU" problem.
type LeakyReLU struct {
	alpha float32 // Slope for negative inputs (e.g., 0.01 or 0.1).
}

func NewLeakyReLU(alpha float32) LeakyReLU {
	return LeakyReLU{alpha: alpha}
}

func (l LeakyReLU) Activate(x float32) float32 {
	if x > 0 {
		return x
	}
	return float32(float64(l.alpha) * float64(x))
}

func (l LeakyReLU) Derivative(x float32) float32 {
	if x > 0 {
		return 1
	}
	return l.alpha
}

// Sigmoid activation function.
// Squashes values into the range (0, 1). Historically used in hidden layers, but less common now due to vanishing gradients.
// Sometimes used in output layers for binary classification probabilities.
type Sigmoid struct{}

func (s Sigmoid) Activate(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

func (s Sigmoid) Derivative(x float32) float32 {
	sigmoidVal := s.Activate(x) // Call updated Activate
	return float32(float64(sigmoidVal) * (1.0 - float64(sigmoidVal)))
}

// Tanh (Hyperbolic Tangent) activation function.
// Squashes values into the range (-1, 1). Being zero-centered can sometimes help learning compared to Sigmoid.
type Tanh struct{}

func (t Tanh) Activate(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

func (t Tanh) Derivative(x float32) float32 {
	tanhVal := t.Activate(x) // Call updated Activate
	return float32(1.0 - float64(tanhVal)*float64(tanhVal))
}

// Linear activation function (identity function).
// Output = Input. Used here in the output layer before the Softmax calculation (which happens externally).
// Also suitable for regression tasks where the output is not bounded.
type Linear struct{}

func (t Linear) Activate(x float32) float32 {
	return x
}

func (t Linear) Derivative(x float32) float32 {
	return 1
}

// GetActivationFunction returns an ActivationFunction interface based on a string name.
func GetActivationFunction(name string) (ActivationFunction, error) {
	switch name {
	case "relu":
		return ReLU{}, nil
	case "sigmoid":
		return Sigmoid{}, nil
	case "tanh":
		return Tanh{}, nil
	case "leakyrelu":
		return NewLeakyReLU(0.01), nil // Default alpha for LeakyReLU
	case "linear":
		return Linear{}, nil
	default:
		return nil, fmt.Errorf("unknown activation function: %s", name)
	}
}
