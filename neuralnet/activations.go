package neuralnet

import "math"

type ActivationFunction interface {
	Activate(x float32) float32
	Derivative(x float32) float32
}

type ReLU struct{}

func (r ReLU) Activate(x float32) float32 {
	return float32(math.Max(float64(x), 0))
}


func (r ReLU) Derivative(x float32) float32 {
	if x > 0 {
		return 1
	}
	return 0
}

type Sigmoid struct{}

func (s Sigmoid) Activate(x float32) float32 {
	return 1 / (1 + float32(math.Exp(-float64(x))))
}


func (s Sigmoid) Derivative(x float32) float32 {
	sigmoid := s.Activate(x)
	return sigmoid * (1 - sigmoid)
}

type Tanh struct{}

func (t Tanh) Activate(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

func (t Tanh) Derivative(x float32) float32 {
	tanh := t.Activate(x)
	return 1 - tanh*tanh
}