package neuralnet

import "math"

type ActivationFunction interface {
	Activate(x float32) float32
	Derivative(x float32) float32
}

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

type LeakyReLU struct {
	alpha float32
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

type Sigmoid struct{}

func (s Sigmoid) Activate(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

func (s Sigmoid) Derivative(x float32) float32 {
	sigmoidVal := s.Activate(x) // Call updated Activate
	return float32(float64(sigmoidVal) * (1.0 - float64(sigmoidVal)))
}

type Tanh struct{}

func (t Tanh) Activate(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

func (t Tanh) Derivative(x float32) float32 {
	tanhVal := t.Activate(x) // Call updated Activate
	return float32(1.0 - float64(tanhVal)*float64(tanhVal))
}

type Linear struct{}

func (t Linear) Activate(x float32) float32 {
	return x
}

func (t Linear) Derivative(x float32) float32 {
	return 1
}
