package neuralnet

import "math"

type ActivationFunction interface {
	Activate(x float32, useFloat64 bool) float32
	Derivative(x float32, useFloat64 bool) float32
}

type ReLU struct{}

func (r ReLU) Activate(x float32, useFloat64 bool) float32 {
	// math.Max operates on float64, so the core logic doesn't change with useFloat64.
	// The input x is float32, so it's always cast to float64 for math.Max.
	return float32(math.Max(float64(x), 0.0))
}

func (r ReLU) Derivative(x float32, useFloat64 bool) float32 {
	// The logic for ReLU derivative is direct and doesn't involve float64-specific math functions
	// that would change based on useFloat64, other than the signature.
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

func (l LeakyReLU) Activate(x float32, useFloat64 bool) float32 {
	if x > 0 {
		return x
	}
	if useFloat64 {
		return float32(float64(l.alpha) * float64(x))
	}
	return l.alpha * x
}

func (l LeakyReLU) Derivative(x float32, useFloat64 bool) float32 {
	// The logic for LeakyReLU derivative is direct.
	if x > 0 {
		return 1
	}
	return l.alpha
}

type Sigmoid struct{}

func (s Sigmoid) Activate(x float32, useFloat64 bool) float32 {
	if useFloat64 {
		return float32(1.0 / (1.0 + math.Exp(-float64(x))))
	}
	return 1.0 / (1.0 + float32(math.Exp(-float64(x))))
}

func (s Sigmoid) Derivative(x float32, useFloat64 bool) float32 {
	sigmoidVal := s.Activate(x, useFloat64) // Call updated Activate
	if useFloat64 {
		return float32(float64(sigmoidVal) * (1.0 - float64(sigmoidVal)))
	}
	return sigmoidVal * (1.0 - sigmoidVal)
}

type Tanh struct{}

func (t Tanh) Activate(x float32, useFloat64 bool) float32 {
	// math.Tanh operates on float64.
	return float32(math.Tanh(float64(x)))
}

func (t Tanh) Derivative(x float32, useFloat64 bool) float32 {
	tanhVal := t.Activate(x, useFloat64) // Call updated Activate
	if useFloat64 {
		return float32(1.0 - float64(tanhVal)*float64(tanhVal))
	}
	return 1.0 - tanhVal*tanhVal
}

type Linear struct{}

func (t Linear) Activate(x float32, useFloat64 bool) float32 {
	// Linear activation is direct, no change in logic for useFloat64.
	return x
}

func (t Linear) Derivative(x float32, useFloat64 bool) float32 {
	// Linear derivative is constant, no change in logic for useFloat64.
	return 1
}
