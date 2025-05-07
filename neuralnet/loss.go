package neuralnet

import "math"

// LossFunction defines the interface for computing loss and its gradient.
type LossFunction interface {
	// Compute returns the loss value given the model output (softmax probabilities) and one-hot target.
	Compute(output []float32, target []float32) float32
	// Gradient returns the gradient ∂L/∂output for each output neuron.
	Gradient(output []float32, target []float32) []float32
}

// CrossEntropy implements categorical cross-entropy loss.
type CrossEntropy struct{}

// Compute returns the cross-entropy loss.
func (ce *CrossEntropy) Compute(output []float32, target []float32) float32 {
	var loss float32
	for i := range output {
		p := output[i]
		if p < 1e-15 {
			p = 1e-15
		}
		loss -= target[i] * float32(math.Log(float64(p)))
	}
	return loss
}

// Gradient returns the derivative of cross-entropy wrt outputs: (output - target).
func (ce *CrossEntropy) Gradient(output []float32, target []float32) []float32 {
	grad := make([]float32, len(output))
	for i := range output {
		grad[i] = output[i] - target[i]
	}
	return grad
}
