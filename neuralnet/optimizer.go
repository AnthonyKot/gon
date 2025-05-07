package neuralnet

import "errors"

// Optimizer defines interface to apply averaged gradients and adjust learning rate.
type Optimizer interface {
    Apply(nn *NeuralNetwork, batchSize int) error
}

// SGD implements stochastic gradient descent optimizer.
type SGD struct{}

// Apply applies averaged gradients and updates learning rate based on decay.
func (o *SGD) Apply(nn *NeuralNetwork, batchSize int) error {
    if batchSize <= 0 {
        return errors.New("invalid batch size")
    }
    nn.applyAveragedGradients(batchSize, nn.Params.Lr)
    nn.Params.Lr *= nn.Params.Decay
    return nil
}
