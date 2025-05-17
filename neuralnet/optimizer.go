package neuralnet

import "errors"

// Gradients holds accumulated gradients for all layers and neurons.
type Gradients struct {
	WeightGradients [][][]float32
	BiasGradients   [][]float32
}

// Optimizer defines interface to apply averaged gradients and adjust learning rate.
type Optimizer interface {
	Apply(params *Params, layers []*Layer, accumulatedGradients *Gradients, batchSize int) error
}

// SGD implements stochastic gradient descent optimizer.
type SGD struct{}

// Apply applies averaged gradients and updates learning rate based on decay.
func (o *SGD) Apply(params *Params, layers []*Layer, accumulatedGradients *Gradients, batchSize int) error {
	if batchSize <= 0 {
		return errors.New("invalid batch size")
	}

	for i, layer := range layers {
		for j, neuron := range layer.Neurons {
			// Update weights
			for k := range neuron.Weights {
				gradient := accumulatedGradients.WeightGradients[i][j][k] / float32(batchSize)
				neuron.Weights[k] -= params.Lr * (gradient + params.L2*neuron.Weights[k])
				// Apply momentum
				if params.MomentumCoefficient > 0 {
					velocity := params.MomentumCoefficient*neuron.WeightVelocities[k] - params.Lr*gradient
					neuron.Weights[k] += velocity
					neuron.WeightVelocities[k] = velocity
				}
			}

			// Update bias
			gradient := accumulatedGradients.BiasGradients[i][j] / float32(batchSize)
			neuron.Bias -= params.Lr * (gradient + params.L2*neuron.Bias)
			// Apply momentum
			if params.MomentumCoefficient > 0 {
				velocity := params.MomentumCoefficient*neuron.BiasVelocity - params.Lr*gradient
				neuron.Bias += velocity
				neuron.BiasVelocity = velocity
			}
		}
	}
	return nil
}
