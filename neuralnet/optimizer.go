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

		// If Batch Normalization is used for this layer, update Gamma and Beta
		if layer.UseBatchNormalization {
			if layer.AccumulatedGammaGradients == nil || layer.AccumulatedBetaGradients == nil {
				// Or handle this potential error more gracefully, though they should be initialized.
				// For now, skip if they are not initialized to prevent panic.
				// Consider logging a warning or returning an error if this state is unexpected.
				continue
			}
			if len(layer.AccumulatedGammaGradients) != len(layer.Neurons) || len(layer.AccumulatedBetaGradients) != len(layer.Neurons) {
				// This would indicate a mismatch in initialization or update logic.
				// Skip, log, or error. For now, skip.
				continue
			}

			for j := range layer.Neurons { // Iterate using index `j` for Gamma and Beta which are per-neuron
				// Update Gamma
				if j < len(layer.Gamma) && j < len(layer.AccumulatedGammaGradients) { // Boundary check
					gammaGradient := layer.AccumulatedGammaGradients[j] / float32(batchSize)
					layer.Gamma[j] -= params.Lr * gammaGradient
					// Optional: Add momentum for Gamma here if desired in the future
					// (would require layer.GammaVelocities, similar to neuron.WeightVelocities)
				}

				// Update Beta
				if j < len(layer.Beta) && j < len(layer.AccumulatedBetaGradients) { // Boundary check
					betaGradient := layer.AccumulatedBetaGradients[j] / float32(batchSize)
					layer.Beta[j] -= params.Lr * betaGradient
					// Optional: Add momentum for Beta here if desired in the future
					// (would require layer.BetaVelocities)
				}
			}
		}
	}
	return nil
}
