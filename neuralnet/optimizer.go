package neuralnet

import (
	"errors"
	"math" // Ensure math package is imported
)

// Gradients holds accumulated gradients for all layers and neurons.
type Gradients struct {
	WeightGradients [][][]float32
	BiasGradients   [][]float32
}

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

// Adam implements the Adam optimizer.
type Adam struct {
	t int // Timestep, incremented with each call to Apply
}

// Apply applies averaged gradients using Adam algorithm and updates learning rate based on decay.
func (o *Adam) Apply(params *Params, layers []*Layer, accumulatedGradients *Gradients, batchSize int) error {
	if batchSize <= 0 {
		return errors.New("optimizer: invalid batch size for Adam")
	}

	o.t++ // Increment timestep

	lr := params.Lr // Current learning rate
	beta1 := params.Beta1
	beta2 := params.Beta2
	epsilon := params.EpsilonAdam
	l2 := params.L2

	for i, layer := range layers {
		for j, neuron := range layer.Neurons {
			// Ensure neuron's Adam moment estimates are initialized (should be done in Neuron creation)
			if neuron.WeightM == nil || neuron.WeightV == nil || len(neuron.WeightM) != len(neuron.Weights) || len(neuron.WeightV) != len(neuron.Weights) {
				// This indicates an issue with Neuron initialization if they are nil or mismatched.
				// For safety, skip this neuron or return an error.
				// fmt.Printf("Warning: Adam moment estimates not initialized or mismatched for neuron %d in layer %d. Skipping.\n", j, i)
				continue
			}

			// Update weights
			for k := range neuron.Weights {
				grad_wk := accumulatedGradients.WeightGradients[i][j][k] / float32(batchSize)

				// Adam update for weight k
				neuron.WeightM[k] = beta1*neuron.WeightM[k] + (1-beta1)*grad_wk
				neuron.WeightV[k] = beta2*neuron.WeightV[k] + (1-beta2)*(grad_wk*grad_wk)

				mHat_wk := neuron.WeightM[k] / (1 - float32(math.Pow(float64(beta1), float64(o.t))))
				vHat_wk := neuron.WeightV[k] / (1 - float32(math.Pow(float64(beta2), float64(o.t))))

				effective_grad := mHat_wk / (float32(math.Sqrt(float64(vHat_wk))) + epsilon)
				neuron.Weights[k] -= lr * (effective_grad + l2*neuron.Weights[k])
			}

			// Update bias
			grad_b := accumulatedGradients.BiasGradients[i][j] / float32(batchSize)

			// Adam update for bias
			neuron.BiasM = beta1*neuron.BiasM + (1-beta1)*grad_b
			neuron.BiasV = beta2*neuron.BiasV + (1-beta2)*(grad_b*grad_b)

			mHat_b := neuron.BiasM / (1 - float32(math.Pow(float64(beta1), float64(o.t))))
			vHat_b := neuron.BiasV / (1 - float32(math.Pow(float64(beta2), float64(o.t))))

			effective_grad_bias := mHat_b / (float32(math.Sqrt(float64(vHat_b))) + epsilon)
			neuron.Bias -= lr * (effective_grad_bias + l2*neuron.Bias)
		}

		// If Batch Normalization is used for this layer, update Gamma and Beta using Adam
		if layer.UseBatchNormalization {
			// Ensure layer's Adam moment estimates for Gamma/Beta are initialized and correctly sized
			if layer.GammaM == nil || layer.GammaV == nil || layer.BetaM == nil || layer.BetaV == nil ||
				len(layer.GammaM) != len(layer.Neurons) || len(layer.GammaV) != len(layer.Neurons) ||
				len(layer.BetaM) != len(layer.Neurons) || len(layer.BetaV) != len(layer.Neurons) {
				// fmt.Printf("Warning: Adam moment estimates for Gamma/Beta not initialized or mismatched length for layer %d. Skipping Gamma/Beta update.\n", i)
				continue
			}
			// Also check accumulated gradients for Gamma/Beta (though these are inputs to this function)
			if layer.AccumulatedGammaGradients == nil || layer.AccumulatedBetaGradients == nil ||
				len(layer.AccumulatedGammaGradients) != len(layer.Neurons) || len(layer.AccumulatedBetaGradients) != len(layer.Neurons) {
				// fmt.Printf("Warning: Accumulated gradients for Gamma/Beta are nil or have mismatched length for layer %d. Skipping Gamma/Beta update.\n", i)
				continue
			}

			for j := range layer.Neurons { // j is neuron index
				// Update Gamma[j]
				grad_gamma_j := layer.AccumulatedGammaGradients[j] / float32(batchSize)
				layer.GammaM[j] = beta1*layer.GammaM[j] + (1-beta1)*grad_gamma_j
				layer.GammaV[j] = beta2*layer.GammaV[j] + (1-beta2)*(grad_gamma_j*grad_gamma_j)

				mHat_gamma_j := layer.GammaM[j] / (1 - float32(math.Pow(float64(beta1), float64(o.t))))
				vHat_gamma_j := layer.GammaV[j] / (1 - float32(math.Pow(float64(beta2), float64(o.t))))

				// L2 regularization is typically not applied to Gamma and Beta.
				layer.Gamma[j] -= lr * (mHat_gamma_j / (float32(math.Sqrt(float64(vHat_gamma_j))) + epsilon))

				// Update Beta[j]
				grad_beta_j := layer.AccumulatedBetaGradients[j] / float32(batchSize)
				layer.BetaM[j] = beta1*layer.BetaM[j] + (1-beta1)*grad_beta_j
				layer.BetaV[j] = beta2*layer.BetaV[j] + (1-beta2)*(grad_beta_j*grad_beta_j)

				mHat_beta_j := layer.BetaM[j] / (1 - float32(math.Pow(float64(beta1), float64(o.t))))
				vHat_beta_j := layer.BetaV[j] / (1 - float32(math.Pow(float64(beta2), float64(o.t))))

				// L2 regularization is typically not applied to Gamma and Beta.
				layer.Beta[j] -= lr * (mHat_beta_j / (float32(math.Sqrt(float64(vHat_beta_j))) + epsilon))
			}
		}
	}
	return nil
}
