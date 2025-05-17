package neuralnet

import "testing"

func TestSGDApplyInvalidBatchSize(t *testing.T) {
	sgd := &SGD{}
	// Create a minimal set of parameters and layers for the test, as Apply doesn't use all nn parts.
	params := &Params{}
	layers := []*Layer{} // No layers, or minimal layers, Apply should check batchSize first.
	gradients := &Gradients{} // Dummy gradients
	if err := sgd.Apply(params, layers, gradients, 0); err == nil {
		t.Error("SGD.Apply with batchSize=0 did not return error")
	}
	if err := sgd.Apply(params, layers, gradients, -1); err == nil {
		t.Error("SGD.Apply with batchSize=-1 did not return error")
	}
}

func TestSGDApplyValidBatchSize(t *testing.T) {
	const epsilon = 1e-6 // Tolerance for float comparisons

	// 1. Network Setup
	// Create a neural network with specified parameters and default activations (ReLU for hidden, Linear for output)
	// NewParamsFull: learningRate, decay, regularization (L2), momentumCoefficient, dropoutRate, enableBatchNorm
	nn := NewNeuralNetwork(
		1,                    // input size
		[]int{1},             // hidden layer config
		1,                    // output size
		NewParamsFull(
			0.1,  // learningRate
			0,    // decay
			0.01, // regularization (L2)
			0.9,  // momentumCoefficient
			0.0,  // dropoutRate
			false,// enableBatchNorm
		),
		ReLU{}, Linear{}, // hidden and output activations
	)
	sgd := &SGD{}
	batchSize := 1

	// Initial values for weights, biases, and velocities
	initialWeight := float32(1.0)
	initialBias := float32(0.5)
	initialWeightVelocity := float32(0.1)
	initialBiasVelocity := float32(0.05)

	// Gradient values
	gradWeight := float32(0.2)
	gradBias := float32(0.1)

	// 2. Neuron Initialization & 3. Gradients Setup
	gradients := Gradients{
		WeightGradients: make([][][]float32, len(nn.Layers)),
		BiasGradients:   make([][]float32, len(nn.Layers)),
	}

	for i, layer := range nn.Layers {
		gradients.WeightGradients[i] = make([][]float32, len(layer.Neurons))
		gradients.BiasGradients[i] = make([]float32, len(layer.Neurons))
		for j, neuron := range layer.Neurons {
			neuron.Weights = make([]float32, len(neuron.Weights))
			neuron.WeightVelocities = make([]float32, len(neuron.Weights))
			gradients.WeightGradients[i][j] = make([]float32, len(neuron.Weights))

			for k := range neuron.Weights {
				neuron.Weights[k] = initialWeight
				neuron.WeightVelocities[k] = initialWeightVelocity
				gradients.WeightGradients[i][j][k] = gradWeight
			}
			neuron.Bias = initialBias
			neuron.BiasVelocity = initialBiasVelocity
			gradients.BiasGradients[i][j] = gradBias
		}
	}
	
	// 4. Store Initial Values (already done by using initialWeight, initialBias etc.)

	// 5. Call sgd.Apply
	err := sgd.Apply(&nn.Params, nn.Layers, &gradients, batchSize)
	if err != nil {
		t.Fatalf("SGD.Apply returned an unexpected error: %v", err)
	}

	// 6. Calculate Expected Values & 7. Verify Updates
	for i, layer := range nn.Layers {
		for j, neuron := range layer.Neurons {
			// Verify Weights and Weight Velocities
			for k := range neuron.Weights {
				avgGradientW := gradWeight / float32(batchSize)
				
				// Weight update part 1 (gradient and L2)
				weightAfterGradL2 := initialWeight - nn.Params.Lr*(avgGradientW+nn.Params.L2*initialWeight)
				
				// Velocity calculation (using original gradient, not L2 adjusted)
				expectedNewVelocityW := nn.Params.MomentumCoefficient*initialWeightVelocity - nn.Params.Lr*avgGradientW
				
				// Weight update part 2 (adding velocity)
				expectedWeight := weightAfterGradL2 + expectedNewVelocityW

				if diff := expectedWeight - neuron.Weights[k]; diff < -epsilon || diff > epsilon {
					t.Errorf("Layer %d, Neuron %d, Weight %d: Expected weight %f, got %f. Diff: %f", i, j, k, expectedWeight, neuron.Weights[k], diff)
				}
				if diff := expectedNewVelocityW - neuron.WeightVelocities[k]; diff < -epsilon || diff > epsilon {
					t.Errorf("Layer %d, Neuron %d, Weight %d: Expected weight velocity %f, got %f. Diff: %f", i, j, k, expectedNewVelocityW, neuron.WeightVelocities[k], diff)
				}
			}

			// Verify Bias and Bias Velocity
			avgGradientB := gradBias / float32(batchSize)

			// Bias update part 1 (gradient and L2)
			biasAfterGradL2 := initialBias - nn.Params.Lr*(avgGradientB+nn.Params.L2*initialBias)

			// Velocity calculation (using original gradient)
			expectedNewVelocityB := nn.Params.MomentumCoefficient*initialBiasVelocity - nn.Params.Lr*avgGradientB
			
			// Bias update part 2 (adding velocity)
			expectedBias := biasAfterGradL2 + expectedNewVelocityB
			
			if diff := expectedBias - neuron.Bias; diff < -epsilon || diff > epsilon {
				t.Errorf("Layer %d, Neuron %d: Expected bias %f, got %f. Diff: %f", i, j, expectedBias, neuron.Bias, diff)
			}
			if diff := expectedNewVelocityB - neuron.BiasVelocity; diff < -epsilon || diff > epsilon {
				t.Errorf("Layer %d, Neuron %d: Expected bias velocity %f, got %f. Diff: %f", i, j, expectedNewVelocityB, neuron.BiasVelocity, diff)
			}
		}
	}
}
