package neuralnet

// Package neuralnet provides a simple feedforward neural network implementation.
import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"sync"
	"time"
)

// DefaultMaxAbsValue defines a large finite number to cap extreme values, preventing Inf propagation.
const DefaultMaxAbsValue = float32(1e10)

const MAX_WORKERS int = 12

type Neuron struct {
	weights  []float32
	bias     float32
	output   float32
	momentum []float32
	// Fields for accumulating gradients in batch training
	accumulatedWeightGradients []float32
	accumulatedBiasGradient    float32
}

type Layer struct {
	neurons    []*Neuron
	deltas     []float32
	activation ActivationFunction
}

// Represents the simplest NN.
type NeuralNetwork struct {
	layers                  []*Layer
	input                   []float32
	params                  Params
	prevLayerOutputsBuffer []float32 // Buffer for backpropagation
}

type Params struct {
	lr     float32
	decay  float32
	L2     float32
	// lowCap float32 // Removed: Unused
	MomentumCoefficient float32 // Coefficient for momentum update (e.g., 0.9)
}


// NewParams creates a Params struct with default values for non-specified fields.
func NewParams(learningRate float32, decay float32, regularization float32) Params {
	// Calls NewParamsFull, providing default values for momentum
	defaults := defaultParams()
	return NewParamsFull(learningRate, decay, regularization, defaults.MomentumCoefficient)
}

// NewParamsFull creates a Params struct with all fields specified.
// Note: 'cap' parameter from NewParams is ignored here as lowCap was removed.
func NewParamsFull(learningRate float32, decay float32, regularization float32, momentumCoefficient float32) Params {
	return Params{
		lr:                  learningRate,
		decay:               decay,
		L2:                  regularization,
		// lowCap:              cap, // Removed
		MomentumCoefficient: momentumCoefficient,
	}
}

func defaultParams() *Params {
	return &Params{
		lr:     0.01,
		decay:  0.95, // Reduced decay rate
		L2:     1e-4, // Enabled L2 regularization
		// lowCap: 0, // Removed
		MomentumCoefficient: 0.9, // Default momentum coefficient
	}
}

func DefaultNeuralNetwork(inputSize int, hidden []int, outputSize int) *NeuralNetwork {
	params := defaultParams()
	nn := initialise(inputSize, hidden, outputSize, *params)
	// optimizer and lossFunction fields removed from NeuralNetwork struct
	return nn
}

// initialise creates and initializes the neural network structure, including layers, neurons, weights, and biases.
func initialise(inputSize int, hiddenConfig []int, outputSize int, params Params) *NeuralNetwork {
	rand.Seed(time.Now().UnixNano()) // Seed random number generator for weight initialization

	// Note: To support zero hidden layers (direct input to output), this function
	// would need adjustments, particularly in how prevLayerNeuronCount is initialized
	// for the output layer. For now, we assume hiddenConfig is not empty.
	if len(hiddenConfig) == 0 {
		panic("initialise: hiddenConfig slice cannot be empty for this network structure.")
	}

	numHiddenLayers := len(hiddenConfig)
	// Total processing layers = number of hidden layers + 1 output layer
	nn := &NeuralNetwork{
		layers: make([]*Layer, numHiddenLayers+1),
		params: params,
		input:  make([]float32, inputSize), // Preallocate input slice for FeedForward
	}

	// Determine max size for prevLayerOutputsBuffer.
	// This buffer will hold outputs of layer i-1 when processing layer i.
	// The largest such i-1 layer could be the input layer or any hidden layer.
	maxPrevLayerSize := inputSize
	if numHiddenLayers > 0 {
		for _, size := range hiddenConfig {
			if size > maxPrevLayerSize {
				maxPrevLayerSize = size
			}
		}
	}
	nn.prevLayerOutputsBuffer = make([]float32, maxPrevLayerSize)

	// prevLayerNeuronCount tracks the number of neurons in the layer that feeds into the current one.
	// It starts with inputSize for the first hidden layer.
	prevLayerNeuronCount := inputSize

	// Create Hidden Layers
	for i := 0; i < numHiddenLayers; i++ {
		currentHiddenLayerSize := hiddenConfig[i]
		hiddenLayer := &Layer{
			neurons:    make([]*Neuron, currentHiddenLayerSize),
			activation: ReLU{}, // Default activation for hidden layers
		}
		for j := 0; j < currentHiddenLayerSize; j++ {
			hiddenLayer.neurons[j] = &Neuron{
				weights:  make([]float32, prevLayerNeuronCount),
				// Initialize bias using Xavier/Glorot initialization.
				bias:     xavierInit(prevLayerNeuronCount, currentHiddenLayerSize, nn.params),
				// Initialize momentum buffer (used for SGD with momentum).
				momentum: make([]float32, prevLayerNeuronCount),
			}
			// Initialize weights using Xavier/Glorot initialization.
			// Helps prevent vanishing/exploding gradients during initial training phases.
			for k := range hiddenLayer.neurons[j].weights {
				hiddenLayer.neurons[j].weights[k] = xavierInit(prevLayerNeuronCount, currentHiddenLayerSize, nn.params)
			}
		}
		nn.layers[i] = hiddenLayer
		prevLayerNeuronCount = currentHiddenLayerSize // Update for the next layer's input count
	}

	// Create Output Layer
	// At this point, prevLayerNeuronCount holds the neuron count of the last hidden layer.
	outputLayer := &Layer{
		neurons:    make([]*Neuron, outputSize),
		activation: Linear{}, // Standard for classification before softmax
	}
	for l := 0; l < outputSize; l++ {
		outputLayer.neurons[l] = &Neuron{
			weights: make([]float32, prevLayerNeuronCount),
			// Output layer biases are often initialized to zero, but using Xavier for consistency with current code.
			// Initialize bias using Xavier/Glorot initialization.
			bias:     xavierInit(prevLayerNeuronCount, outputSize, nn.params),
			// Initialize momentum buffer.
			momentum: make([]float32, prevLayerNeuronCount),
		}
		// Initialize weights using Xavier/Glorot initialization.
		for k := range outputLayer.neurons[l].weights {
			outputLayer.neurons[l].weights[k] = xavierInit(prevLayerNeuronCount, outputSize, nn.params)
		}
	}
	// The output layer is the last layer in the nn.layers slice.
	nn.layers[numHiddenLayers] = outputLayer

	return nn
}

func NewNeuralNetwork(inputSize int, hiddenConfig []int, outputSize int, params Params) *NeuralNetwork {
	return initialise(inputSize, hiddenConfig, outputSize, params)
}


// FeedForward propagates the input signal through the network, layer by layer,
// calculating the output of each neuron based on weights, biases, and activation functions.
func (nn *NeuralNetwork) FeedForward(input []float32) {
	// Input is now []float32. Copy it to nn.input.
	// nn.input slice is preallocated in initialise. This loop populates it.
	if len(input) != len(nn.input) {
		// This shouldn't happen if inputSize matches, but as a safeguard:
		panic(fmt.Sprintf("FeedForward: input size %d does not match network input size %d", len(input), len(nn.input)))
	}
	copy(nn.input, input) // Copy input slice to internal buffer

	// The first layer takes inputs as X from nn.input
	for _, neuron := range nn.layers[0].neurons {
		neuron.output = neuron.bias
		for j := 0; j < len(nn.input); j++ { // Iterate over nn.input
			neuron.output += nn.input[j] * neuron.weights[j] // Use nn.input directly
		}
		neuron.output = nn.layers[0].activation.Activate(neuron.output) // Removed UseFloat64 flag
		neuron.output = capValue(neuron.output)
	}
	// nn.input is already set.
	// Process hidden and output layers (i >= 1) using direct loops
	for i := 1; i < len(nn.layers); i++ {
		for _, neuron := range nn.layers[i].neurons {
			// Use float64 for intermediate sum
			var sum64 float64 = float64(neuron.bias)
			for j := 0; j < len(nn.layers[i-1].neurons); j++ {
				sum64 += float64(nn.layers[i-1].neurons[j].output) * float64(neuron.weights[j])
			}
			neuron.output = float32(sum64)
			neuron.output = nn.layers[i].activation.Activate(neuron.output)
			neuron.output = capValue(neuron.output)
		}
	}
}

// applyAveragedGradients updates the network's weights and biases using accumulated gradients.
// It should be called after processing a batch and accumulating gradients.
func (nn *NeuralNetwork) applyAveragedGradients(batchSize int, learningRate float32) {
	if batchSize == 0 {
		// Avoid division by zero if training data is empty
		fmt.Println("applyAveragedGradients: batchSize is zero, skipping updates.")
		return
	}
	fBatchSize := float32(batchSize)

	for _, layer := range nn.layers {
		for _, neuron := range layer.neurons {
			// Update weights
			if neuron.accumulatedWeightGradients != nil { // Check if initialized
				for wIdx := range neuron.weights {
					// Calculate average gradient for this weight over the batch.
					avgGrad64 := float64(neuron.accumulatedWeightGradients[wIdx]) / float64(fBatchSize)
					// Add L2 regularization gradient component: lambda * weight
					// This penalizes large weights to prevent overfitting.
					avgGrad64 += float64(nn.params.L2) * float64(neuron.weights[wIdx])

					// Apply momentum update: new_momentum = momentum_coeff * old_momentum + learning_rate * gradient
					// Momentum helps accelerate convergence and overcome local minima.
					momentum64 := float64(nn.params.MomentumCoefficient)*float64(neuron.momentum[wIdx]) + float64(learningRate)*avgGrad64
					neuron.momentum[wIdx] = float32(momentum64)
					// Update weight: weight = weight - new_momentum
					neuron.weights[wIdx] = float32(float64(neuron.weights[wIdx]) - momentum64)
					neuron.weights[wIdx] = capValue(neuron.weights[wIdx]) // Cap extreme values
				}
			}

			// Update bias (using float64 intermediate)
			// Calculate average gradient for bias over the batch.
			avgBiasGrad64 := float64(neuron.accumulatedBiasGradient) / float64(fBatchSize)
			// Apply simple gradient descent step for bias (momentum could also be added here).
			neuron.bias = float32(float64(neuron.bias) - float64(learningRate)*avgBiasGrad64)
			neuron.bias = capValue(neuron.bias) // Cap extreme values
		}
	}
}

// The old UpdateWeights function is now replaced by applyAveragedGradients
// and the gradient accumulation logic within backpropagateAndAccumulateForSample.

// TrainSGD function removed as unused in the current main application flow.

// Clone creates a deep copy of a neural network. This is crucial for thread-safe parallel processing
// during mini-batch training or parallel accuracy calculation, as each worker goroutine needs
// its own independent copy of the network state to avoid race conditions during FeedForward
// and gradient accumulation.
func (nn *NeuralNetwork) Clone() *NeuralNetwork {
	// Create a new neural network with the same structure
	clone := &NeuralNetwork{
		layers: make([]*Layer, len(nn.layers)),
		params: nn.params, // Corrected: use receiver nn
	}

	// Deep copy all layers
	for i, layer := range nn.layers { // Corrected: use receiver nn
		cloneLayer := &Layer{
			neurons:    make([]*Neuron, len(layer.neurons)),
			activation: layer.activation,
		}

		// Deep copy all neurons
		for j, neuron := range layer.neurons {
			cloneNeuron := &Neuron{
				weights:  make([]float32, len(neuron.weights)),
				bias:     neuron.bias,
				output:   neuron.output,
				momentum: make([]float32, len(neuron.momentum)), // Initialize momentum slice
			}

			// Copy weights
			copy(cloneNeuron.weights, neuron.weights)
			// Copy momentum
			if neuron.momentum != nil { // Guard against nil if original momentum could be nil (though init suggests it won't be)
				copy(cloneNeuron.momentum, neuron.momentum)
			}
			cloneLayer.neurons[j] = cloneNeuron
		}

		clone.layers[i] = cloneLayer
	}

	// Copy input if it exists
	if nn.input != nil {
		clone.input = make([]float32, len(nn.input))
		copy(clone.input, nn.input)
	}
	if nn.prevLayerOutputsBuffer != nil {
		clone.prevLayerOutputsBuffer = make([]float32, len(nn.prevLayerOutputsBuffer))
		copy(clone.prevLayerOutputsBuffer, nn.prevLayerOutputsBuffer)
	}

	return clone
}

func (nn *NeuralNetwork) zeroAccumulatedGradients() {
	for _, layer := range nn.layers {
		for _, neuron := range layer.neurons {
			// Ensure accumulatedWeightGradients slice is allocated and zeroed
			if neuron.accumulatedWeightGradients == nil || len(neuron.accumulatedWeightGradients) != len(neuron.weights) {
				neuron.accumulatedWeightGradients = make([]float32, len(neuron.weights))
			} else {
				for k := range neuron.accumulatedWeightGradients {
					neuron.accumulatedWeightGradients[k] = 0.0
				}
			}
			// Zero out accumulatedBiasGradient
			neuron.accumulatedBiasGradient = 0.0
		}
	}
}


/*
TrainMiniBatch processes the training data in mini-batches.
- trainingData: The input samples.
- expectedOutputs: The corresponding target labels.
- batchSize: The number of samples in each mini-batch.
- epochs: The total number of times to iterate over the entire dataset.
- numWorkers: The number of goroutines to use for processing samples within a mini-batch. If <= 1, runs single-threaded.
*/
func (nn *NeuralNetwork) TrainMiniBatch(trainingData [][]float32, expectedOutputs [][]float32, batchSize int, epochs int, numWorkers int) {
	if numWorkers > MAX_WORKERS {
		numWorkers = MAX_WORKERS
	}
	numSamples := len(trainingData)
	if numSamples == 0 {
		fmt.Println("TrainMiniBatch: No training data provided.")
		return
	}
	if batchSize <= 0 || batchSize > numSamples {
		fmt.Printf("TrainMiniBatch: Invalid batchSize %d. Setting to numSamples %d.\n", batchSize, numSamples)
		batchSize = numSamples // Fallback to full batch if batchSize is invalid
	}

	for e := 0; e < epochs; e++ {
		var totalEpochLoss float32 = 0.0
		var samplesProcessedInEpoch int = 0

		// Shuffle data at the beginning of each epoch
		permutation := rand.Perm(numSamples)
		shuffledTrainingData := make([][]float32, numSamples)
		shuffledExpectedOutputs := make([][]float32, numSamples)
		for i := 0; i < numSamples; i++ {
			shuffledTrainingData[i] = trainingData[permutation[i]]
			shuffledExpectedOutputs[i] = expectedOutputs[permutation[i]]
		}

		// Iterate over mini-batches
		for i := 0; i < numSamples; i += batchSize {
			end := i + batchSize
			if end > numSamples {
				end = numSamples // Adjust for the last batch if it's smaller
			}

			currentMiniBatchSize := end - i
			if currentMiniBatchSize == 0 {
				continue // Should not happen if numSamples > 0
			}

			nn.zeroAccumulatedGradients() // Zero out accumulators for the new mini-batch
			var miniBatchLoss float32 = 0.0

			if numWorkers <= 1 {
				// Single-threaded processing for the mini-batch
				for j := i; j < end; j++ {
					dataSample := shuffledTrainingData[j]
					labelSample := shuffledExpectedOutputs[j]
					sampleLoss := nn.backpropagateAndAccumulateForSample(dataSample, labelSample)
					miniBatchLoss += sampleLoss
				}
			} else {
				// Multi-threaded processing for the mini-batch
				var wg sync.WaitGroup
				workerLosses := make([]float32, numWorkers)
				workerClones := make([]*NeuralNetwork, numWorkers)
				samplesPerWorker := (currentMiniBatchSize + numWorkers - 1) / numWorkers // Ceiling division

				for w := 0; w < numWorkers; w++ {
					wg.Add(1)
					workerStart := i + w*samplesPerWorker
					workerEnd := workerStart + samplesPerWorker
					if workerStart >= end { // No samples for this worker
						wg.Done()
						continue
					}
					if workerEnd > end {
						workerEnd = end
					}

					go func(workerID int, startIdx int, endIdx int) {
						defer wg.Done()
						if startIdx >= endIdx { // Double check, no samples for this worker
							return
						}

						clone := nn.Clone()              // Each worker gets a clone
						clone.zeroAccumulatedGradients() // Initialize clone's accumulators
						var currentWorkerLoss float32 = 0.0

						for k := startIdx; k < endIdx; k++ {
							dataSample := shuffledTrainingData[k]
							labelSample := shuffledExpectedOutputs[k]
							sampleLoss := clone.backpropagateAndAccumulateForSample(dataSample, labelSample)
							currentWorkerLoss += sampleLoss
						}
						workerLosses[workerID] = currentWorkerLoss
						workerClones[workerID] = clone
					}(w, workerStart, workerEnd)
				}
				wg.Wait()

				// Aggregate losses from workers
				for _, l := range workerLosses {
					miniBatchLoss += l
				}

				// Aggregate gradients from worker clones into the main network's accumulators
				// The main network's accumulators were zeroed by nn.zeroAccumulatedGradients()
				for _, clone := range workerClones {
					if clone == nil { // Can happen if a worker had no samples
						continue
					}
					for layerIdx, cloneLayer := range clone.layers {
						for neuronIdx, cloneNeuron := range cloneLayer.neurons {
							mainNeuron := nn.layers[layerIdx].neurons[neuronIdx]
							if cloneNeuron.accumulatedWeightGradients != nil {
								for wIdx, grad := range cloneNeuron.accumulatedWeightGradients {
									mainNeuron.accumulatedWeightGradients[wIdx] += grad
								}
							}
							mainNeuron.accumulatedBiasGradient += cloneNeuron.accumulatedBiasGradient
						}
					}
				}
			}

			// After processing all samples in the mini-batch (either single or multi-threaded), apply the averaged gradients
			nn.applyAveragedGradients(currentMiniBatchSize, nn.params.lr)

			totalEpochLoss += miniBatchLoss
			samplesProcessedInEpoch += currentMiniBatchSize
		}

		averageEpochLoss := float32(0.0)
		if samplesProcessedInEpoch > 0 {
			averageEpochLoss = totalEpochLoss / float32(samplesProcessedInEpoch)
		}

		fmt.Printf("Loss MiniBatch Epoch %d = %.2f (LR: %.5f)\n", e, averageEpochLoss, nn.params.lr)

		// Apply learning rate decay at the end of each epoch
		nn.params.lr *= nn.params.decay
	}
}

// backpropagateAndAccumulateForSample performs feedforward, calculates loss,
// computes sample-specific deltas (errors) and accumulates gradients for a single sample.
// It returns the loss calculated for this sample.
func (nn *NeuralNetwork) backpropagateAndAccumulateForSample(dataSample []float32, labelSample []float32) float32 {
	// 1. FeedForward: Calculate neuron outputs for the current sample.
	nn.FeedForward(dataSample)

	// 2. Calculate Output Layer Error (Delta):
	// For cross-entropy loss combined with a softmax output layer, the error signal (delta)
	// for each output neuron simplifies beautifully to (softmax_probability - target_probability).
	props := nn.calculateProps() // Get softmax probabilities (float64)
	if len(props) != len(labelSample) {
		panic("backpropagateAndAccumulateForSample: props and labelSample length mismatch")
	}
	errVecData := make([]float64, len(props)) // Use []float64 directly
	for i := range props {
		errVecData[i] = props[i] - float64(labelSample[i]) // Manual subtraction
	}

	loss := nn.calculateLoss(labelSample) // Calculate loss for this sample

	// 3. Backpropagate Deltas: Calculate error signals (deltas) for each layer, starting from the output.
	//    The delta for a neuron represents how much its pre-activation input needs to change
	//    to reduce the overall loss.

	// Calculate deltas for the output layer (using the simplified gradient)
	outputLayer := nn.layers[len(nn.layers)-1]
	if outputLayer.deltas == nil || len(outputLayer.deltas) != len(outputLayer.neurons) {
		outputLayer.deltas = make([]float32, len(outputLayer.neurons))
	}
	// Removed incorrect line: errVecData := errVec.RawVector().Data
	// Use the errVecData slice calculated earlier in this function.
	for j := 0; j < len(outputLayer.neurons); j++ {
		// For a single sample, the delta is the error component (softmax_output - target_j).
		outputLayer.deltas[j] = capValue(float32(errVecData[j])) // Store the calculated delta
	}

	// Propagate deltas backward through hidden layers:
	// The delta for a neuron in a hidden layer is calculated based on the weighted sum
	// of the deltas from the layer *ahead* of it, multiplied by the derivative of its own activation function.
	// Formula: delta_j = (Sum_k(delta_k * weight_kj)) * activation_derivative(output_j)
	// where j is current layer neuron index, k is next layer neuron index.
	for i := len(nn.layers) - 2; i >= 0; i-- { // Iterate backwards from last hidden layer
		layer := nn.layers[i]
		nextLayer := nn.layers[i+1]

		if layer.deltas == nil || len(layer.deltas) != len(layer.neurons) {
			layer.deltas = make([]float32, len(layer.neurons))
		}

		for j, neuron := range layer.neurons { // For each neuron 'j' in current layer 'i'
			// Removed redundant float32 errorSumTimesWeight calculation loop.
			// Calculate error sum using float64 directly.
			var errorSumTimesWeight64 float64 = 0.0
			// Sum (delta_k * weight_kj) from the next layer
			for k, nextNeuron := range nextLayer.neurons { // k = index in next layer
				// nextNeuron.weights[j] is the weight connecting neuron j (current layer) to neuron k (next layer)
				errorSumTimesWeight64 += float64(nextNeuron.weights[j]) * float64(nextLayer.deltas[k])
			}
			// Multiply by the derivative of the activation function for the current neuron j
			derivative := layer.activation.Derivative(neuron.output)
			layer.deltas[j] = capValue(float32(errorSumTimesWeight64*float64(derivative))) // Store the calculated delta
		}
	}

	// 4. Accumulate Gradients: Calculate the gradient contribution of this sample for each weight and bias,
	//    and add it to the running total for the current mini-batch.
	//    Gradient for weight w_ij = delta_j * output_i (where i is prev layer neuron, j is current layer neuron)
	//    Gradient for bias b_j = delta_j
	for layerIndex, layer := range nn.layers {
		var prevLayerOutputs []float32 // Holds outputs from the layer feeding *into* the current layer
		if layerIndex == 0 {
			// First layer's inputs are the network inputs
			prevLayerOutputs = nn.input // Activations from input layer (i.e., the input sample itself)
		} else {
			prevLayer := nn.layers[layerIndex-1]
			currentPrevLayerNumNeurons := len(prevLayer.neurons)
			// Ensure buffer is large enough (should be by design from initialise)
			if cap(nn.prevLayerOutputsBuffer) < currentPrevLayerNumNeurons {
				// This is a fallback, ideally initialise sizes it correctly.
				nn.prevLayerOutputsBuffer = make([]float32, currentPrevLayerNumNeurons)
			}
			prevLayerOutputs = nn.prevLayerOutputsBuffer[:currentPrevLayerNumNeurons] // Re-slice the buffer
			for pIdx, pNeuron := range prevLayer.neurons {
				prevLayerOutputs[pIdx] = pNeuron.output // Activations from the previous layer
			}
		}

		for nIdx, neuron := range layer.neurons { // For neuron 'nIdx' in current 'layer'
			sampleDelta := layer.deltas[nIdx] // Delta for this neuron, for this sample

			// Accumulate weight gradients: gradient_w = delta_current_neuron * output_prev_layer_neuron
			if neuron.accumulatedWeightGradients != nil { // Should be initialized by zeroAccumulatedGradients
				for wIdx := range neuron.weights {
					// Use float64 for intermediate calculation
					gradContrib64 := float64(sampleDelta) * float64(prevLayerOutputs[wIdx])
					neuron.accumulatedWeightGradients[wIdx] += float32(gradContrib64)
				}
			}
			// Accumulate bias gradient: gradient_b = delta_current_neuron
			neuron.accumulatedBiasGradient += sampleDelta
		}
	}
	return loss
}

// TrainBatch function removed as unused (superseded by TrainMiniBatch).

func (nn *NeuralNetwork) Output() []float32 {
	outputLayer := nn.layers[len(nn.layers)-1].neurons
	output := make([]float32, len(outputLayer))
	for i, neuron := range outputLayer {
		output[i] = neuron.output
	}
	return output
}

func (nn *NeuralNetwork) Predict(data []float32) int {
	nn.FeedForward(data)
	propsData := nn.calculateProps() // returns []float64
	if len(propsData) == 0 {
		panic("Predict: calculateProps returned empty slice") // Or handle error appropriately
	}
	maxVal := propsData[0]
	idx := 0
	for i := 1; i < len(propsData); i++ {
		val := propsData[i]
		if val > maxVal {
			maxVal = val
			idx = i
		}
	}
	return idx
}

func (nn *NeuralNetwork) calculateProps() []float64 { // Return []float64
	outputLayer := nn.layers[len(nn.layers)-1].neurons
	output := make([]float32, len(outputLayer))
	for i, neuron := range outputLayer {
		neuron.output = capValue(neuron.output)
		output[i] = neuron.output
	}
	softmax := softmax(output, nn.params)
	softmaxFloat64 := make([]float64, len(softmax))
	for i, v := range softmax {
		v = capValue(v)
		softmaxFloat64[i] = float64(v)
	}
	return softmaxFloat64 // Return the slice directly
}

// calculateLoss computes the cross-entropy loss for a single sample, including L2 regularization.
// Cross-Entropy Loss: - Sum_i (target_i * log(predicted_probability_i))
// L2 Regularization Term: (lambda / 2) * Sum_all_weights (weight^2)
func (nn *NeuralNetwork) calculateLoss(target []float32) float32 { // Target is now []float32
	propsData := nn.calculateProps() // Get predicted probabilities (float64)
	var loss float32 = 0.0

	// Calculate Cross-Entropy part
	// propsData is already []float64
	// targetData is now the input []float32

	// Use float64 for intermediate loss calculation
	var loss64 float64
	if len(propsData) != len(target) {
		panic("calculateLoss: propsData and target length mismatch")
	}
	for i := 0; i < len(propsData); i++ {
		p64 := math.Max(propsData[i], 1e-15)
		loss64 -= float64(target[i]) * math.Log(p64)
	}
	// Calculate L2 regularization part
	var reg64 float64 = 0.0
	for _, layer := range nn.layers {
		for _, neuron := range layer.neurons {
			for _, w := range neuron.weights {
				reg64 += float64(w) * float64(w)
			}
		}
	}
	loss64 += 0.5 * float64(nn.params.L2) * reg64
	loss = float32(loss64) // Cast final result back to float32
	// Guard against NaN or Inf
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		return 0.0
	}
	return loss
}


// stableSoftmax computes the softmax function in a numerically stable way.
// Softmax converts a vector of scores (logits) into a probability distribution.
// Formula: probability_i = exp(score_i) / Sum_j(exp(score_j))
// Numerical Stability: Subtracting the maximum score from all scores before exponentiation
// prevents large intermediate values that could cause overflow (exp(large_number) -> Inf)
// or underflow (exp(very_negative_number) -> 0), without changing the final probabilities.
func stableSoftmax(output []float32) []float32 {
	if len(output) == 0 {
		return []float32{}
	}

	exps := make([]float32, len(output))
	// Use float64 for intermediate calculations for stability
	var maxVal64 float64 = float64(output[0])
	for _, v := range output[1:] {
		if float64(v) > maxVal64 {
			maxVal64 = float64(v)
		}
	}

	var sumExps64 float64 = 0.0
	tempExps64 := make([]float64, len(output)) // Temporary for float64 exponentiated values
	for i, v := range output {
		expVal64 := math.Exp(float64(v) - maxVal64)
		tempExps64[i] = expVal64
		sumExps64 += expVal64
	}

	// or if sumExps becomes Inf (less likely with max subtraction but theoretically possible with extreme inputs)
	if sumExps64 == 0 || math.IsInf(sumExps64, 0) {
		// Fallback: distribute probability uniformly.
		// This prevents division by zero or NaN results (e.g. 0/0 or Inf/Inf).
		// This situation indicates that the network outputs are either all extremely small or problematic.
		uniformProb := 1.0 / float32(len(exps))
		for i := range exps {
			exps[i] = uniformProb
		}
		return exps
	}

	for i := range tempExps64 {
		exps[i] = float32(tempExps64[i] / sumExps64) // Cast back to float32
	}
	return exps
}

func softmax(output []float32, params Params) []float32 {
	// Use the numerically stable softmax calculation.
	stableOutput := stableSoftmax(output)

	// Apply capValue to the results. With default params (lowCap=0),
	// this primarily handles any residual NaNs by converting them to 0.
	// If lowCap were non-zero, it would enforce min/max probability magnitudes.
	for i, v := range stableOutput {
		stableOutput[i] = capValue(v)
	}
	return stableOutput
}

func xavierInit(numInputs int, numOutputs int, params Params) float32 {
	limit := math.Sqrt(6.0 / float64(numInputs+numOutputs))
	xavier := float32(2*rand.Float64()*limit - limit)
	return capValue(xavier)
}

// capValue ensures numerical stability by handling NaN (Not a Number) and Inf (Infinity) values.
// It replaces NaN with 0 and caps Inf values at a large predefined constant.
func capValue(value float32) float32 {
	if math.IsNaN(float64(value)) {
		return 0.0 // Replace NaN with 0
	}
	if math.IsInf(float64(value), 1) { // +Inf
		return DefaultMaxAbsValue
	}
	if math.IsInf(float64(value), -1) {
		return -DefaultMaxAbsValue
	}
	// Note: The logic for handling non-zero lowCap (enforcing min/max magnitudes)
	// was removed as lowCap is currently unused and always defaults to 0.
	return value
}


func (nn *NeuralNetwork) Save(filename string) {
	file, err := os.Create(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	err = encoder.Encode(nn)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Model saved as %s\n", filename) // Use fmt.Printf for consistency
}
func LoadModel(filename string) *NeuralNetwork {
	file, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	var nn *NeuralNetwork
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&nn)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Model loaded from %s\n", filename) // Use fmt.Printf for consistency
	return nn
}
func (l *Layer) String() string {
	var sb strings.Builder

	// Print each neuron
	for i, neuron := range l.neurons {
		sb.WriteString(fmt.Sprintf("Neuron %d: output=%.2f\n", i, neuron.output))
	}

	// Print deltas
	sb.WriteString(fmt.Sprintf("Deltas: %v\n", l.deltas))

	return sb.String()
}

func (nn *NeuralNetwork) String() string {
	var sb strings.Builder

	// Iterate through the layers and print them
	for i, layer := range nn.layers {
		sb.WriteString(fmt.Sprintf("Layer %d:\n%s\n", i, layer.String()))
	}

	return sb.String()
}
