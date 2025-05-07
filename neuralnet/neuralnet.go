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

func init() {
	rand.Seed(time.Now().UnixNano())
}

// DefaultMaxAbsValue defines a large finite number to cap extreme values, preventing Inf propagation.
const DefaultMaxAbsValue = float32(1e10)

const MAX_WORKERS int = 12

type Neuron struct {
	Weights  []float32
	Bias     float32
	Output   float32
	Momentum []float32
	// Fields for accumulating gradients in batch training
	AccumulatedWeightGradients []float32
	AccumulatedBiasGradient    float32
}

type Layer struct {
	Neurons    []*Neuron
	Deltas     []float32
	Activation ActivationFunction
}

// Represents the simplest NN.
type NeuralNetwork struct {
	Layers                  []*Layer
	Input                   []float32
	Params                  Params
	PrevLayerOutputsBuffer []float32 // Buffer for backpropagation
}

type Params struct {
	Lr     float32
	Decay  float32
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
func NewParamsFull(learningRate float32, decay float32, regularization float32, momentumCoefficient float32) Params {
	return Params{
		Lr:                  learningRate, // Exported field name
		Decay:               decay,        // Exported field name
		L2:                  regularization, // Exported field name
		MomentumCoefficient: momentumCoefficient,
	}
}

func defaultParams() *Params {
	return &Params{
		Lr:     0.01, // Exported field name
		Decay:  0.95, // Exported field name
		L2:     1e-4, // Exported field name
		MomentumCoefficient: 0.9,
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

	// Note: To support zero hidden layers (direct input to output), this function
	// would need adjustments, particularly in how prevLayerNeuronCount is initialized
	// for the output layer. For now, we assume hiddenConfig is not empty.
	if len(hiddenConfig) == 0 {
		panic("initialise: hiddenConfig slice cannot be empty for this network structure.")
	}

	numHiddenLayers := len(hiddenConfig)
	// Total processing layers = number of hidden layers + 1 output layer
	nn := &NeuralNetwork{
		Layers: make([]*Layer, numHiddenLayers+1), // Exported field name
		Params: params,                            // Exported field name
		Input:  make([]float32, inputSize),        // Exported field name
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
	nn.PrevLayerOutputsBuffer = make([]float32, maxPrevLayerSize)

	// prevLayerNeuronCount tracks the number of neurons in the layer that feeds into the current one.
	// It starts with inputSize for the first hidden layer.
	prevLayerNeuronCount := inputSize

	// Create Hidden Layers
	for i := 0; i < numHiddenLayers; i++ {
		currentHiddenLayerSize := hiddenConfig[i]
		hiddenLayer := &Layer{
			Neurons:    make([]*Neuron, currentHiddenLayerSize), // Exported field name
			Activation: ReLU{},                                 // Exported field name
		}
		for j := 0; j < currentHiddenLayerSize; j++ {
			hiddenLayer.Neurons[j] = &Neuron{ // Exported field name
				Weights:  make([]float32, prevLayerNeuronCount), // Exported field name
				Bias:     xavierInit(prevLayerNeuronCount, currentHiddenLayerSize, nn.Params), // Exported field name
				Momentum: make([]float32, prevLayerNeuronCount), // Exported field name
			}
			for k := range hiddenLayer.Neurons[j].Weights { // Exported field names
				hiddenLayer.Neurons[j].Weights[k] = xavierInit(prevLayerNeuronCount, currentHiddenLayerSize, nn.Params) // Exported field name
			}
		}
		nn.Layers[i] = hiddenLayer // Exported field name
		prevLayerNeuronCount = currentHiddenLayerSize // Update for the next layer's input count
	}

	// Create Output Layer
	outputLayer := &Layer{
		Neurons:    make([]*Neuron, outputSize), // Exported field name
		Activation: Linear{},                     // Exported field name
	}
	for l := 0; l < outputSize; l++ {
		outputLayer.Neurons[l] = &Neuron{ // Exported field name
			Weights: make([]float32, prevLayerNeuronCount), // Exported field name
			Bias:     xavierInit(prevLayerNeuronCount, outputSize, nn.Params), // Exported field name
			Momentum: make([]float32, prevLayerNeuronCount), // Exported field name
		}
		for k := range outputLayer.Neurons[l].Weights { // Exported field names
			outputLayer.Neurons[l].Weights[k] = xavierInit(prevLayerNeuronCount, outputSize, nn.Params) // Exported field name
		}
	}
	nn.Layers[numHiddenLayers] = outputLayer // Exported field name

	return nn
}

func NewNeuralNetwork(inputSize int, hiddenConfig []int, outputSize int, params Params) *NeuralNetwork {
	return initialise(inputSize, hiddenConfig, outputSize, params)
}


// FeedForward propagates the input signal through the network, layer by layer,
// calculating the output of each neuron based on weights, biases, and activation functions.
func (nn *NeuralNetwork) FeedForward(input []float32) {
	// Input is now []float32. Copy it to the internal nn.Input buffer.
	// nn.Input slice is preallocated in initialise.
	if len(input) != len(nn.Input) { // Exported field name
		panic(fmt.Sprintf("FeedForward: input size %d does not match network input size %d", len(input), len(nn.Input))) // Exported field name
	}
	copy(nn.Input, input) // Exported field name

	// Process the first layer (connected to the input)
	for _, neuron := range nn.Layers[0].Neurons { // Exported field names
		neuron.Output = neuron.Bias // Exported field names
		for j := 0; j < len(nn.Input); j++ { // Exported field name
			neuron.Output += nn.Input[j] * neuron.Weights[j] // Exported field names
		}
		neuron.Output = nn.Layers[0].Activation.Activate(neuron.Output) // Exported field names
		neuron.Output = capValue(neuron.Output) // Exported field name
	}
	// Process hidden and output layers (i >= 1) using direct loops
	for i := 1; i < len(nn.Layers); i++ { // Exported field name
		for _, neuron := range nn.Layers[i].Neurons { // Exported field names
			var sum64 float64 = float64(neuron.Bias) // Exported field name
			for j := 0; j < len(nn.Layers[i-1].Neurons); j++ { // Exported field names
				sum64 += float64(nn.Layers[i-1].Neurons[j].Output) * float64(neuron.Weights[j]) // Exported field names
			}
			neuron.Output = float32(sum64) // Exported field name
			neuron.Output = nn.Layers[i].Activation.Activate(neuron.Output) // Exported field names
			neuron.Output = capValue(neuron.Output) // Exported field name
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

	for _, layer := range nn.Layers { // Exported field name
		for _, neuron := range layer.Neurons { // Exported field name
			// Update weights
			if neuron.AccumulatedWeightGradients != nil { // Exported field name
				for wIdx := range neuron.Weights { // Exported field name
					avgGrad64 := float64(neuron.AccumulatedWeightGradients[wIdx]) / float64(fBatchSize) // Exported field name
					avgGrad64 += float64(nn.Params.L2) * float64(neuron.Weights[wIdx]) // Exported field names

					momentum64 := float64(nn.Params.MomentumCoefficient)*float64(neuron.Momentum[wIdx]) + float64(learningRate)*avgGrad64 // Exported field names
					neuron.Momentum[wIdx] = float32(momentum64) // Exported field name
					neuron.Weights[wIdx] = float32(float64(neuron.Weights[wIdx]) - momentum64) // Exported field name
					neuron.Weights[wIdx] = capValue(neuron.Weights[wIdx]) // Exported field name
				}
			}

			// Update bias
			avgBiasGrad64 := float64(neuron.AccumulatedBiasGradient) / float64(fBatchSize) // Exported field name
			neuron.Bias = float32(float64(neuron.Bias) - float64(learningRate)*avgBiasGrad64) // Exported field name
			neuron.Bias = capValue(neuron.Bias) // Exported field name
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
		Layers: make([]*Layer, len(nn.Layers)),
		Params: nn.Params,
	}

	// Deep copy all layers
	for i, layer := range nn.Layers { // Corrected: use receiver nn
		cloneLayer := &Layer{
			Neurons:    make([]*Neuron, len(layer.Neurons)),
			Activation: layer.Activation,
		}

		// Deep copy all neurons
		for j, neuron := range layer.Neurons {
			cloneNeuron := &Neuron{
				Weights:  make([]float32, len(neuron.Weights)),
				Bias:     neuron.Bias,
				Output:   neuron.Output,
				Momentum: make([]float32, len(neuron.Momentum)), // Initialize momentum slice
			}

			// Copy weights
			copy(cloneNeuron.Weights, neuron.Weights)
			// Copy momentum
			if neuron.Momentum != nil { // Guard against nil if original momentum could be nil (though init suggests it won't be)
				copy(cloneNeuron.Momentum, neuron.Momentum)
			}
			cloneLayer.Neurons[j] = cloneNeuron
		}

		clone.Layers[i] = cloneLayer
	}

	// Copy input if it exists
	if nn.Input != nil {
		clone.Input = make([]float32, len(nn.Input))
		copy(clone.Input, nn.Input)
	}
	if nn.PrevLayerOutputsBuffer != nil {
		clone.PrevLayerOutputsBuffer = make([]float32, len(nn.PrevLayerOutputsBuffer))
		copy(clone.PrevLayerOutputsBuffer, nn.PrevLayerOutputsBuffer)
	}

	return clone
}

// zeroAccumulatedGradients resets the gradient accumulators for all neurons before processing a new mini-batch.
func (nn *NeuralNetwork) zeroAccumulatedGradients() {
	for _, layer := range nn.Layers { // Exported field name
		for _, neuron := range layer.Neurons { // Exported field name
			if neuron.AccumulatedWeightGradients == nil || len(neuron.AccumulatedWeightGradients) != len(neuron.Weights) { // Exported field names
				neuron.AccumulatedWeightGradients = make([]float32, len(neuron.Weights)) // Exported field names
			} else {
				for k := range neuron.AccumulatedWeightGradients { // Exported field name
					neuron.AccumulatedWeightGradients[k] = 0.0 // Exported field name
				}
			}
			neuron.AccumulatedBiasGradient = 0.0 // Exported field name
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
					for layerIdx, cloneLayer := range clone.Layers {
						for neuronIdx, cloneNeuron := range cloneLayer.Neurons {
							mainNeuron := nn.Layers[layerIdx].Neurons[neuronIdx]
							if cloneNeuron.AccumulatedWeightGradients != nil {
								for wIdx, grad := range cloneNeuron.AccumulatedWeightGradients {
									mainNeuron.AccumulatedWeightGradients[wIdx] += grad
								}
							}
							mainNeuron.AccumulatedBiasGradient += cloneNeuron.AccumulatedBiasGradient
						}
					}
				}
			}

			// After processing all samples in the mini-batch (either single or multi-threaded), apply the averaged gradients
			nn.applyAveragedGradients(currentMiniBatchSize, nn.Params.Lr)

			totalEpochLoss += miniBatchLoss
			samplesProcessedInEpoch += currentMiniBatchSize
		}

		averageEpochLoss := float32(0.0)
		if samplesProcessedInEpoch > 0 {
			averageEpochLoss = totalEpochLoss / float32(samplesProcessedInEpoch)
		}

		fmt.Printf("Loss MiniBatch Epoch %d = %.2f (LR: %.5f)\n", e, averageEpochLoss, nn.Params.Lr) // Exported field name

		// Apply learning rate decay at the end of each epoch.
		nn.Params.Lr *= nn.Params.Decay // Exported field names
	} // End of epoch loop
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
	props := nn.SoftmaxProbabilities() // Get softmax probabilities (float32)
	if len(props) != len(labelSample) {
		panic("backpropagateAndAccumulateForSample: props and labelSample length mismatch")
	}
	errVecData := make([]float64, len(props)) // Use []float64 directly
	for i := range props {
		errVecData[i] = float64(props[i]) - float64(labelSample[i]) // Manual subtraction, ensure props[i] is float64
	}

	loss := nn.calculateLoss(labelSample) // Calculate loss for this sample

	// 3. Backpropagate Deltas: Calculate error signals (deltas) for each layer, starting from the output.
	//    The delta for a neuron represents how much its pre-activation input needs to change
	//    to reduce the overall loss.

	// Calculate deltas for the output layer (using the simplified gradient)
	outputLayer := nn.Layers[len(nn.Layers)-1] // Exported field name
	if outputLayer.Deltas == nil || len(outputLayer.Deltas) != len(outputLayer.Neurons) { // Exported field names
		outputLayer.Deltas = make([]float32, len(outputLayer.Neurons)) // Exported field names
	}
	for j := 0; j < len(outputLayer.Neurons); j++ { // Exported field name
		outputLayer.Deltas[j] = capValue(float32(errVecData[j])) // Exported field name
	}

	// Propagate deltas backward through hidden layers
	for i := len(nn.Layers) - 2; i >= 0; i-- { // Exported field name
		layer := nn.Layers[i]     // Exported field name
		nextLayer := nn.Layers[i+1] // Exported field name

		if layer.Deltas == nil || len(layer.Deltas) != len(layer.Neurons) { // Exported field names
			layer.Deltas = make([]float32, len(layer.Neurons)) // Exported field names
		}

		for j, neuron := range layer.Neurons { // Exported field name
			var errorSumTimesWeight64 float64 = 0.0
			for k, nextNeuron := range nextLayer.Neurons { // Exported field name
				errorSumTimesWeight64 += float64(nextNeuron.Weights[j]) * float64(nextLayer.Deltas[k]) // Exported field names
			}
			derivative := layer.Activation.Derivative(neuron.Output) // Exported field names
			layer.Deltas[j] = capValue(float32(errorSumTimesWeight64*float64(derivative))) // Exported field name
		}
	}

	// 4. Accumulate Gradients
	for layerIndex, layer := range nn.Layers { // Exported field name
		var prevLayerOutputs []float32
		if layerIndex == 0 {
			prevLayerOutputs = nn.Input // Exported field name
		} else {
			prevLayer := nn.Layers[layerIndex-1] // Exported field name
			currentPrevLayerNumNeurons := len(prevLayer.Neurons) // Exported field name
			if cap(nn.PrevLayerOutputsBuffer) < currentPrevLayerNumNeurons { // Exported field name
				nn.PrevLayerOutputsBuffer = make([]float32, currentPrevLayerNumNeurons) // Exported field name
			}
			prevLayerOutputs = nn.PrevLayerOutputsBuffer[:currentPrevLayerNumNeurons] // Exported field name
			for pIdx, pNeuron := range prevLayer.Neurons { // Exported field name
				prevLayerOutputs[pIdx] = pNeuron.Output // Exported field name
			}
		}

		for nIdx, neuron := range layer.Neurons { // Exported field name
			sampleDelta := layer.Deltas[nIdx] // Exported field name

			if neuron.AccumulatedWeightGradients != nil { // Exported field name
				for wIdx := range neuron.Weights { // Exported field name
					gradContrib64 := float64(sampleDelta) * float64(prevLayerOutputs[wIdx])
					neuron.AccumulatedWeightGradients[wIdx] += float32(gradContrib64) // Exported field name
				}
			}
			neuron.AccumulatedBiasGradient += sampleDelta // Exported field name
		}
	}
	return loss
}

// TrainBatch function removed as unused (superseded by TrainMiniBatch).

func (nn *NeuralNetwork) Output() []float32 {
	outputLayer := nn.Layers[len(nn.Layers)-1].Neurons // Use unexported field names
	output := make([]float32, len(outputLayer))
	for i, neuron := range outputLayer {
		output[i] = neuron.Output // Use unexported field name
	}
	return output
}

func (nn *NeuralNetwork) Predict(data []float32) int {
	nn.FeedForward(data)
	propsData := nn.SoftmaxProbabilities() // returns []float32
	if len(propsData) == 0 {
		panic("Predict: SoftmaxProbabilities returned empty slice") // Or handle error appropriately
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

// SoftmaxProbabilities calculates the softmax probabilities for the output layer.
// Renamed from calculateProps for clarity. Returns []float32 for consistency.
func (nn *NeuralNetwork) SoftmaxProbabilities() []float32 {
	outputLayerNeurons := nn.Layers[len(nn.Layers)-1].Neurons // Exported field names
	outputValues := make([]float32, len(outputLayerNeurons))
	for i, neuron := range outputLayerNeurons {
		neuron.Output = capValue(neuron.Output) // Ensure output is capped before softmax
		outputValues[i] = neuron.Output         // Exported field name
	}
	// Apply softmax to the final layer's outputs to get probabilities.
	softmaxProbs := softmax(outputValues, nn.Params) // Exported field name
	// No need to convert to float64 here, return float32 directly.
	return softmaxProbs
}

// calculateLoss computes the cross-entropy loss for a single sample, including L2 regularization.
// Cross-Entropy Loss: - Sum_i (target_i * log(predicted_probability_i))
// L2 Regularization Term: (lambda / 2) * Sum_all_weights (weight^2)
func (nn *NeuralNetwork) calculateLoss(target []float32) float32 {
	softmaxProbs := nn.SoftmaxProbabilities() // Returns []float32
	var loss float32 = 0.0

	// Calculate Cross-Entropy part
	// propsData is already []float64
	// targetData is now the input []float32

	// Use float64 for intermediate loss calculation
	var loss64 float64
	if len(softmaxProbs) != len(target) {
		panic("calculateLoss: softmaxProbs and target length mismatch")
	}
	for i := 0; i < len(softmaxProbs); i++ {
		p64 := math.Max(float64(softmaxProbs[i]), 1e-15) // Ensure p64 is float64
		loss64 -= float64(target[i]) * math.Log(p64)
	}
	// Calculate L2 regularization part
	var reg64 float64 = 0.0
	for _, layer := range nn.Layers { // Exported field name
		for _, neuron := range layer.Neurons { // Exported field name
			for _, w := range neuron.Weights { // Exported field name
				reg64 += float64(w) * float64(w) // Sum of squared weights
			}
		}
	}
	// Add L2 penalty to the loss.
	loss64 += 0.5 * float64(nn.Params.L2) * reg64 // Exported field name
	loss = float32(loss64) // Cast final result back to float32

	// Guard against NaN or Inf results, returning 0 loss in such cases.
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

	// Apply capValue to the results.
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
	for i, neuron := range l.Neurons {
		sb.WriteString(fmt.Sprintf("Neuron %d: output=%.2f\n", i, neuron.Output))
	}

	// Print deltas
	sb.WriteString(fmt.Sprintf("Deltas: %v\n", l.Deltas))

	return sb.String()
}

func (nn *NeuralNetwork) String() string {
	var sb strings.Builder

	// Iterate through the layers and print them
	for i, layer := range nn.Layers {
		sb.WriteString(fmt.Sprintf("Layer %d:\n%s\n", i, layer.String()))
	}

	return sb.String()
}
