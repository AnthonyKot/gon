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

	"gonum.org/v1/gonum/mat"
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

// Represents of the simlest NN.
type NeuralNetwork struct {
	layers                  []*Layer
	input                   []float32
	params                  Params
	prevLayerOutputsBuffer []float32 // Buffer for backpropagation
	optimizer               Optimizer   // Pluggable optimizer interface
	lossFunction            LossFunction // Pluggable loss function
}

type Params struct {
	lr     float32
	decay  float32
	L2     float32
	lowCap float32
	relu   float32
	// jacobian bool // Removed, no longer used
	bn                  float32
	MomentumCoefficient float32
	// UseFloat64          bool // Removed: Calculations will consistently use float64 internally
}

type Task struct {
	data   mat.VecDense
	output mat.VecDense
}

func CreateTask(data mat.VecDense, output mat.VecDense) Task {
	return Task{
		data:   data,
		output: output,
	}
}

func NewParams(learningRate float32, decay float32, regularization float32, cap float32) Params {
	// Calls NewParamsFull, providing default values for relu, momentum, and bn
	defaults := defaultParams()
	return NewParamsFull(learningRate, decay, regularization, cap, defaults.relu, defaults.MomentumCoefficient, defaults.bn)
}

func NewParamsFull(learningRate float32, decay float32, regularization float32, cap float32, relu float32, momentumCoefficient float32, bn float32) Params {
	return Params{
		lr:                  learningRate,
		decay:               decay,
		L2:                  regularization,
		lowCap:              cap,
		relu:                relu,
		bn:                  bn,
		MomentumCoefficient: momentumCoefficient,
		// UseFloat64 field removed
	}
}

func defaultParams() *Params {
	return &Params{
		lr:     0.01,
		decay:  0.95, // Reduced decay rate
		L2:     1e-4, // Enabled L2 regularization
		lowCap: 0,
		relu:   0,
		// jacobian:false, // Removed
		bn:                  0.0,
		MomentumCoefficient: 0.9, // Default momentum coefficient
	}
}

func DefaultNeuralNetwork(inputSize int, hidden []int, outputSize int) *NeuralNetwork {
	params := defaultParams()
	nn := initialise(inputSize, hidden, outputSize, *params)
	nn.optimizer = &SGD{}             // Default optimizer
	nn.lossFunction = &CrossEntropy{} // Default loss
	return nn
}

func initialise(inputSize int, hiddenConfig []int, outputSize int, params Params) *NeuralNetwork {
	rand.Seed(time.Now().UnixNano())

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
				bias:     xavierInit(prevLayerNeuronCount, currentHiddenLayerSize, nn.params),
				momentum: make([]float32, prevLayerNeuronCount),
			}
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
			bias:     xavierInit(prevLayerNeuronCount, outputSize, nn.params),
			momentum: make([]float32, prevLayerNeuronCount),
		}
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

func (nn *NeuralNetwork) SetActivation(layerIndex int, activation ActivationFunction) {
	nn.layers[layerIndex].activation = activation
}

func (nn *NeuralNetwork) FeedForward(input mat.VecDense) {
	// Convert input mat.VecDense to []float32 once and store in nn.input.
	// Reuse nn.input slice if possible to reduce allocations.
	// nn.input is preallocated in initialise; no need to allocate here
	for j := 0; j < input.Len(); j++ {
		nn.input[j] = float32(input.AtVec(j))
	}

	// The first layer takes inputs as X from nn.input
	for _, neuron := range nn.layers[0].neurons {
		neuron.output = neuron.bias
		for j := 0; j < len(nn.input); j++ { // Iterate over nn.input
			neuron.output += nn.input[j] * neuron.weights[j] // Use nn.input directly
		}
		neuron.output = nn.layers[0].activation.Activate(neuron.output) // Removed UseFloat64 flag
		neuron.output = capValue(neuron.output, nn.params)
	}
	// nn.input is already set.
	for i := 1; i < len(nn.layers); i++ {
		for _, neuron := range nn.layers[i].neurons {
			neuron.output = neuron.bias
			for j := 0; j < len(nn.layers[i-1].neurons); j++ {
				neuron.output += nn.layers[i-1].neurons[j].output * neuron.weights[j]
			}
			neuron.output = nn.layers[i].activation.Activate(neuron.output) // Removed UseFloat64 flag
			neuron.output = capValue(neuron.output, nn.params)
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
					// Add L2 regularization gradient component (using float64 intermediate)
					// avgGrad variable removed as it was unused.
					avgGrad64 := float64(neuron.accumulatedWeightGradients[wIdx])/float64(fBatchSize) + float64(nn.params.L2)*float64(neuron.weights[wIdx])

					momentum64 := float64(nn.params.MomentumCoefficient)*float64(neuron.momentum[wIdx]) + float64(learningRate)*avgGrad64
					neuron.momentum[wIdx] = float32(momentum64)
					neuron.weights[wIdx] = float32(float64(neuron.weights[wIdx]) - momentum64) // Update using float64 intermediate
					neuron.weights[wIdx] = capValue(neuron.weights[wIdx], nn.params)
				}
			}

			// Update bias (using float64 intermediate)
			avgBiasGrad64 := float64(neuron.accumulatedBiasGradient) / float64(fBatchSize)
			neuron.bias = float32(float64(neuron.bias) - float64(learningRate)*avgBiasGrad64)
			neuron.bias = capValue(neuron.bias, nn.params)
		}
	}
}

// The old UpdateWeights function is now replaced by applyAveragedGradients
// and the gradient accumulation logic within backpropagateAndAccumulateForSample.

// TrainSGD is updated to use the new gradient accumulation and application mechanism.
// For SGD, the "batch size" is 1 for gradient application.
func (nn *NeuralNetwork) TrainSGD(trainingData []mat.VecDense, expectedOutputs []mat.VecDense, epochs int) {
	numSamples := len(trainingData)
	if numSamples == 0 {
		fmt.Println("TrainSGD: No training data provided.")
		return
	}

	for e := 0; e < epochs; e++ {
		var totalEpochLoss float32 = 0.0
		permutation := rand.Perm(numSamples) // Shuffle data for each epoch

		for _, idx := range permutation { // Iterate through shuffled samples
			dataSample := trainingData[idx]
			labelSample := expectedOutputs[idx]

			// For SGD, gradients are calculated and applied for each sample individually.
			nn.zeroAccumulatedGradients() // Zero out before processing the single sample

			sampleLoss := nn.backpropagateAndAccumulateForSample(dataSample, labelSample)
			totalEpochLoss += sampleLoss

			// Apply gradients for this single sample (effective batchSize=1)
			nn.applyAveragedGradients(1, nn.params.lr)
		}

		// Apply learning rate decay once per epoch
		nn.params.lr *= nn.params.decay

		if numSamples > 0 {
			averageLoss := totalEpochLoss / float32(numSamples)
			fmt.Printf("Loss SGD %d = %.2f\n", e, averageLoss)
		} else {
			fmt.Printf("Loss SGD %d = 0.00 (No samples processed)\n", e)
		}
	}
}

// Clone creates a deep copy of a neural network for thread-safe parallel processing.
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
func (nn *NeuralNetwork) TrainMiniBatch(trainingData []mat.VecDense, expectedOutputs []mat.VecDense, batchSize int, epochs int, numWorkers int) {
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
		shuffledTrainingData := make([]mat.VecDense, numSamples)
		shuffledExpectedOutputs := make([]mat.VecDense, numSamples)
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
// computes sample-specific deltas, and accumulates gradients for a single sample.
// It returns the loss for this sample.
func (nn *NeuralNetwork) backpropagateAndAccumulateForSample(dataSample mat.VecDense, labelSample mat.VecDense) float32 {
	// 1. FeedForward for the current sample
	nn.FeedForward(dataSample)

	// 2. Calculate error vector for this sample (softmax_output - target)
	props := nn.calculateProps() // Uses current nn.output (from dataSample's FeedForward)
	errVec := mat.NewVecDense(props.Len(), nil)
	errVec.SubVec(props, &labelSample)

	loss := nn.calculateLoss(labelSample) // Uses current nn.output

	// 3. Backpropagate error for THIS sample to get sample-specific deltas.
	//    These deltas are stored in layer.deltas.

	// Calculate deltas for the output layer
	outputLayer := nn.layers[len(nn.layers)-1]
	if outputLayer.deltas == nil || len(outputLayer.deltas) != len(outputLayer.neurons) {
		outputLayer.deltas = make([]float32, len(outputLayer.neurons))
	}
	errVecData := errVec.RawVector().Data // Get raw data from errVec
	for j := 0; j < len(outputLayer.neurons); j++ {
		// For a single sample, the delta is the error component (softmax_output - target_j).
		outputLayer.deltas[j] = capValue(float32(errVecData[j]), nn.params) // Use raw data
	}

	// Propagate deltas backward through hidden layers
	for i := len(nn.layers) - 2; i >= 0; i-- {
		layer := nn.layers[i]
		nextLayer := nn.layers[i+1]

		if layer.deltas == nil || len(layer.deltas) != len(layer.neurons) {
			layer.deltas = make([]float32, len(layer.neurons))
		}

		for j, neuron := range layer.neurons { // For each neuron 'j' in current layer 'i'
			var errorSumTimesWeight float32 = 0.0
			// Sum (delta_k_nextLayer * weight_kj_nextLayer)
			for k, nextNeuron := range nextLayer.neurons { // For each neuron 'k' in next layer 'i+1'
				// nextNeuron.weights[j] is the weight connecting neuron 'j' (current layer) to neuron 'k' (next layer)
				errorSumTimesWeight += nextNeuron.weights[j] * nextLayer.deltas[k]
			}
			// Delta for neuron 'j' in layer 'i' = errorSumTimesWeight * derivative_of_activation(neuron 'j' output)
			// Use float64 for intermediate sum
			var errorSumTimesWeight64 float64
			for k, nextNeuron := range nextLayer.neurons {
				errorSumTimesWeight64 += float64(nextNeuron.weights[j]) * float64(nextLayer.deltas[k])
			}
			derivative := layer.activation.Derivative(neuron.output) // Removed UseFloat64 flag
			layer.deltas[j] = capValue(float32(errorSumTimesWeight64*float64(derivative)), nn.params)
		}
	}

	// 4. Accumulate gradients based on these sample-specific deltas and current activations
	//    (nn.input and neuron.output are from the current sample's FeedForward pass).
	for layerIndex, layer := range nn.layers {
		var prevLayerOutputs []float32
		if layerIndex == 0 {
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

func (nn *NeuralNetwork) TrainBatch(trainingData []mat.VecDense, expectedOutputs []mat.VecDense, epochs int) {
	numSamples := len(trainingData)
	if numSamples == 0 {
		fmt.Println("TrainBatch: No training data provided.")
		return
	}

	for e := 0; e < epochs; e++ {
		nn.zeroAccumulatedGradients() // Zero out accumulators at the start of each epoch
		var totalEpochLoss float32 = 0.0

		// Shuffle data for each epoch
		permutation := rand.Perm(numSamples)

		for _, idx := range permutation { // Iterate through shuffled samples
			dataSample := trainingData[idx]
			labelSample := expectedOutputs[idx]

			sampleLoss := nn.backpropagateAndAccumulateForSample(dataSample, labelSample)
			totalEpochLoss += sampleLoss
		}

		// After processing all samples in the batch, apply the averaged gradients
		nn.applyAveragedGradients(numSamples, nn.params.lr)

		averageLoss := totalEpochLoss / float32(numSamples)
		fmt.Printf("Loss Batch %d = %.2f\n", e, averageLoss)

		// Apply learning rate decay
		nn.params.lr *= nn.params.decay
	}
}

func (nn *NeuralNetwork) Output() []float32 {
	outputLayer := nn.layers[len(nn.layers)-1].neurons
	output := make([]float32, len(outputLayer))
	for i, neuron := range outputLayer {
		output[i] = neuron.output
	}
	return output
}

func (nn *NeuralNetwork) Predict(data mat.VecDense) int {
	nn.FeedForward(data)
	props := nn.calculateProps()
	propsData := props.RawVector().Data // Get raw data
	maxVal := propsData[0]              // Access raw data
	idx := 0
	for i := 1; i < len(propsData); i++ { // Iterate using len of slice
		val := propsData[i]             // Access raw data
		if val > maxVal {
			maxVal = val
			idx = i
		}
	}
	return idx
}

func (nn *NeuralNetwork) calculateProps() *mat.VecDense {
	outputLayer := nn.layers[len(nn.layers)-1].neurons
	output := make([]float32, len(outputLayer))
	for i, neuron := range outputLayer {
		neuron.output = capValue(neuron.output, nn.params)
		output[i] = neuron.output
	}
	softmax := softmax(output, nn.params)
	softmaxFloat64 := make([]float64, len(softmax))
	for i, v := range softmax {
		v = capValue(v, nn.params)
		softmaxFloat64[i] = float64(v)
	}
	return mat.NewVecDense(len(outputLayer), softmaxFloat64)
}

func (nn *NeuralNetwork) calculateLoss(target mat.VecDense) float32 {
	props := nn.calculateProps()
	var loss float32 = 0.0

	// Optimization: Access raw vector data to avoid repeated AtVec calls
	propsData := props.RawVector().Data
	targetData := target.RawVector().Data // target is mat.VecDense, so direct RawVector()

	// Use float64 for intermediate loss calculation
	var loss64 float64
	for i := 0; i < len(propsData); i++ {
		p64 := math.Max(propsData[i], 1e-15) // propsData is already float64
		loss64 -= targetData[i] * math.Log(p64) // targetData is already float64
	}
	// L2 regularization: (lambda/2) * sum weights^2 (using float64)
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

func convertWeightsDense(neurons []*Neuron) *mat.Dense {
	n := len(neurons)
	m := len(neurons[0].weights)
	weights := make([]float64, n*m)

	for j, _ := range neurons[0].weights {
		for i, neuron := range neurons {
			weights[i*m+j] = float64(neuron.weights[j])
		}
	}

	return mat.NewDense(n, m, weights)
}

func convertBiasToDense(neurons []*Neuron) *mat.VecDense {
	dense := mat.NewVecDense(len(neurons), nil)
	for i, neuron := range neurons {
		dense.SetVec(i, float64(neuron.bias))
	}
	return dense
}


// stableSoftmax computes softmax in a numerically stable way.
func stableSoftmax(output []float32) []float32 {
	if len(output) == 0 {
		return []float32{}
	}
	// Subtract max for numerical stability
	maxVal := output[0]
	for _, v := range output[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	exps := make([]float32, len(output))
	var sumExps float32 = 0.0
	for i, v := range output {
		exps[i] = float32(math.Exp(float64(v - maxVal)))
		sumExps += exps[i]
	}

	// Handle case where sumExps is zero (e.g., all inputs were very small, leading to exp(v-max) -> 0)
	// or if sumExps becomes Inf (less likely with max subtraction but theoretically possible with extreme inputs)
	if sumExps == 0 || math.IsInf(float64(sumExps), 0) {
		// Fallback: distribute probability uniformly.
		// This prevents division by zero or NaN results (e.g. 0/0 or Inf/Inf).
		// This situation indicates that the network outputs are either all extremely small or problematic.
		uniformProb := 1.0 / float32(len(exps))
		for i := range exps {
			exps[i] = uniformProb
		}
		return exps
	}

	for i := range exps {
		exps[i] /= sumExps
	}
	return exps
}

func softmax(output []float32, params Params) []float32 {
	// Use the numerically stable softmax calculation.
	stableOutput := stableSoftmax(output)

	// Apply capValue to the results. With default params (lowCap=0),
	// this primarily handles any residual NaNs by converting them to 0.
	// If lowCap is non-zero, it will enforce min/max probability magnitudes.
	for i, v := range stableOutput {
		stableOutput[i] = capValue(v, params)
	}
	return stableOutput
}

func xavierInit(numInputs int, numOutputs int, params Params) float32 {
	limit := math.Sqrt(6.0 / float64(numInputs+numOutputs))
	xavier := float32(2*rand.Float64()*limit - limit)
	return capValue(xavier, params)
}

func capValue(value float32, params Params) float32 {
	if math.IsNaN(float64(value)) {
		return params.lowCap
	}
	if math.IsInf(float64(value), 1) { // +Inf
		return DefaultMaxAbsValue
	}
	if math.IsInf(float64(value), -1) { // -Inf
		return -DefaultMaxAbsValue
	}

	// If lowCap is 0, no further capping for finite values beyond NaN/Inf.
	if params.lowCap == 0 {
		// Optional: could cap 'value' against DefaultMaxAbsValue here too for extreme finite values
		// if abs(value) > DefaultMaxAbsValue, but current logic defers this to lowCap != 0 case.
		return value
	}

	// This part executes if params.lowCap != 0 (and value is finite).
	// params.lowCap is assumed to be the minimum desired absolute magnitude.
	// Ensure minMagnitude is non-negative.
	minMagnitude := params.lowCap
	if params.lowCap < 0 { // If lowCap is negative (unusual), treat min magnitude as 0.
		minMagnitude = 0
	}

	// Determine the effective upper cap for magnitude.
	effectiveUpperCap := DefaultMaxAbsValue
	if params.lowCap > 0 { // Ensure lowCap is positive before division for 1/lowCap.
		calculatedUpperCap := 1.0 / params.lowCap
		// Use calculatedUpperCap if it's smaller than DefaultMaxAbsValue and positive.
		if calculatedUpperCap < effectiveUpperCap && calculatedUpperCap > 0 {
			effectiveUpperCap = calculatedUpperCap
		}
	}

	// Use absolute value for capping logic, then restore sign.
	absVal := value
	sign := float32(1.0)
	if value < 0 {
		absVal = -value
		sign = -1.0
	}

	cappedAbsVal := absVal
	if cappedAbsVal < minMagnitude { // Apply minimum magnitude
		cappedAbsVal = minMagnitude
	}
	if cappedAbsVal > effectiveUpperCap { // Apply maximum magnitude
		cappedAbsVal = effectiveUpperCap
	}

	return sign * cappedAbsVal
}

func selectSamples(trainingData []mat.VecDense, expectedOutputs []mat.VecDense, samples int) ([]mat.VecDense, []mat.VecDense) {
	selectedIndices := make(map[int]bool)
	for len(selectedIndices) < samples {
		randomIndex := rand.Intn(len(trainingData))
		if !selectedIndices[randomIndex] {
			selectedIndices[randomIndex] = true
		}
	}
	selectedInputs := make([]mat.VecDense, samples)
	selectedLabels := make([]mat.VecDense, samples)
	// Iterate over selected indices and copy corresponding elements
	i := 0
	for index := range selectedIndices {
		selectedInputs[i] = trainingData[index]
		selectedLabels[i] = expectedOutputs[index]
		i++
	}
	return selectedInputs, selectedLabels
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
	println("Model saved as " + filename)
}
func loadModel(filename string) *NeuralNetwork {
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
	println("Model load as " + filename)
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
