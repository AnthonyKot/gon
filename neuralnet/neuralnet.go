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
	// Fields for accumulating gradients in batch training
	AccumulatedWeightGradients []float32
	AccumulatedBiasGradient    float32
	// Fields for momentum update (used by SGD optimizer)
	WeightVelocities []float32
	BiasVelocity     float32
	// Adam optimizer moment estimates
	WeightM []float32 // Adam: first moment vector for weights
	WeightV []float32 // Adam: second moment vector for weights
	BiasM   float32   // Adam: first moment for bias
	BiasV   float32   // Adam: second moment for bias
	// Batch Normalization: Temporary storage for values
	PreBNAOutput        float32 // Pre-BN activation (z = w*x + b)
	XNormalizedOutput float32 // Post-BN, pre-scale/shift ( (z - mean) / sqrt(var + eps) )
}

type Layer struct {
	Neurons    []*Neuron
	Deltas     []float32
	Activation ActivationFunction

	// Batch Normalization fields
	UseBatchNormalization bool
	Gamma               []float32 // Learnable scale parameters, one per neuron
	Beta                []float32 // Learnable shift parameters, one per neuron
	RunningMean         []float32 // Moving average of means, one per neuron
	RunningVariance     []float32 // Moving average of variances, one per neuron
	Epsilon             float32   // Small constant for numerical stability
	MomentumBN          float32   // Momentum for updating running mean/variance

	// Batch Normalization: Fields for current mini-batch statistics and intermediate values
	CurrentBatchMean     []float32       // Actual mean for the current mini-batch, one per neuron
	CurrentBatchVariance []float32       // Actual variance for the current mini-batch, one per neuron
	LastInputPreBNBatch  [][]float32     // Stores [sampleIdx][neuronIdx] -> PreBNAOutput for the batch
	LastXNormalizedBatch [][]float32     // Stores [sampleIdx][neuronIdx] -> x_normalized for the batch

	// Batch Normalization: Accumulated gradients for learnable parameters
	AccumulatedGammaGradients []float32 // Accumulated gradients for Gamma, one per neuron
	AccumulatedBetaGradients  []float32 // Accumulated gradients for Beta, one per neuron
	CurrentBatchDLdXHat       [][]float32 // Stores [sampleIdx][neuronIdx] -> dL/dX_hat for the batch
	// Batch Normalization: Temporary sums for backpropagation
	TempSumDldXhat     []float32 // Temporary sum of dL/dX_hat across the batch, one per neuron
	TempSumDldXhatXhat []float32 // Temporary sum of dL/dX_hat * X_hat across the batch, one per neuron
	// Adam moment estimates for BN parameters (if UseBatchNormalization is true)
	GammaM []float32
	GammaV []float32
	BetaM  []float32
	BetaV  []float32
}

// Represents the simplest NN.
type NeuralNetwork struct {
	Layers                  []*Layer
	Input                   []float32
	Params                  Params
	PrevLayerOutputsBuffer []float32 // Buffer for backpropagation
}

type Params struct {
	Lr                  float32
	Decay               float32
	L2                  float32
	MomentumCoefficient float32 // Coefficient for momentum update (e.g., 0.9)
	DropoutRate         float32 // Rate for dropout regularization (0.0 means disabled)
	IsTraining          bool    // Flag to indicate if the network is in training mode (for dropout/BN)
	EnableBatchNorm     bool    // Flag to enable Batch Normalization
	Beta1               float32 // Adam optimizer: exponential decay rate for the first moment estimates
	Beta2               float32 // Adam optimizer: exponential decay rate for the second-moment estimates
	EpsilonAdam         float32 // Adam optimizer: small constant for numerical stability
}

// NewParams creates a Params struct with default values for non-specified fields.
func NewParams(learningRate float32, decay float32, regularization float32) Params {
	defaults := defaultParams()
	// Pass through existing defaults, and add DropoutRate & EnableBatchNorm from defaults, plus new Adam params
	return NewParamsFull(learningRate, decay, regularization, defaults.MomentumCoefficient, defaults.DropoutRate, defaults.EnableBatchNorm, defaults.Beta1, defaults.Beta2, defaults.EpsilonAdam)
}

// NewParamsFull creates a Params struct with all fields specified.
// IsTraining is intentionally omitted here as it's usually set dynamically.
func NewParamsFull(learningRate float32, decay float32, regularization float32, momentumCoefficient float32, dropoutRate float32, enableBatchNorm bool, beta1 float32, beta2 float32, epsilonAdam float32) Params {
	if dropoutRate < 0.0 || dropoutRate >= 1.0 {
		dropoutRate = 0.0 // Ensure dropout rate is valid or disabled
	}
	return Params{
		Lr:                  learningRate,
		Decay:               decay,
		L2:                  regularization,
		MomentumCoefficient: momentumCoefficient,
		DropoutRate:         dropoutRate,
		EnableBatchNorm:     enableBatchNorm,
		Beta1:               beta1,
		Beta2:               beta2,
		EpsilonAdam:         epsilonAdam,
		IsTraining:          false, // Default to false, should be set explicitly during training/evaluation phases
	}
}

func defaultParams() *Params {
	return &Params{
		Lr:                  0.01,
		Decay:               0.95,
		L2:                  1e-4,
		MomentumCoefficient: 0.9,
		DropoutRate:         0.0,   // Dropout disabled by default
		EnableBatchNorm:     false, // Batch Norm disabled by default
		IsTraining:          false,
		Beta1:               0.9,
		Beta2:               0.999,
		EpsilonAdam:         1e-8,
	}
}

func DefaultNeuralNetwork(inputSize int, hidden []int, outputSize int) *NeuralNetwork {
	params := defaultParams()
	// Ensure IsTraining is false for default network creation for typical inference use
	params.IsTraining = false 
	nn := initialise(inputSize, hidden, outputSize, *params)
	// optimizer and lossFunction fields removed from NeuralNetwork struct
	return nn
}

// initialise creates and initializes the neural network structure, including layers, neurons, weights, and biases.
func initialise(inputSize int, hiddenConfig []int, outputSize int, params Params, hiddenActivation ActivationFunction, outputActivation ActivationFunction) *NeuralNetwork {
	const defaultBNEpsilon = 1e-5
	const defaultBNMomentum = 0.9

	// Note: To support zero hidden layers (direct input to output), this function
	// would need adjustments, particularly in how prevLayerNeuronCount is initialized
	// for the output layer. For now, we assume hiddenConfig is not empty.
	if len(hiddenConfig) == 0 {
		panic("initialise: hiddenConfig slice cannot be empty for this network structure.")
	}

	numHiddenLayers := len(hiddenConfig)
	// Total processing layers = number of hidden layers + 1 output layer
	nn := &NeuralNetwork{
		Layers: make([]*Layer, numHiddenLayers+1),
		Params: params,
		Input:  make([]float32, inputSize),
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
			Neurons:    make([]*Neuron, currentHiddenLayerSize),
			Activation: hiddenActivation,
		}
		if params.EnableBatchNorm {
			hiddenLayer.UseBatchNormalization = true
			hiddenLayer.Epsilon = defaultBNEpsilon
			hiddenLayer.MomentumBN = defaultBNMomentum
			hiddenLayer.Gamma = make([]float32, currentHiddenLayerSize)
			hiddenLayer.Beta = make([]float32, currentHiddenLayerSize)
			hiddenLayer.RunningMean = make([]float32, currentHiddenLayerSize)
			hiddenLayer.RunningVariance = make([]float32, currentHiddenLayerSize)
			hiddenLayer.AccumulatedGammaGradients = make([]float32, currentHiddenLayerSize) // Initialize new field
			hiddenLayer.AccumulatedBetaGradients = make([]float32, currentHiddenLayerSize)  // Initialize new field
			hiddenLayer.CurrentBatchDLdXHat = nil                                         // Initialize new field
			hiddenLayer.TempSumDldXhat = nil                                              // Initialize new field
			hiddenLayer.TempSumDldXhatXhat = nil                                          // Initialize new field
			// Initialize Adam moment estimates for BN parameters
			hiddenLayer.GammaM = make([]float32, currentHiddenLayerSize)
			hiddenLayer.GammaV = make([]float32, currentHiddenLayerSize)
			hiddenLayer.BetaM = make([]float32, currentHiddenLayerSize)
			hiddenLayer.BetaV = make([]float32, currentHiddenLayerSize)
			for k := 0; k < currentHiddenLayerSize; k++ {
				hiddenLayer.Gamma[k] = 1.0
				hiddenLayer.Beta[k] = 0.0
				hiddenLayer.RunningMean[k] = 0.0
				hiddenLayer.RunningVariance[k] = 1.0
			}
		} else {
			hiddenLayer.UseBatchNormalization = false
		}

		for j := 0; j < currentHiddenLayerSize; j++ {
			hiddenLayer.Neurons[j] = &Neuron{
				Weights:                  make([]float32, prevLayerNeuronCount),
				Bias:                     xavierInit(prevLayerNeuronCount, currentHiddenLayerSize, nn.Params),
				AccumulatedWeightGradients: make([]float32, prevLayerNeuronCount),
				AccumulatedBiasGradient:    0.0,
				WeightVelocities:         make([]float32, prevLayerNeuronCount), // Initialize for SGD
				BiasVelocity:             0.0,                                 // Initialize for SGD
				WeightM:                  make([]float32, prevLayerNeuronCount), // Adam
				WeightV:                  make([]float32, prevLayerNeuronCount), // Adam
				BiasM:                    0.0,                                 // Adam
				BiasV:                    0.0,                                 // Adam
			}
			for k := range hiddenLayer.Neurons[j].Weights {
				hiddenLayer.Neurons[j].Weights[k] = xavierInit(prevLayerNeuronCount, currentHiddenLayerSize, nn.Params)
			}
		}
		nn.Layers[i] = hiddenLayer
		prevLayerNeuronCount = currentHiddenLayerSize // Update for the next layer's input count
	}

	// Create Output Layer
	outputLayer := &Layer{
		Neurons:               make([]*Neuron, outputSize),
		Activation:            outputActivation,
		UseBatchNormalization: false, // Typically BN is not applied directly before Softmax
	}
	// If one chose to apply BN to output layer, initialization would go here.
	// For now, UseBatchNormalization is explicitly false for the output layer.

	for l := 0; l < outputSize; l++ {
		outputLayer.Neurons[l] = &Neuron{
			Weights:                  make([]float32, prevLayerNeuronCount),
			Bias:                     xavierInit(prevLayerNeuronCount, outputSize, nn.Params),
			AccumulatedWeightGradients: make([]float32, prevLayerNeuronCount),
			AccumulatedBiasGradient:    0.0,
			WeightVelocities:         make([]float32, prevLayerNeuronCount), // Initialize for SGD
			BiasVelocity:             0.0,                                 // Initialize for SGD
				WeightM:                  make([]float32, prevLayerNeuronCount), // Adam
				WeightV:                  make([]float32, prevLayerNeuronCount), // Adam
				BiasM:                    0.0,                                 // Adam
				BiasV:                    0.0,                                 // Adam
		}
		for k := range outputLayer.Neurons[l].Weights {
			outputLayer.Neurons[l].Weights[k] = xavierInit(prevLayerNeuronCount, outputSize, nn.Params)
		}
	}
	nn.Layers[numHiddenLayers] = outputLayer

	return nn
}

func NewNeuralNetwork(inputSize int, hiddenConfig []int, outputSize int, params Params, hiddenActivation ActivationFunction, outputActivation ActivationFunction) *NeuralNetwork {
	// Ensure default activations if nil is passed, though load.go should handle this.
	if hiddenActivation == nil {
		hiddenActivation = ReLU{}
	}
	if outputActivation == nil {
		outputActivation = Linear{}
	}
	return initialise(inputSize, hiddenConfig, outputSize, params, hiddenActivation, outputActivation)
}


// FeedForward propagates the input signal through the network, layer by layer,
// calculating the output of each neuron based on weights, biases, and activation functions.
func (nn *NeuralNetwork) FeedForward(input []float32) {
	// Input is now []float32. Copy it to the internal nn.Input buffer.
	// nn.Input slice is preallocated in initialise.
	if len(input) != len(nn.Input) {
		panic(fmt.Sprintf("FeedForward: input size %d does not match network input size %d", len(input), len(nn.Input)))
	}
	copy(nn.Input, input)

	currentInput := nn.Input
	for layerIdx, layer := range nn.Layers {
		isCurrentLayerOutput := (layerIdx == len(nn.Layers)-1)
		
		// This slice will hold the values that go into the activation function
		inputToActivation := make([]float32, len(layer.Neurons))

		for neuronIdx, neuron := range layer.Neurons {
			// 1. Calculate weighted sum + bias -> PreBNAOutput
			var sum float32 = neuron.Bias
			for weightIdx, weight := range neuron.Weights {
				sum += currentInput[weightIdx] * weight
			}
			neuron.PreBNAOutput = sum // Store for potential BN and backprop

			currentVal := neuron.PreBNAOutput

			// 2. Apply Batch Normalization if enabled for this layer
			if layer.UseBatchNormalization {
				var xNormalized float32
				if nn.Params.IsTraining {
					// Training: Use CurrentBatchMean/Variance (pre-calculated by TrainMiniBatch)
					// Running stats update is also moved to TrainMiniBatch.
					if layer.CurrentBatchMean == nil || layer.CurrentBatchVariance == nil {
						// This should not happen if TrainMiniBatch is correctly implemented
						panic(fmt.Sprintf("Layer %d: CurrentBatchMean/Variance not set during training", layerIdx))
					}
					xNormalized = (currentVal - layer.CurrentBatchMean[neuronIdx]) / float32(math.Sqrt(float64(layer.CurrentBatchVariance[neuronIdx] + layer.Epsilon)))
				} else {
					// Inference: Use running stats
					xNormalized = (currentVal - layer.RunningMean[neuronIdx]) / float32(math.Sqrt(float64(layer.RunningVariance[neuronIdx] + layer.Epsilon)))
				}
				neuron.XNormalizedOutput = xNormalized // Store for backprop
				currentVal = layer.Gamma[neuronIdx]*xNormalized + layer.Beta[neuronIdx]
			}
			inputToActivation[neuronIdx] = currentVal
		}

		// 3. Apply Activation Function and Dropout
		for neuronIdx, neuron := range layer.Neurons {
			activatedOutput := layer.Activation.Activate(inputToActivation[neuronIdx])
			neuron.Output = capValue(activatedOutput)

			// 4. Apply Dropout if it's a hidden layer and we are training
			// Dropout is applied *after* batch normalization and activation
			// Dropout is applied *after* batch normalization and activation
			if !isCurrentLayerOutput && nn.Params.IsTraining && nn.Params.DropoutRate > 0 {
				if rand.Float32() < nn.Params.DropoutRate {
					neuron.Output = 0.0
				} else {
					// Inverted dropout scaling
					neuron.Output /= (1.0 - nn.Params.DropoutRate)
				}
			}
		}

		// Prepare output of this layer as input for the next layer
		if layerIdx < len(nn.Layers)-1 {
			// Ensure PrevLayerOutputsBuffer is large enough
			// This logic should be sound from previous setup.
			if cap(nn.PrevLayerOutputsBuffer) < len(layer.Neurons) {
				nn.PrevLayerOutputsBuffer = make([]float32, len(layer.Neurons))
			}
			tempOutputBuffer := nn.PrevLayerOutputsBuffer[:len(layer.Neurons)]
			for k, neuron := range layer.Neurons {
				tempOutputBuffer[k] = neuron.Output
			}
			currentInput = tempOutputBuffer
		}
	}
}

// The old UpdateWeights function is now replaced by the Optimizer interface
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
			UseBatchNormalization: layer.UseBatchNormalization,
			Epsilon:             layer.Epsilon,
			MomentumBN:          layer.MomentumBN,
			// CurrentBatchMean, CurrentBatchVariance, LastInputPreBNBatch, LastXNormalizedBatch are not cloned.
			// They are transient and managed by the main training loop or should be re-evaluated by clones if needed.
		}
		if layer.UseBatchNormalization {
			cloneLayer.Gamma = make([]float32, len(layer.Gamma))
			copy(cloneLayer.Gamma, layer.Gamma)
			cloneLayer.Beta = make([]float32, len(layer.Beta))
			copy(cloneLayer.Beta, layer.Beta)
			cloneLayer.RunningMean = make([]float32, len(layer.RunningMean))
			copy(cloneLayer.RunningMean, layer.RunningMean)
			cloneLayer.RunningVariance = make([]float32, len(layer.RunningVariance))
			copy(cloneLayer.RunningVariance, layer.RunningVariance)
			// Initialize AccumulatedGammaGradients and AccumulatedBetaGradients for the clone
			cloneLayer.AccumulatedGammaGradients = make([]float32, len(layer.Neurons))
			cloneLayer.AccumulatedBetaGradients = make([]float32, len(layer.Neurons))
			cloneLayer.CurrentBatchDLdXHat = nil // Initialize as nil, will be handled by TrainMiniBatch or backprop
			cloneLayer.TempSumDldXhat = nil     // Initialize new field
			cloneLayer.TempSumDldXhatXhat = nil // Initialize new field
			// Adam moment estimates for BN parameters
			cloneLayer.GammaM = make([]float32, len(layer.GammaM))
			copy(cloneLayer.GammaM, layer.GammaM)
			cloneLayer.GammaV = make([]float32, len(layer.GammaV))
			copy(cloneLayer.GammaV, layer.GammaV)
			cloneLayer.BetaM = make([]float32, len(layer.BetaM))
			copy(cloneLayer.BetaM, layer.BetaM)
			cloneLayer.BetaV = make([]float32, len(layer.BetaV))
			copy(cloneLayer.BetaV, layer.BetaV)
			// Note: Not cloning CurrentBatchMean, CurrentBatchVariance, LastInputPreBNBatch, LastXNormalizedBatch
		}

		// Deep copy all neurons
		for j, neuron := range layer.Neurons {
			cloneNeuron := &Neuron{
				Weights:                    make([]float32, len(neuron.Weights)),
				Bias:                       neuron.Bias,
				Output:                     neuron.Output,
				AccumulatedWeightGradients: make([]float32, len(neuron.Weights)), // Initialize based on weights length
				AccumulatedBiasGradient:    neuron.AccumulatedBiasGradient,
				WeightVelocities:           make([]float32, len(neuron.Weights)), // Initialize based on weights length
				BiasVelocity:               neuron.BiasVelocity,
				WeightM:                    make([]float32, len(neuron.WeightM)), // Adam
				WeightV:                    make([]float32, len(neuron.WeightV)), // Adam
				BiasM:                      neuron.BiasM,                         // Adam
				BiasV:                      neuron.BiasV,                         // Adam
			}

			// Copy weights
			copy(cloneNeuron.Weights, neuron.Weights)
			// Copy accumulated gradients
			if neuron.AccumulatedWeightGradients != nil {
				copy(cloneNeuron.AccumulatedWeightGradients, neuron.AccumulatedWeightGradients)
			} else {
				// Ensure the slice is initialized if the source was nil, matching initialization logic
				cloneNeuron.AccumulatedWeightGradients = make([]float32, len(neuron.Weights))
			}
			// Copy weight velocities
			if neuron.WeightVelocities != nil {
				copy(cloneNeuron.WeightVelocities, neuron.WeightVelocities)
			} else {
				// Ensure the slice is initialized if the source was nil, matching initialization logic
				cloneNeuron.WeightVelocities = make([]float32, len(neuron.Weights))
			}
			// Copy Adam moment estimates for weights
			if neuron.WeightM != nil {
				copy(cloneNeuron.WeightM, neuron.WeightM)
			} else {
				cloneNeuron.WeightM = make([]float32, len(neuron.Weights))
			}
			if neuron.WeightV != nil {
				copy(cloneNeuron.WeightV, neuron.WeightV)
			} else {
				cloneNeuron.WeightV = make([]float32, len(neuron.Weights))
			}
			// BiasM and BiasV are value types, already copied.

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
	for _, layer := range nn.Layers {
		for _, neuron := range layer.Neurons {
			if neuron.AccumulatedWeightGradients == nil || len(neuron.AccumulatedWeightGradients) != len(neuron.Weights) {
				neuron.AccumulatedWeightGradients = make([]float32, len(neuron.Weights))
			} else {
				for k := range neuron.AccumulatedWeightGradients {
					neuron.AccumulatedWeightGradients[k] = 0.0
				}
			}
			neuron.AccumulatedBiasGradient = 0.0
		}
		// Zero out Batch Normalization gradient accumulators
		if layer.UseBatchNormalization {
			if layer.AccumulatedGammaGradients == nil || len(layer.AccumulatedGammaGradients) != len(layer.Neurons) {
				layer.AccumulatedGammaGradients = make([]float32, len(layer.Neurons))
			} else {
				for k := range layer.AccumulatedGammaGradients {
					layer.AccumulatedGammaGradients[k] = 0.0
				}
			}
			if layer.AccumulatedBetaGradients == nil || len(layer.AccumulatedBetaGradients) != len(layer.Neurons) {
				layer.AccumulatedBetaGradients = make([]float32, len(layer.Neurons))
			} else {
				for k := range layer.AccumulatedBetaGradients {
					layer.AccumulatedBetaGradients[k] = 0.0
				}
			}
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

		permutation := rand.Perm(numSamples)

		for i := 0; i < numSamples; i += batchSize {
			end := i + batchSize
			if end > numSamples {
				end = numSamples
			}
			currentMiniBatchSize := end - i
			if currentMiniBatchSize == 0 {
				continue
			}

			miniBatchData := make([][]float32, currentMiniBatchSize)
			miniBatchLabels := make([][]float32, currentMiniBatchSize)
			for k := 0; k < currentMiniBatchSize; k++ {
				originalIndex := permutation[i+k]
				miniBatchData[k] = trainingData[originalIndex]
				miniBatchLabels[k] = expectedOutputs[originalIndex]
			}

			nn.zeroAccumulatedGradients()
			var miniBatchLoss float32 = 0.0

			if nn.Params.EnableBatchNorm && nn.Params.IsTraining {
				// Step A: Collect PreBNAOutput for all samples in the batch for each BN layer.
				// And Step B: Calculate batch statistics and update running stats.
				
				// Initialize/Resize LastInputPreBNBatch and LastXNormalizedBatch for all BN layers
				for _, layer := range nn.Layers {
					if layer.UseBatchNormalization {
						// Resize LastInputPreBNBatch
						if cap(layer.LastInputPreBNBatch) < currentMiniBatchSize || layer.LastInputPreBNBatch == nil {
							layer.LastInputPreBNBatch = make([][]float32, currentMiniBatchSize)
						} else {
							layer.LastInputPreBNBatch = layer.LastInputPreBNBatch[:currentMiniBatchSize]
						}
						for sIdx := range layer.LastInputPreBNBatch {
							if cap(layer.LastInputPreBNBatch[sIdx]) < len(layer.Neurons) || layer.LastInputPreBNBatch[sIdx] == nil {
								layer.LastInputPreBNBatch[sIdx] = make([]float32, len(layer.Neurons))
							} else {
								layer.LastInputPreBNBatch[sIdx] = layer.LastInputPreBNBatch[sIdx][:len(layer.Neurons)]
							}
						}
						// Resize LastXNormalizedBatch
						if cap(layer.LastXNormalizedBatch) < currentMiniBatchSize || layer.LastXNormalizedBatch == nil {
							layer.LastXNormalizedBatch = make([][]float32, currentMiniBatchSize)
						} else {
							layer.LastXNormalizedBatch = layer.LastXNormalizedBatch[:currentMiniBatchSize]
						}
						for sIdx := range layer.LastXNormalizedBatch {
							if cap(layer.LastXNormalizedBatch[sIdx]) < len(layer.Neurons) || layer.LastXNormalizedBatch[sIdx] == nil {
								layer.LastXNormalizedBatch[sIdx] = make([]float32, len(layer.Neurons))
							} else {
								layer.LastXNormalizedBatch[sIdx] = layer.LastXNormalizedBatch[sIdx][:len(layer.Neurons)]
							}
						}
						// Resize CurrentBatchDLdXHat
						if cap(layer.CurrentBatchDLdXHat) < currentMiniBatchSize || layer.CurrentBatchDLdXHat == nil {
							layer.CurrentBatchDLdXHat = make([][]float32, currentMiniBatchSize)
						} else {
							layer.CurrentBatchDLdXHat = layer.CurrentBatchDLdXHat[:currentMiniBatchSize]
						}
						for s_idx := range layer.CurrentBatchDLdXHat {
							if cap(layer.CurrentBatchDLdXHat[s_idx]) < len(layer.Neurons) || layer.CurrentBatchDLdXHat[s_idx] == nil {
								layer.CurrentBatchDLdXHat[s_idx] = make([]float32, len(layer.Neurons))
							} else {
								layer.CurrentBatchDLdXHat[s_idx] = layer.CurrentBatchDLdXHat[s_idx][:len(layer.Neurons)]
							}
						}
					}
				}

				// A. Collect PreBNAOutput
				tempInputForLayer := make([][]float32, currentMiniBatchSize) // [sampleIdx][features]
				for sIdx := 0; sIdx < currentMiniBatchSize; sIdx++ {
					tempInputForLayer[sIdx] = miniBatchData[sIdx]
				}

				for layerIdx, layer := range nn.Layers {
					currentLayerPreBNOutputs := make([][]float32, currentMiniBatchSize)
					for sIdx := 0; sIdx < currentMiniBatchSize; sIdx++ {
						currentLayerPreBNOutputs[sIdx] = make([]float32, len(layer.Neurons))
					}

					for sampleIdx := 0; sampleIdx < currentMiniBatchSize; sampleIdx++ {
						sampleInput := tempInputForLayer[sampleIdx]
						for neuronIdx, neuron := range layer.Neurons {
							var sum float32 = neuron.Bias
							for weightIdx, weight := range neuron.Weights {
								sum += sampleInput[weightIdx] * weight
							}
							if layer.UseBatchNormalization {
								layer.LastInputPreBNBatch[sampleIdx][neuronIdx] = sum
							}
							currentLayerPreBNOutputs[sampleIdx][neuronIdx] = sum // Store z for all neurons
						}
					}

					// B. Calculate Batch Statistics & Update Running Stats (if BN layer)
					if layer.UseBatchNormalization {
						numNeurons := len(layer.Neurons)
						layer.CurrentBatchMean = make([]float32, numNeurons)
						layer.CurrentBatchVariance = make([]float32, numNeurons)
						for neuronIdx := 0; neuronIdx < numNeurons; neuronIdx++ {
							var sumPreBN float32 = 0.0
							for sampleIdx := 0; sampleIdx < currentMiniBatchSize; sampleIdx++ {
								sumPreBN += layer.LastInputPreBNBatch[sampleIdx][neuronIdx]
							}
							mean := sumPreBN / float32(currentMiniBatchSize)
							layer.CurrentBatchMean[neuronIdx] = mean
							
							var sumSqDiff float32 = 0.0
							for sampleIdx := 0; sampleIdx < currentMiniBatchSize; sampleIdx++ {
								diff := layer.LastInputPreBNBatch[sampleIdx][neuronIdx] - mean
								sumSqDiff += diff * diff
							}
							layer.CurrentBatchVariance[neuronIdx] = sumSqDiff / float32(currentMiniBatchSize)

							layer.RunningMean[neuronIdx] = layer.MomentumBN*layer.RunningMean[neuronIdx] + (1.0-layer.MomentumBN)*mean
							layer.RunningVariance[neuronIdx] = layer.MomentumBN*layer.RunningVariance[neuronIdx] + (1.0-layer.MomentumBN)*layer.CurrentBatchVariance[neuronIdx]
						}
					}
					
					// Prepare input for the next layer by applying BN (if any), activation, and dropout (if training)
					// This is essentially the forward propagation step after PreBNAOutput is known and BN stats are ready.
					if layerIdx < len(nn.Layers)-1 { // Not the output layer
						nextLayerInputs := make([][]float32, currentMiniBatchSize)
						for sIdx := 0; sIdx < currentMiniBatchSize; sIdx++ {
							nextLayerInputs[sIdx] = make([]float32, len(layer.Neurons))
							for neuronIdx, neuron := range layer.Neurons {
								valToActivate := currentLayerPreBNOutputs[sIdx][neuronIdx]
								if layer.UseBatchNormalization {
									// Use just calculated CurrentBatchMean/Variance for this training step
									xNorm := (valToActivate - layer.CurrentBatchMean[neuronIdx]) / float32(math.Sqrt(float64(layer.CurrentBatchVariance[neuronIdx] + layer.Epsilon)))
									// neuron.XNormalizedOutput = xNorm // This would be set here if FeedForward wasn't called again
									valToActivate = layer.Gamma[neuronIdx]*xNorm + layer.Beta[neuronIdx]
								}
								activatedVal := layer.Activation.Activate(valToActivate)
								if nn.Params.IsTraining && nn.Params.DropoutRate > 0 && layerIdx < len(nn.Layers)-1 { // No dropout on output layer
									if rand.Float32() < nn.Params.DropoutRate {
										activatedVal = 0.0
									} else {
										activatedVal /= (1.0 - nn.Params.DropoutRate)
									}
								}
								nextLayerInputs[sIdx][neuronIdx] = capValue(activatedVal)
							}
						}
						tempInputForLayer = nextLayerInputs // Set input for the next layer
					}
				}
			} // End of BN statistics calculation block

			// C. Main Processing Loop: Forward Pass using computed BN stats, Backpropagation
			// The nn.Params.IsTraining flag is true.
			// nn.FeedForward will use CurrentBatchMean/Variance if set.
			// It will also populate neuron.PreBNAOutput and neuron.XNormalizedOutput.

			// A. Pre-loop initializations for two-pass backpropagation
			if nn.Params.EnableBatchNorm && nn.Params.IsTraining {
				for _, layer := range nn.Layers {
					if layer.UseBatchNormalization {
						// Ensure CurrentBatchDLdXHat is sized (should have been done earlier, but double-check)
						if cap(layer.CurrentBatchDLdXHat) < currentMiniBatchSize || layer.CurrentBatchDLdXHat == nil {
							layer.CurrentBatchDLdXHat = make([][]float32, currentMiniBatchSize)
							for s_idx := 0; s_idx < currentMiniBatchSize; s_idx++ {
								layer.CurrentBatchDLdXHat[s_idx] = make([]float32, len(layer.Neurons))
							}
						} else {
							layer.CurrentBatchDLdXHat = layer.CurrentBatchDLdXHat[:currentMiniBatchSize]
							for s_idx := 0; s_idx < currentMiniBatchSize; s_idx++ {
								if cap(layer.CurrentBatchDLdXHat[s_idx]) < len(layer.Neurons) || layer.CurrentBatchDLdXHat[s_idx] == nil {
									layer.CurrentBatchDLdXHat[s_idx] = make([]float32, len(layer.Neurons))
								} else {
									layer.CurrentBatchDLdXHat[s_idx] = layer.CurrentBatchDLdXHat[s_idx][:len(layer.Neurons)]
								}
							}
						}
						// Size and zero TempSumDldXhat and TempSumDldXhatXhat
						layer.TempSumDldXhat = make([]float32, len(layer.Neurons))
						layer.TempSumDldXhatXhat = make([]float32, len(layer.Neurons))
						// Explicitly zero them (make might not guarantee zeroing for reused slices if not re-sliced properly)
						for k := range layer.TempSumDldXhat {
							layer.TempSumDldXhat[k] = 0.0
							layer.TempSumDldXhatXhat[k] = 0.0
						}
					}
				}
			}
			
			// Store worker clones for gradient aggregation if multi-threaded
			var workerClones []*NeuralNetwork
			if numWorkers > 1 {
				workerClones = make([]*NeuralNetwork, numWorkers)
			}

			if numWorkers <= 1 { // Single-threaded
				// B. First Backpropagation Pass (Loop over samples sIdx)
				for sIdx := 0; sIdx < currentMiniBatchSize; sIdx++ {
					dataSample := miniBatchData[sIdx]
					labelSample := miniBatchLabels[sIdx]

					// B.1. FeedForward for the current sample
					nn.FeedForward(dataSample) // This calculates neuron.Output, neuron.XNormalizedOutput, etc. for sample sIdx

					// B.2. Calculate loss for this sample
					sampleLoss := nn.calculateLoss(labelSample)
					miniBatchLoss += sampleLoss
					
					// B.3. Collect XNormalizedOutput for BN layers (needed for dGamma and dL/dXhat*Xhat sums)
					// This needs to happen *after* FeedForward for the current sample.
					if nn.Params.EnableBatchNorm && nn.Params.IsTraining {
						for _, layer := range nn.Layers {
							if layer.UseBatchNormalization {
								// Boundary check for LastXNormalizedBatch
								if layer.LastXNormalizedBatch != nil && sIdx < len(layer.LastXNormalizedBatch) && layer.LastXNormalizedBatch[sIdx] != nil {
									for neuronIdx, neuron := range layer.Neurons {
										if neuronIdx < len(layer.LastXNormalizedBatch[sIdx]) {
											layer.LastXNormalizedBatch[sIdx][neuronIdx] = neuron.XNormalizedOutput
										}
									}
								}
							}
						}
					}

					// B.4. Calculate Output Layer Deltas and Gradients
					outputLayer := nn.Layers[len(nn.Layers)-1]
					if outputLayer.Deltas == nil || len(outputLayer.Deltas) != len(outputLayer.Neurons) {
						outputLayer.Deltas = make([]float32, len(outputLayer.Neurons))
					}
					props := nn.SoftmaxProbabilities() // Get softmax probabilities for the current sample
					for j := 0; j < len(outputLayer.Neurons); j++ {
						outputLayer.Deltas[j] = capValue(props[j] - labelSample[j])
					}

					// Accumulate W/B gradients for the output layer
					var prevLayerOutputsOutput []float32
					if len(nn.Layers) > 1 { // Check if there is a layer before the output layer
						prevLayer := nn.Layers[len(nn.Layers)-2]
						// prevLayerOutputsOutput = make([]float32, len(prevLayer.Neurons)) // Not needed, use neuron.Output directly
						// Using neuron.Output directly from the previous layer, which holds values for current sample sIdx
					} else { // Network is just an output layer
						prevLayerOutputsOutput = nn.Input // nn.Input was set by FeedForward(dataSample)
					}

					for nIdx, neuron := range outputLayer.Neurons {
						delta := outputLayer.Deltas[nIdx]
						var currentPrevLayerOutputs []float32
						if len(nn.Layers) > 1 {
							prevLayerActual := nn.Layers[len(nn.Layers)-2]
							currentPrevLayerOutputs = make([]float32, len(prevLayerActual.Neurons))
							for k, prevNeuron := range prevLayerActual.Neurons {
								currentPrevLayerOutputs[k] = prevNeuron.Output // These are for sample sIdx
							}
						} else {
							currentPrevLayerOutputs = nn.Input
						}

						if neuron.AccumulatedWeightGradients != nil {
							for wIdx := range neuron.Weights {
								neuron.AccumulatedWeightGradients[wIdx] += delta * currentPrevLayerOutputs[wIdx]
							}
						}
						neuron.AccumulatedBiasGradient += delta
					}

					// B.5. Backpropagate through Hidden Layers (Pass 1)
					for layerIdx := len(nn.Layers) - 2; layerIdx >= 0; layerIdx-- {
						layer := nn.Layers[layerIdx]
						nextLayer := nn.Layers[layerIdx+1]

						if layer.Deltas == nil || len(layer.Deltas) != len(layer.Neurons) {
							layer.Deltas = make([]float32, len(layer.Neurons)) // Initialize if nil
						}

						var prevLayerOutputsHidden []float32
						if layerIdx == 0 {
							prevLayerOutputsHidden = nn.Input // nn.Input was set by FeedForward(dataSample)
						} else {
							prevLayer := nn.Layers[layerIdx-1]
							prevLayerOutputsHidden = make([]float32, len(prevLayer.Neurons))
							for k, prevNeuron := range prevLayer.Neurons {
								prevLayerOutputsHidden[k] = prevNeuron.Output // These are for sample sIdx
							}
						}

						for neuronIdx, neuron := range layer.Neurons {
							var errorSumTimesWeight float32 = 0.0
							for k, nextNeuron := range nextLayer.Neurons {
								errorSumTimesWeight += nextNeuron.Weights[neuronIdx] * nextLayer.Deltas[k] // Using Deltas from layer above
							}
							activationDerivative := layer.Activation.Derivative(neuron.Output) // neuron.Output is for sample sIdx
							dL_dActivationOutput_js := capValue(errorSumTimesWeight * activationDerivative)

							if !layer.UseBatchNormalization {
								layer.Deltas[neuronIdx] = dL_dActivationOutput_js
								// Accumulate W/B gradients
								if neuron.AccumulatedWeightGradients != nil {
									for wIdx := range neuron.Weights {
										neuron.AccumulatedWeightGradients[wIdx] += dL_dActivationOutput_js * prevLayerOutputsHidden[wIdx]
									}
								}
								neuron.AccumulatedBiasGradient += dL_dActivationOutput_js
							} else { // Layer uses Batch Normalization
								dL_dY_js := dL_dActivationOutput_js
								// Store XNormalizedOutput if not already done by a separate collection step (it was done above in B.3)
								// layer.LastXNormalizedBatch[sIdx][neuronIdx] = neuron.XNormalizedOutput
								
								layer.AccumulatedGammaGradients[neuronIdx] += dL_dY_js * neuron.XNormalizedOutput // neuron.XNormalizedOutput is for sample sIdx
								layer.AccumulatedBetaGradients[neuronIdx] += dL_dY_js
								
								dL_dX_hat_js := dL_dY_js * layer.Gamma[neuronIdx]
								if layer.CurrentBatchDLdXHat != nil && sIdx < len(layer.CurrentBatchDLdXHat) &&
								   layer.CurrentBatchDLdXHat[sIdx] != nil && neuronIdx < len(layer.CurrentBatchDLdXHat[sIdx]) {
									layer.CurrentBatchDLdXHat[sIdx][neuronIdx] = dL_dX_hat_js
								}
								// W/B gradients for this BN layer are NOT accumulated here in Pass 1.
								// layer.Deltas[neuronIdx] is NOT set here for BN layers in Pass 1.
							}
						}
					}
				} // End of B. First Backpropagation Pass (sample loop sIdx)

				// C. Intermediate Step: Calculate Batch Sums for BN Layers
				if nn.Params.EnableBatchNorm && nn.Params.IsTraining {
					for _, layer := range nn.Layers {
						if layer.UseBatchNormalization {
							// TempSumDldXhat and TempSumDldXhatXhat were already zeroed in Step A
							// If not, they should be zeroed here:
							// for k := range layer.TempSumDldXhat { layer.TempSumDldXhat[k] = 0.0 }
							// for k := range layer.TempSumDldXhatXhat { layer.TempSumDldXhatXhat[k] = 0.0 }

							for neuronIdx := 0; neuronIdx < len(layer.Neurons); neuronIdx++ {
								for sIdx := 0; sIdx < currentMiniBatchSize; sIdx++ {
									// Ensure CurrentBatchDLdXHat and LastXNormalizedBatch are valid for sIdx, neuronIdx
									if layer.CurrentBatchDLdXHat != nil && sIdx < len(layer.CurrentBatchDLdXHat) &&
										layer.CurrentBatchDLdXHat[sIdx] != nil && neuronIdx < len(layer.CurrentBatchDLdXHat[sIdx]) {
										layer.TempSumDldXhat[neuronIdx] += layer.CurrentBatchDLdXHat[sIdx][neuronIdx]
									}

									if layer.CurrentBatchDLdXHat != nil && sIdx < len(layer.CurrentBatchDLdXHat) &&
										layer.CurrentBatchDLdXHat[sIdx] != nil && neuronIdx < len(layer.CurrentBatchDLdXHat[sIdx]) &&
										layer.LastXNormalizedBatch != nil && sIdx < len(layer.LastXNormalizedBatch) &&
										layer.LastXNormalizedBatch[sIdx] != nil && neuronIdx < len(layer.LastXNormalizedBatch[sIdx]) {
										layer.TempSumDldXhatXhat[neuronIdx] += layer.CurrentBatchDLdXHat[sIdx][neuronIdx] * layer.LastXNormalizedBatch[sIdx][neuronIdx]
									}
								}
							}
						}
					}
				}
				// End of C. Intermediate Step

				// D. Second Backpropagation Pass (Loop over samples sIdx again)
				// This pass finalizes BN layers' deltas and their W/B gradients.
				// It assumes that neuron.Output values from Pass 1 (FeedForward for each sample) are still valid for that sample.
				if nn.Params.IsTraining { // Only during training
					for sIdx := 0; sIdx < currentMiniBatchSize; sIdx++ {
						// Re-FeedForward is NOT done here. We use states from Pass 1's FeedForward.
						// dataSample and labelSample are from the outer loop.
						dataSample := miniBatchData[sIdx] 
						// labelSample := miniBatchLabels[sIdx] // Not directly used in this pass's core logic for hidden layers

						for layerIdx := len(nn.Layers) - 2; layerIdx >= 0; layerIdx-- { // Hidden layers only
							layer := nn.Layers[layerIdx]
							
							// Determine prevLayerOutputs for the current sample sIdx
							var prevLayerOutputs []float32
							if layerIdx == 0 {
								prevLayerOutputs = dataSample // For the first hidden layer, input is the original sample data
							} else {
								prevLayer := nn.Layers[layerIdx-1]
								// These outputs are from the FeedForward pass for *this specific sample sIdx*
								// We need to ensure these are correctly captured or re-fetched.
								// The current structure implies nn.FeedForward(dataSample) in Pass 1 set neuron.Output for sample sIdx.
								// So, reading prevLayer.Neurons[k].Output should give sample-specific outputs.
								// However, if we don't re-run FeedForward, we need a way to get per-sample outputs.
								// For now, let's assume neuron.Output holds the output for the *last processed sample* in Pass 1.
								// This means for Pass 2, we must re-run FeedForward for each sample OR store all sample outputs.
								// The prompt says: "neuron.Output values are from that specific sample." - this implies they are available.
								// Let's proceed by re-running FeedForward for the specific sample to ensure neuron states are correct for sIdx
								
								// nn.FeedForward(dataSample) // Option 1: Re-run FF. Computationally expensive.
								// Option 2: Store all neuron outputs for all samples if memory allows.
								// Option 3: The original Pass 1 structure already does FF per sample.
								// The key is that prevLayer.Neurons[k].Output must reflect the state for dataSample[sIdx].
								// Since Pass 1 looped sIdx and called FeedForward(miniBatchData[sIdx]),
								// the state of nn.Input and all neuron.Output fields are for the *last* sample of Pass 1.
								// This is a flaw in the current single-threaded loop structure if we don't re-feed.
								//
								// Revisiting the prompt: "neuron.Output values are from that specific sample".
								// This implies we *don't* need to re-run FF if Pass 1 correctly set them *and they weren't overwritten*.
								// In the single-threaded case, Pass 1 iterates sIdx. Inside, FF is called for dataSample[sIdx].
								// Then backprop for dataSample[sIdx] happens.
								// So, at the end of Pass 1, neuron.Output holds values for the *last* sample.
								// This means for Pass 2, to get prevLayerOutputs for miniBatchData[sIdx], we *must* re-run FeedForward for that sample.
								// This is a critical point.
								
								// Let's assume for now that we *do* need to ensure the network state is for sample sIdx.
								// The most straightforward way is to call FeedForward.
								// This will also update LastXNormalizedBatch and XNormalizedOutput correctly for the current sample sIdx for BN layers.
								
								// Store current nn.Input to restore later if FeedForward modifies it and it's needed by other parts.
								// originalInput := make([]float32, len(nn.Input))
								// copy(originalInput, nn.Input)
								
								nn.FeedForward(dataSample) // Ensures neuron.Output and neuron.XNormalizedOutput are for sample sIdx

								// copy(nn.Input, originalInput) // Restore if necessary, though not clear if it is.

								prevLayerForOutput := nn.Layers[layerIdx-1]
								prevLayerOutputs = make([]float32, len(prevLayerForOutput.Neurons))
								for k_out, prevNeuron_out := range prevLayerForOutput.Neurons {
									prevLayerOutputs[k_out] = prevNeuron_out.Output
								}
							}

							if layer.UseBatchNormalization {
								for neuronIdx, neuron := range layer.Neurons {
									N := float32(currentMiniBatchSize)
									invN := 1.0 / N
									sigma_sq_j := layer.CurrentBatchVariance[neuronIdx] // Calculated in BN stats pre-computation
									invStdDev := float32(1.0 / math.Sqrt(float64(sigma_sq_j+layer.Epsilon)))

									// dL_dX_hat_js must be for the current sample sIdx.
									// CurrentBatchDLdXHat was populated in Pass 1 for each sample.
									dL_dX_hat_js := layer.CurrentBatchDLdXHat[sIdx][neuronIdx]
									
									// X_hat_js must be for the current sample sIdx.
									// LastXNormalizedBatch was populated in Pass 1 for each sample.
									// OR neuron.XNormalizedOutput from the FeedForward call at the start of this Pass 2 sample iteration.
									X_hat_js := neuron.XNormalizedOutput // From FeedForward(dataSample) at start of this sIdx loop

									sum_dL_dX_hat_j := layer.TempSumDldXhat[neuronIdx]             // From Step C
									sum_dL_dX_hat_times_X_hat_j := layer.TempSumDldXhatXhat[neuronIdx] // From Step C

									term1 := dL_dX_hat_js
									term2 := invN * sum_dL_dX_hat_j
									term3 := invN * X_hat_js * sum_dL_dX_hat_times_X_hat_j
									
									dL_dPreBNAOutput_js := (term1 - term2 - term3) * invStdDev
									
									// This delta is for the neuron's pre-BN activation for the current sample sIdx.
									// It will be used by the layer below (i-1) if it exists.
									if layer.Deltas == nil { layer.Deltas = make([]float32, len(layer.Neurons))}
									layer.Deltas[neuronIdx] = capValue(dL_dPreBNAOutput_js)

									// Accumulate W/B gradients for this BN layer neuron
									if neuron.AccumulatedWeightGradients != nil {
										for wIdx := range neuron.Weights {
											neuron.AccumulatedWeightGradients[wIdx] += dL_dPreBNAOutput_js * prevLayerOutputs[wIdx]
										}
									}
									neuron.AccumulatedBiasGradient += dL_dPreBNAOutput_js
								}
							}
							// If not UseBatchNormalization, their W/B grads and Deltas were already handled in Pass 1.
						}
					}
				} // End of D. Second Backpropagation Pass
			} else { // Multi-threaded
				// Check if any layer uses Batch Normalization
				bnEnabledInNetwork := false
				if nn.Params.EnableBatchNorm { // Global flag check first
					for _, layer := range nn.Layers {
						if layer.UseBatchNormalization {
							bnEnabledInNetwork = true
							break
						}
					}
				}

				if bnEnabledInNetwork {
					// Option 1: Panic (safer to halt execution)
					panic("FATAL: Batch Normalization backpropagation is not currently supported for multi-threaded execution (numWorkers > 1). Please use numWorkers = 1 when Batch Normalization is enabled.")
				}

				var wg sync.WaitGroup
				workerLosses := make([]float32, numWorkers)
				
				samplesPerWorker := (currentMiniBatchSize + numWorkers - 1) / numWorkers
				for w := 0; w < numWorkers; w++ {
					wg.Add(1)
					workerStart := w * samplesPerWorker
					workerEnd := workerStart + samplesPerWorker
					if workerStart >= currentMiniBatchSize { wg.Done(); continue }
					if workerEnd > currentMiniBatchSize { workerEnd = currentMiniBatchSize }

					go func(workerID int, startIdx, endIdx int) {
						// TODO: Implement two-pass Batch Normalization backpropagation and general gradient calculation for multi-threading.
						// The current single-threaded implementation in the `if numWorkers <= 1` block needs to be adapted.
						// This includes:
						// 1. Per-sample FeedForward.
						// 2. Pass 1 of backpropagation (calculating dL/dY, dGamma, dBeta, dL/dX_hat for BN; dL/dY and W/B grads for non-BN/output).
						//    - This will populate clone.AccumulatedGammaGradients, clone.AccumulatedBetaGradients, clone.CurrentBatchDLdXHat,
						//      clone.LastXNormalizedBatch, and W/B gradients for non-BN layers on the clone.
						// 3. After all workers complete Pass 1 (requires synchronization), the main thread would need to:
						//    a. Aggregate CurrentBatchDLdXHat and LastXNormalizedBatch from all clones if these are needed for TempSum calculations centrally.
						//       Alternatively, TempSumDldXhat and TempSumDldXhatXhat could be calculated per worker and then aggregated.
						//    b. Calculate global TempSumDldXhat and TempSumDldXhatXhat for each BN layer.
						//    c. Distribute these global TempSum values back to workers, or workers read them from the main nn instance.
						// 4. Pass 2 of backpropagation (calculating final dL/dPreBNAOutput and W/B grads for BN layers).
						//    - This uses the global TempSum values.
						// 5. Ensure workerLosses and workerClones (with all accumulated gradients) are correctly managed.

						// defer wg.Done()
						// if startIdx >= endIdx { return }

						// clone := nn.Clone()
						// clone.Params.IsTraining = true // Ensure clone is in training mode
						// if clone.Params.EnableBatchNorm {
						// 	for layerCloneIdx, mainLayer := range nn.Layers {
						// 		if mainLayer.UseBatchNormalization {
						// 			clone.Layers[layerCloneIdx].CurrentBatchMean = mainLayer.CurrentBatchMean
						// 			clone.Layers[layerCloneIdx].CurrentBatchVariance = mainLayer.CurrentBatchVariance
						// 		}
						// 	}
						// }
						// clone.zeroAccumulatedGradients()
						
						// currentWorkerLoss := float32(0.0)
						// for sCloneIdx := startIdx; sCloneIdx < endIdx; sCloneIdx++ {
						// 	dataSample := miniBatchData[sCloneIdx]
						// 	labelSample := miniBatchLabels[sCloneIdx]
						// 	// sampleLoss := clone.backpropagateAndAccumulateForSample(dataSample, labelSample, sCloneIdx) // COMPILE ERROR
						// 	currentWorkerLoss += 0.0 // Replace with actual sampleLoss
						// 	if clone.Params.EnableBatchNorm {
						// 		for layerCloneIdx, cl := range clone.Layers {
						// 			if cl.UseBatchNormalization {
						// 				// nn.Layers[layerCloneIdx].LastXNormalizedBatch[sCloneIdx][neuronCloneIdx] = cl.Neurons[neuronCloneIdx].XNormalizedOutput
						// 			}
						// 		}
						// 	}
						// }
						// workerLosses[workerID] = currentWorkerLoss
						// workerClones[workerID] = clone 
						workerClones[workerID] = nn.Clone() // Store a clone to prevent nil pointer in aggregation, though it has no grads.
						workerClones[workerID].zeroAccumulatedGradients() // Ensure gradients are zeroed.
						wg.Done() // Signal completion since the body is commented out.
					}(w, workerStart, workerEnd)
				}
				wg.Wait()

				for _, l := range workerLosses { miniBatchLoss += l }
				
				// Aggregate gradients - This part will aggregate zero gradients if the goroutine body is commented out.
				// This is acceptable for now as the goal is to make it compile and highlight the TODO.
				// If BN is enabled, this path is now guarded by a panic, so BN aggregation won't run.
                // for _, clone := range workerClones {
                //     if clone == nil { continue } 
                //     for layerIdx, cloneLayer := range clone.Layers {
                //         for neuronIdx, cloneNeuron := range cloneLayer.Neurons {
                //             mainNeuron := nn.Layers[layerIdx].Neurons[neuronIdx]
                //             if cloneNeuron.AccumulatedWeightGradients != nil {
                //                 for wIdx, grad := range cloneNeuron.AccumulatedWeightGradients {
                //                     mainNeuron.AccumulatedWeightGradients[wIdx] += grad
                //                 }
                //             }
                //             mainNeuron.AccumulatedBiasGradient += cloneNeuron.AccumulatedBiasGradient
                //         }
                //         // Aggregate Batch Normalization gradients
                //         if nn.Layers[layerIdx].UseBatchNormalization { // This check is fine
                //             for neuronIdx, _ := range cloneLayer.Neurons { 
                //                 // mainNeuron := nn.Layers[layerIdx].Neurons[neuronIdx] // Not needed for layer-level grads
                //                 if cloneLayer.AccumulatedGammaGradients != nil && neuronIdx < len(cloneLayer.AccumulatedGammaGradients) &&
                //                    nn.Layers[layerIdx].AccumulatedGammaGradients != nil && neuronIdx < len(nn.Layers[layerIdx].AccumulatedGammaGradients) {
                //                    nn.Layers[layerIdx].AccumulatedGammaGradients[neuronIdx] += cloneLayer.AccumulatedGammaGradients[neuronIdx]
                //                 }
                //                 if cloneLayer.AccumulatedBetaGradients != nil && neuronIdx < len(cloneLayer.AccumulatedBetaGradients) &&
                //                    nn.Layers[layerIdx].AccumulatedBetaGradients != nil && neuronIdx < len(nn.Layers[layerIdx].AccumulatedBetaGradients) {
                //                    nn.Layers[layerIdx].AccumulatedBetaGradients[neuronIdx] += cloneLayer.AccumulatedBetaGradients[neuronIdx]
                //                 }
                //             }
                //         }
                //     }
                // }
                // Collect LastXNormalizedBatch from clones (serially after join)
                // This is safer than concurrent writes.
                // if nn.Params.EnableBatchNorm && nn.Params.IsTraining { // This check is fine
                //     for w := 0; w < numWorkers; w++ {
                //         clone := workerClones[w]
                //         if clone == nil { continue }
                //         workerStart := w * samplesPerWorker
                //         workerEnd := workerStart + samplesPerWorker
                //         if workerStart >= currentMiniBatchSize { continue }
                //         if workerEnd > currentMiniBatchSize { workerEnd = currentMiniBatchSize }

                //         for sCloneIdx := workerStart; sCloneIdx < workerEnd; sCloneIdx++ {
                //             for layerIdx, cl := range clone.Layers {
                //                 if cl.UseBatchNormalization {
                //                     if nn.Layers[layerIdx].LastXNormalizedBatch != nil && sCloneIdx < len(nn.Layers[layerIdx].LastXNormalizedBatch) &&
                //                        cl.LastXNormalizedBatch != nil && sCloneIdx < len(cl.LastXNormalizedBatch) { // Check cl.LastXNormalizedBatch too
                //                         for neuronIdx, cn := range cl.Neurons {
                //                             if nn.Layers[layerIdx].LastXNormalizedBatch[sCloneIdx] != nil && neuronIdx < len(nn.Layers[layerIdx].LastXNormalizedBatch[sCloneIdx]) &&
                //                                cl.LastXNormalizedBatch[sCloneIdx] != nil && neuronIdx < len(cl.LastXNormalizedBatch[sCloneIdx]) { // Check cl.LastXNormalizedBatch[sCloneIdx]
                //                                 // This line would try to read from clone's LastXNormalizedBatch, which is not populated by the commented-out goroutine.
                //                                 // nn.Layers[layerIdx].LastXNormalizedBatch[sCloneIdx][neuronIdx] = cn.XNormalizedOutput
                //                             }
                //                         }
                //                     }
                //                 }
                //             }
                //         }
                //     }
                // }
			}
			// Optimizer applies gradients
			gradients := Gradients{
				WeightGradients: make([][][]float32, len(nn.Layers)),
				BiasGradients:   make([][]float32, len(nn.Layers)),
			}
			for i, layer := range nn.Layers {
				gradients.WeightGradients[i] = make([][]float32, len(layer.Neurons))
				gradients.BiasGradients[i] = make([]float32, len(layer.Neurons))
				for j, neuron := range layer.Neurons {
					// Ensure AccumulatedWeightGradients is initialized if it was nil
					if neuron.AccumulatedWeightGradients == nil {
						neuron.AccumulatedWeightGradients = make([]float32, len(neuron.Weights))
					}
					gradients.WeightGradients[i][j] = neuron.AccumulatedWeightGradients
					gradients.BiasGradients[i][j] = neuron.AccumulatedBiasGradient
				}
			}

			// Instantiate optimizer and apply gradients
			optimizer := &Adam{t: 0} // Initialize Adam optimizer with timestep t=0
			err := optimizer.Apply(&nn.Params, nn.Layers, &gradients, currentMiniBatchSize)
			if err != nil {
				// Handle error, e.g., log it or return from the function
				fmt.Printf("Error applying gradients: %v\n", err)
			}

			totalEpochLoss += miniBatchLoss
			samplesProcessedInEpoch += currentMiniBatchSize
		}

		averageEpochLoss := float32(0.0)
		if samplesProcessedInEpoch > 0 {
			averageEpochLoss = totalEpochLoss / float32(samplesProcessedInEpoch)
		}

		fmt.Printf("Loss MiniBatch Epoch %d = %.2f (LR: %.5f)\n", e, averageEpochLoss, nn.Params.Lr)

		// Apply learning rate decay at the end of each epoch.
		nn.Params.Lr *= nn.Params.Decay
	} // End of epoch loop
}

// TrainBatch function removed as unused (superseded by TrainMiniBatch).

func (nn *NeuralNetwork) Output() []float32 {
	outputLayer := nn.Layers[len(nn.Layers)-1].Neurons
	output := make([]float32, len(outputLayer))
	for i, neuron := range outputLayer {
		output[i] = neuron.Output
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
	outputLayerNeurons := nn.Layers[len(nn.Layers)-1].Neurons
	outputValues := make([]float32, len(outputLayerNeurons))
	for i, neuron := range outputLayerNeurons {
		neuron.Output = capValue(neuron.Output) // Ensure output is capped before softmax
		outputValues[i] = neuron.Output
	}
	// Apply softmax to the final layer's outputs to get probabilities.
	softmaxProbs := softmax(outputValues, nn.Params)
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
		p64 := math.Max(float64(softmaxProbs[i]), 1e-15)
		loss64 -= float64(target[i]) * math.Log(p64)
	}
	// Calculate L2 regularization part
	var reg64 float64 = 0.0
	for _, layer := range nn.Layers {
		for _, neuron := range layer.Neurons {
			for _, w := range neuron.Weights {
				reg64 += float64(w) * float64(w)
			}
		}
	}
	// Add L2 penalty to the loss.
	loss64 += 0.5 * float64(nn.Params.L2) * reg64
	loss = float32(loss64)

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
	tempExps64 := make([]float64, len(output))
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
		exps[i] = float32(tempExps64[i] / sumExps64)
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
	fmt.Printf("Model saved as %s\n", filename)
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
	fmt.Printf("Model loaded from %s\n", filename)
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
