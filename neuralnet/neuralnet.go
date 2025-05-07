package neuralnet

import (
        "fmt"
	"strings"
        "math"
        "time"
        "math/rand"
        "gonum.org/v1/gonum/mat"
        "encoding/json"
        "os"
)

// DefaultMaxAbsValue defines a large finite number to cap extreme values, preventing Inf propagation.
const DefaultMaxAbsValue = float32(1e10)

const MAX_WORKERS int = 8

type Neuron struct {
        weights []float32
        bias    float32
        output  float32
        momentum []float32
        // Fields for accumulating gradients in batch training
        accumulatedWeightGradients []float32
        accumulatedBiasGradient    float32
}

type Layer struct {
        neurons []*Neuron
        deltas  []float32
        activation ActivationFunction
}

// Represents of the simlest NN.
type NeuralNetwork struct {
        layers  []*Layer
        input   []float32
        params  Params
}

type Params struct {
        lr       float32
        decay    float32
        L2       float32
        lowCap   float32
        relu     float32
        jacobian bool
        bn float32
}

type Task struct {
        data mat.VecDense
        output mat.VecDense
}

func CreateTask(data mat.VecDense, output mat.VecDense) Task {
        return Task{
                data: data,
                output: output,
        }
}

func NewParams(learningRate float32, decay float32, regularization float32, cap float32) Params {
        return NewParamsFull(learningRate, decay, regularization, cap, defaultParams().relu, defaultParams().jacobian)
}

func NewParamsFull(learningRate float32, decay float32, regularization float32, cap float32, relu float32, jacobian bool) Params {
        return Params{
                lr:      learningRate,
                decay:   decay,
                L2:      regularization,
                lowCap:  cap,
                relu:    relu,
                jacobian: jacobian,
                bn: 0.0,
            }
}

func defaultParams() *Params {
        return &Params{
            lr:      0.01,
            decay:   0.95, // Reduced decay rate
            L2:      1e-4, // Enabled L2 regularization
            lowCap:  0,
            relu:    0,
            jacobian:false,
            bn: 0.0,
        }
}

func DefaultNeuralNetwork(inputSize int, hidden []int, outputSize int) *NeuralNetwork {
        params := defaultParams()
        return initialise(inputSize, hidden, outputSize, *params)
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
    }

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
            weights:  make([]float32, prevLayerNeuronCount),
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
        saveInput := make([]float32, input.Len())
        // the first layer takes inputs as X
        for _, neuron := range nn.layers[0].neurons {
                neuron.output = neuron.bias
                for j := 0; j < input.Len(); j++ {
                        currentInput := float32(input.AtVec(j))
                        saveInput[j] = currentInput
                        neuron.output += currentInput * neuron.weights[j]
                }
                neuron.output = nn.layers[0].activation.Activate(neuron.output)
                neuron.output = capValue(neuron.output, nn.params)
        }
        nn.input = saveInput
        for i := 1; i < len(nn.layers); i++ {
                for _, neuron := range nn.layers[i].neurons {
                        neuron.output = neuron.bias
                        for j := 0; j < len(nn.layers[i - 1].neurons); j++ {
                                neuron.output += nn.layers[i - 1].neurons[j].output * neuron.weights[j]
                        }
                        neuron.output = nn.layers[i].activation.Activate(neuron.output)
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
                    avgGrad := neuron.accumulatedWeightGradients[wIdx] / fBatchSize
                        
                    // Add L2 regularization gradient component
                    avgGrad += nn.params.L2 * neuron.weights[wIdx]
                        
                    neuron.momentum[wIdx] = 0.9*neuron.momentum[wIdx] + learningRate*avgGrad
                    neuron.weights[wIdx] -= neuron.momentum[wIdx]
                    neuron.weights[wIdx] = capValue(neuron.weights[wIdx], nn.params)
                }
            }
                
            // Update bias
            avgBiasGrad := neuron.accumulatedBiasGradient / fBatchSize
            neuron.bias -= learningRate * avgBiasGrad
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
    
// Clone a neural network for thread-safe parallel processing
func cloneNeuralNetwork(original *NeuralNetwork) *NeuralNetwork {
        // Create a new neural network with the same structure
        clone := &NeuralNetwork{
                layers: make([]*Layer, len(original.layers)),
                params: original.params,
        }
        
        // Deep copy all layers
        for i, layer := range original.layers {
                cloneLayer := &Layer{
                        neurons: make([]*Neuron, len(layer.neurons)),
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
        
        // Copy input and loss if they exist
        if original.input != nil {
                clone.input = make([]float32, len(original.input))
                copy(clone.input, original.input)
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
 Original implementation with potential race conditions.
 TODO: TrainMiniBatchOriginal is marked stub and not currently used.
*/
// Note: This function, even with the loss variable fix below, has inherent race conditions
// if used concurrently because nn.FeedForward and nn.AccumulateLoss modify the shared nn object's
// internal state from multiple goroutines without proper synchronization for those shared fields.
// For safe concurrent training, TrainMiniBatchThreadSafe (which uses cloning) is recommended.
func (nn *NeuralNetwork) TrainMiniBatchOriginal(trainingData []mat.VecDense, expectedOutputs []mat.VecDense, batchRatio int, epochs int) {
        panic("TrainMiniBatchOriginal not implemented")
}

/*
 Thread-safe implementation using worker clones.
 TODO: TrainMiniBatchThreadSafe is marked stub and not currently used.
*/
func (nn *NeuralNetwork) TrainMiniBatchThreadSafe(trainingData []mat.VecDense, expectedOutputs []mat.VecDense, batchRatio int, epochs int) {
        panic("TrainMiniBatchThreadSafe not implemented")
}

/*
 TrainMiniBatch wrapper is currently stubbed pending refactor.
 Direct use of minibatch training is disabled.
*/
func (nn *NeuralNetwork) TrainMiniBatch(trainingData []mat.VecDense, expectedOutputs []mat.VecDense, batchRatio int, epochs int) {
        panic("TrainMiniBatch is not implemented; pending refactor")
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
    for j := 0; j < len(outputLayer.neurons); j++ {
        // For a single sample, the delta is the error component (softmax_output - target_j).
        outputLayer.deltas[j] = capValue(float32(errVec.AtVec(j)), nn.params)
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
            derivative := layer.activation.Derivative(neuron.output) // neuron.output is from current sample's FeedForward
            layer.deltas[j] = capValue(errorSumTimesWeight * derivative, nn.params)
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
            prevLayerOutputs = make([]float32, len(prevLayer.neurons))
            for pIdx, pNeuron := range prevLayer.neurons {
                prevLayerOutputs[pIdx] = pNeuron.output // Activations from the previous layer
            }
        }

        for nIdx, neuron := range layer.neurons { // For neuron 'nIdx' in current 'layer'
            sampleDelta := layer.deltas[nIdx] // Delta for this neuron, for this sample
            
            // Accumulate weight gradients: gradient_w = delta_current_neuron * output_prev_layer_neuron
            if neuron.accumulatedWeightGradients != nil { // Should be initialized by zeroAccumulatedGradients
                for wIdx := range neuron.weights {
                    gradContribution := sampleDelta * prevLayerOutputs[wIdx]
                    neuron.accumulatedWeightGradients[wIdx] += gradContribution
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
        outputLayer := nn.layers[len(nn.layers) - 1].neurons
        output := make([]float32, len(outputLayer))
        for i, neuron := range outputLayer {
                output[i] = neuron.output
        }
        return output
}


func (nn *NeuralNetwork) Predict(data mat.VecDense) int {
        nn.FeedForward(data)
        props := nn.calculateProps()
        maxVal := props.AtVec(0)
        idx := 0
        for i := 1; i < props.Len(); i++ {
                val := props.AtVec(i)
                if val > maxVal {
                        maxVal = val
                        idx = i
                }
        }
        return idx
}

func (nn *NeuralNetwork) calculateProps() *mat.VecDense {
        outputLayer := nn.layers[len(nn.layers) - 1].neurons
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
        for i := 0; i < props.Len(); i++ {
                p := float32(math.Max(props.AtVec(i), 1e-15))
                // Cross-entropy: -sum target * log(p)
                loss -= float32(target.AtVec(i)) * float32(math.Log(float64(p)))
        }
        // L2 regularization: (lambda/2) * sum weights^2
        var reg float32 = 0.0
        for _, layer := range nn.layers {
                for _, neuron := range layer.neurons {
                        for _, w := range neuron.weights {
                                reg += w * w
                        }
                }
        }
        loss += 0.5 * nn.params.L2 * reg
        // Guard against NaN or Inf
        if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
                return 0.0
        }
        return loss
}

func convertWeightsDense(neurons []*Neuron) *mat.Dense {
        n := len(neurons)
        m := len(neurons[0].weights)
	weights := make([]float64, n * m)

        for j, _ := range neurons[0].weights {
                for i, neuron := range neurons {
                        weights[i * m + j] = float64(neuron.weights[j])
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

// The old Backpropagate function has been replaced by the logic within
// backpropagateAndAccumulateForSample and applyAveragedGradients.

// convertDeltasToDense was used by the old Backpropagate.
// It might not be needed with the new per-sample delta handling.
// Kept for now, can be removed if confirmed unused.
func convertDeltasToDense(layer *Layer, params Params) *mat.VecDense {
        dense := mat.NewVecDense(len(layer.deltas), nil)
        for i, d := range layer.deltas {
                if math.IsNaN(float64(d)) {
                        dense.SetVec(i, float64(params.lowCap))
                } else {
                        dense.SetVec(i, float64(capValue(d, params)))
                }
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
        limit := math.Sqrt(6.0 / float64(numInputs + numOutputs))
        xavier := float32(2 * rand.Float64() * limit - limit)
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

// Note: The import block below was duplicated and misplaced.
// It's generally better to have all imports at the top of the file.
// For this change, I am only removing the duplicated block.
// The necessary imports (json, os) should already be part of the main import block at the top.
// If not, they would need to be added there.
// Assuming "encoding/json" and "os" are already in the top import block.

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
func loadModel(filename string) {
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
