package neuralnet

import (
        "fmt"
	"strings"
        "math"
        "sync"
        "time"
        "math/rand"
        "gonum.org/v1/gonum/diff/fd"
        "gonum.org/v1/gonum/mat"
        "encoding/json"
        "os"
)

const MAX_WORKERS int = 8

type Neuron struct {
        weights []float32
        bias    float32
        output  float32
        momentum []float32
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
            lr:      0.001,
            decay:   0.8,
            L2:      0,
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

func initialise(inputSize int, hidden []int, outputSize int, params Params) *NeuralNetwork {
        rand.Seed(time.Now().UnixNano())

        if len(hidden) == 0 {
                // The current code structure relies on hidden[0] existing for the first layer's size
                // and subsequent loop iterations. If hidden can be empty, this needs specific handling
                // (e.g., a network with no hidden layers, or a different loop structure).
                panic("initialise: hidden layers slice cannot be empty as hidden[0] is accessed")
        }

        nn := &NeuralNetwork{
                // input + hidden + output
                layers: make([]*Layer, len(hidden) + 2),
                params: params,
        }

        first := &Layer{
                neurons: make([]*Neuron, hidden[0]),
                activation: Linear{},
        }
        for j := range first.neurons {
                first.neurons[j] = &Neuron{
                        weights:  make([]float32, inputSize),
                        bias:     0,
                        momentum: make([]float32, inputSize),
                }
                for k := range first.neurons[j].weights {
                        first.neurons[j].weights[k] = xavierInit(inputSize, hidden[0], nn.params)
                }
        }
        nn.layers[0] = first

        for i, size := range hidden {
                layer := &Layer{
                        neurons: make([]*Neuron, size),
                        activation: ReLU{},
                }
                out := 0
                if i < len(hidden) - 1 {
                        out = hidden[i + 1]
                } else {
                        out = outputSize
                }
                for j := range layer.neurons {
                        layer.neurons[j] = &Neuron{
                                weights: make([]float32, len(nn.layers[i].neurons)),
                                bias:    xavierInit(size, out, nn.params),
                        }
                        // TODO: use gauss to init weights
                        for k := range layer.neurons[j].weights {
                                layer.neurons[j].weights[k] =  xavierInit(size, out, nn.params)
                        }
                        layer.neurons[j].momentum = make([]float32, len(nn.layers[i].neurons))

                }
                nn.layers[i + 1] = layer
        }
        output := &Layer{
                neurons: make([]*Neuron, outputSize),
                activation: NewLeakyReLU(0.1),
        }
        for l := 0; l < outputSize; l++ {
                output.neurons[l] = &Neuron{
                        weights: make([]float32, len(nn.layers[len(nn.layers) - 2].neurons)),
                        bias:    xavierInit(outputSize, outputSize, nn.params),
                }
                // TODO: use gauss to init weights
                for k := range output.neurons[l].weights {
                        output.neurons[l].weights[k] = xavierInit(outputSize, outputSize, nn.params)
                }
                output.neurons[l].momentum = make([]float32, len(nn.layers[len(nn.layers) - 2].neurons))
        }
        nn.layers[len(hidden) + 1] = output

        nn.loss = make([]float32, outputSize)
        for i := 0; i < outputSize; i++ {
                nn.loss[i] = 0
        }

        return nn
}

func NewNeuralNetwork(inputSize int, hidden []int, outputSize int, params Params) *NeuralNetwork {
        return initialise(inputSize, hidden, outputSize, params)
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



func (nn *NeuralNetwork) UpdateWeights(learningRate float32) {
        // 1st layer update
        for j, neuron := range nn.layers[0].neurons {
                for k := range neuron.weights {
                        grad := nn.layers[0].deltas[j] * nn.input[k]
                        neuron.momentum[k] = 0.9 * neuron.momentum[k] + learningRate * grad
                        neuron.weights[k] -= neuron.momentum[k]
                        // regulisation term
                        neuron.weights[k] -= nn.params.L2 * neuron.weights[k]
                        neuron.weights[k] = capValue(neuron.weights[k], nn.params)
                    }
        
                    // Update bias: bias_new = bias_old + learningRate * delta
                    neuron.bias -= learningRate * nn.layers[0].deltas[j] // Bias momentum can also be added if desired
        }
        
        for i := 1; i < len(nn.layers); i++ {
            currentLayer := nn.layers[i]
            previousLayer := nn.layers[i-1]

            for j, neuron := range currentLayer.neurons {
                for k := range neuron.weights {
                    grad := currentLayer.deltas[j] * previousLayer.neurons[k].output
                    neuron.momentum[k] = 0.9 * neuron.momentum[k] + learningRate * grad
                    neuron.weights[k] -= neuron.momentum[k]
                    // regulisation term
                    neuron.weights[k] -= nn.params.L2 * neuron.weights[k]
                    neuron.weights[k] = capValue(neuron.weights[k], nn.params)
                }
    
                // Update bias: bias_new = bias_old + learningRate * delta
                neuron.bias -= learningRate * currentLayer.deltas[j] // Bias momentum can also be added if desired
            }
        }
}

func (nn *NeuralNetwork) TrainSGD(trainingData []mat.VecDense, expectedOutputs []mat.VecDense,  epochs int) {
        // Standard SGD processes one sample at a time, so batchSize is effectively 1.
        // The variable itself was unused.
        for e := 0; e < epochs; e++ {
                var totalEpochLoss float32 = 0.0
                // Create a permutation of indices to shuffle the training data for each epoch
                permutation := rand.Perm(len(trainingData))

                for _, idx := range permutation {
                        dataSample := trainingData[idx]
                        labelSample := expectedOutputs[idx]
                        
                        nn.FeedForward(dataSample)
                        nn.AccumulateLoss(labelSample) // Accumulates loss for backprop
                        totalEpochLoss += nn.calculateLoss(labelSample) // Sums individual sample losses for reporting
                        nn.Backpropagate(1) // Backpropagate and update weights for this single sample
                }
                
                // Apply learning rate decay once per epoch
                nn.params.lr *= nn.params.decay 
                
                if len(trainingData) > 0 {
                    fmt.Println(fmt.Sprintf("Loss SGD %d = %.2f", e, totalEpochLoss / float32(len(trainingData))))
                } else {
                    fmt.Println(fmt.Sprintf("Loss SGD %d = %.2f (No training data)", e, 0.0))
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
        
        if original.loss != nil {
                clone.loss = make([]float32, len(original.loss))
                copy(clone.loss, original.loss)
        }
        
        return clone
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
        for e := 0; e < epochs; e++ {
                var loss float32 = 0.0
                var lossMutex sync.Mutex
                start := time.Now()
                for i := 0; i < batchRatio; i++ {
                        currentData := trainingData[i : int((i + 1) * len(trainingData) / batchRatio)]
                        currentOutput := expectedOutputs[i : int((i + 1) * len(expectedOutputs) / batchRatio)]
                        size := len(currentOutput)
                        var tasks []Task
                        for j := 0; j < size; j++ {
                                tasks = append(tasks, CreateTask(currentData[j], currentOutput[j]))
                        }
                        var wg sync.WaitGroup
                        var perWorker [][]Task
                        for w := 0; w < MAX_WORKERS; w++ {
                                min := (w * size / MAX_WORKERS)
                                max := ((w + 1) * size) / MAX_WORKERS
                                if min < size && max <= size && min < max { // Ensure slice indices are valid
                                        perWorker = append(perWorker, tasks[min:max])
                                } else if min < size && max > size && min < max { // Handle last worker potentially getting a smaller slice
                                        perWorker = append(perWorker, tasks[min:size])
                                }
                        }
                        if len(perWorker) == 0 && size > 0 { // Fallback if no workers got tasks but tasks exist (e.g. size < MAX_WORKERS)
                            perWorker = append(perWorker, tasks)
                        }
                        for _, task := range(perWorker) {
                                wg.Add(1)
                                go func(work []Task) {
                                        defer wg.Done()
                                        for _, w := range(work) {
                                                nn.FeedForward(w.data)
                                                nn.AccumulateLoss(w.output)
                                                
                                                // Safely update shared loss
                                                batchItemLoss := nn.calculateLoss(w.output)
                                                lossMutex.Lock()
                                                loss += batchItemLoss
                                                lossMutex.Unlock()
                                        }
                                }(task)
                        }
                        wg.Wait()
                        nn.Backpropagate(len(currentData))
                        // Learning rate decay was here, moved to end of epoch
                }
                // Apply learning rate decay once per epoch
                nn.params.lr = nn.params.lr * nn.params.decay
                fmt.Printf("Time elapsed: %s\n", time.Since(start))
                fmt.Println(fmt.Sprintf("Loss MB %d = %.2f", e, loss / float32(len(trainingData))))
        }
}

/*
 Thread-safe implementation using worker clones.
 TODO: TrainMiniBatchThreadSafe is marked stub and not currently used.
*/
func (nn *NeuralNetwork) TrainMiniBatchThreadSafe(trainingData []mat.VecDense, expectedOutputs []mat.VecDense, batchRatio int, epochs int) {
        for e := 0; e < epochs; e++ {
                var totalLoss float32 = 0.0
                var totalLossMutex sync.Mutex
                start := time.Now()
                
                for i := 0; i < batchRatio; i++ {
                        currentData := trainingData[i : int((i + 1) * len(trainingData) / batchRatio)]
                        currentOutput := expectedOutputs[i : int((i + 1) * len(expectedOutputs) / batchRatio)]
                        size := len(currentOutput)
                        
                        // Create tasks
                        var tasks []Task
                        for j := 0; j < size; j++ {
                                tasks = append(tasks, CreateTask(currentData[j], currentOutput[j]))
                        }
                        
                        // Distribute tasks among workers
                        var perWorker [][]Task
                        for w := 0; w < MAX_WORKERS; w++ {
                                min := (w * size / MAX_WORKERS)
                                max := ((w + 1) * size) / MAX_WORKERS
                                if min < size && max <= size && min < max {
                                        perWorker = append(perWorker, tasks[min:max])
                                }
                        }
                        
                        // Create per-worker neural network clones to avoid race conditions
                        var workerNNs []*NeuralNetwork
                        for w := 0; w < len(perWorker); w++ {
                                // Create a clone of the neural network for each worker
                                workerNN := cloneNeuralNetwork(nn)
                                workerNNs = append(workerNNs, workerNN)
                        }
                        
                        var wg sync.WaitGroup
                        for w, task := range perWorker {
                                wg.Add(1)
                                go func(work []Task, workerNN *NeuralNetwork, workerId int) {
                                        defer wg.Done()
                                        var localLoss float32 = 0.0
                                        
                                        for _, t := range work {
                                                workerNN.FeedForward(t.data)
                                                workerNN.AccumulateLoss(t.output)
                                                localLoss += workerNN.calculateLoss(t.output)
                                        }
                                        
                                        // Safely update the shared loss
                                        totalLossMutex.Lock()
                                        totalLoss += localLoss
                                        totalLossMutex.Unlock()
                                }(task, workerNNs[w], w)
                        }
                        wg.Wait()
                        
                        // Reset main nn.loss before accumulating from workers
                        for k := range nn.loss {
                            nn.loss[k] = 0
                        }

                        // Accumulate losses from workerNNs to main nn.loss
                        // Also, ensure workerNN.loss is reset for its next use if AccumulateLoss doesn't reset it.
                        // The current AccumulateLoss sums, so workerNN.loss would grow indefinitely if not reset.
                        // For simplicity here, we assume workerNN.loss contains the loss for its tasks for this batch.
                        for _, workerNN := range workerNNs {
                            for k := 0; k < len(nn.loss); k++ {
                                if k < len(workerNN.loss) { // safety check
                                     nn.loss[k] += workerNN.loss[k]
                                     // Optionally, reset workerNN.loss[k] = 0 here if it's reused directly
                                }
                            }
                        }
                        
                        // Set main nn.input from a worker to avoid empty nn.input
                        if len(workerNNs) > 0 {
                                nn.input = workerNNs[0].input
                        }
                        // Do single backpropagation with accumulated loss from all workers
                        nn.Backpropagate(len(currentData))
                        // Learning rate decay was here, moved to end of epoch
                }
                
                // Apply learning rate decay once per epoch
                nn.params.lr = nn.params.lr * nn.params.decay
                fmt.Printf("Time elapsed: %s\n", time.Since(start))
                fmt.Println(fmt.Sprintf("Loss MB %d = %.2f", e, totalLoss / float32(len(trainingData))))
        }
}

/*
 TrainMiniBatch wrapper is currently stubbed pending refactor.
 Direct use of minibatch training is disabled.
*/
func (nn *NeuralNetwork) TrainMiniBatch(trainingData []mat.VecDense, expectedOutputs []mat.VecDense, batchRatio int, epochs int) {
        panic("TrainMiniBatch is not implemented; pending refactor")
}

func (nn *NeuralNetwork) TrainBatch(trainingData []mat.VecDense, expectedOutputs []mat.VecDense,  epochs int) {
        for e := 0; e < epochs; e++ {
                nn.params.lr = nn.params.lr * nn.params.decay
                var loss float32 = 0.0
                for i := 0; i < len(trainingData); i++ {
                        nn.FeedForward(trainingData[i])
                        nn.AccumulateLoss(expectedOutputs[i])
                        loss += nn.calculateLoss(expectedOutputs[i])
                }
                nn.Backpropagate(len(trainingData))
                fmt.Println(fmt.Sprintf("Loss Batch %d = %.2f", e, loss / float32(len(trainingData))))
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
        max := props.AtVec(0)
        idx := 0
        for i := 0; i < props.Len(); i++ {
                if max < props.AtVec(i) {
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

func (nn *NeuralNetwork) Backpropagate(errors []mat.VecDense) {
        outputLayer := nn.layers[len(nn.layers) - 1]
        outputLayer.deltas = make([]float32, len(outputLayer.neurons))

        // average (softmax - target) across the batch
        batchSize := float32(len(errors))
        for j := 0; j < len(outputLayer.neurons); j++ {
                var sumErr float32
                for _, errVec := range errors {
                        sumErr += float32(errVec.AtVec(j))
                }
                outputLayer.deltas[j] = capValue(sumErr/batchSize, nn.params)
        }

        for i := len(nn.layers) - 2; i >= 0; i-- {
                // M neurons neurons to calculate deltas for    
                layer := nn.layers[i]
                m := len(layer.neurons)
                // N neurons to propagate error from
                nextLayer := nn.layers[i + 1]
                n := len(nextLayer.neurons)

                // N * M
                weights := convertWeightsDense(nextLayer.neurons)

                input := make([]float64, m)
                for i, neuron := range layer.neurons {
                        input[i] = float64(neuron.output)
                }

                jac := mat.NewDense(n, m, nil)
                if (nn.params.jacobian) {
                        // N * 1
                        bias := convertBiasToDense(nextLayer.neurons)
                        // jac is delta(ActicatonN)/delta(OutputM)
                        fd.Jacobian(jac,
                                func(next, prev []float64) {
                                        var nextLayerOutputs mat.VecDense
                                        nextLayerOutputs.MulVec(weights, mat.NewVecDense(m, prev))
                                        nextLayerOutputs.AddVec(bias, &nextLayerOutputs)
                                        for i, _ := range next {
                                                next[i] = nextLayerOutputs.AtVec(i)
                                        }
                                },
                                input,
                                &fd.JacobianSettings{
                                        Formula:    fd.Central,
                                        // Concurrent: true ?
                                        Concurrent: false,
                                })
                } else {
                        jac = weights
                }

                // [M * N] x [N * 1] => [M * 1]
                var activationDelta mat.VecDense
                // Safely compute matrix multiplication with NaN checking
                nextLayerDeltas := convertDeltasToDense(nextLayer, nn.params)
                
                // Check if any deltas are NaN
                hasNaN := false
                for i := 0; i < nextLayerDeltas.Len(); i++ {
                        if math.IsNaN(nextLayerDeltas.AtVec(i)) {
                                hasNaN = true
                                break
                        }
                }
                
                if hasNaN {
                        // Use zeros instead of NaN values
                        activationDelta = *mat.NewVecDense(m, nil)
                } else {
                        activationDelta.MulVec(jac.T(), nextLayerDeltas)
                }
                
                deltas := mat.NewVecDense(activationDelta.Len(), nil)
                for i := 0; i < activationDelta.Len(); i++ {
                        delta := capValue(float32(activationDelta.AtVec(i)), nn.params)
                        derivative := capValue(layer.activation.Derivative(float32(input[i])), nn.params)
                        // Protect against NaN in the multiplication
                        result := delta * derivative
                        if math.IsNaN(float64(result)) {
                                result = 0.0
                        }
                        deltas.SetVec(i, float64(result))
                }

                layer.deltas = make([]float32, m)
                for i, _ := range layer.deltas {
                        layer.deltas[i] = float32(deltas.AtVec(i))
                        layer.deltas[i] = capValue(layer.deltas[i], nn.params)
                }
        }
        // TODO can be changed in real time based on schedule
        nn.UpdateWeights(nn.params.lr)
}

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

func softmax(output []float32, params Params) []float32 {
        expValues := make([]float32, len(output))

        for i, value := range output {
            expValues[i] = capValue(float32(math.Exp(float64(value))), params)
        }

        var sum float32 = 0.0
        for i := range expValues {
                sum += expValues[i]
        }

        for i := range expValues {
            expValues[i] /= sum
        }
    
        return expValues
 }

func xavierInit(numInputs int, numOutputs int, params Params) float32 {
        limit := math.Sqrt(6.0 / float64(numInputs + numOutputs))
        xavier := float32(2 * rand.Float64() * limit - limit)
        return capValue(xavier, params)
}

func capValue(value float32, params Params) float32 {
        // Check for NaN and return a small value instead
        if math.IsNaN(float64(value)) {
                return params.lowCap
        }
        
        if math.IsInf(float64(value), 0) {
                if value > 0 {
                        return 1.0 / params.lowCap // Return a large positive value
                } else {
                        return -1.0 / params.lowCap // Return a large negative value
                }
        }
        
        if (params.lowCap == 0) {
                return value
        }
        
        if value >= 0 {
                return float32(math.Min(math.Max(float64(value), float64(params.lowCap)), float64(1/params.lowCap)))
        }
        
        return -capValue(-value, params)
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
