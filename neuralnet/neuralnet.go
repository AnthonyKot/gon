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
)

const MAX_WORKERS int = 8

type Neuron struct {
        weights []float32
        bias    float32
        output  float32
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
        loss    []float32
        params  Params
}

type Params struct {
        lr      float32
        decay   float32
        L2      float32
        lowCap  float32
        relu    float32
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
        return NewParamsFull(learningRate, decay, regularization, cap, defaultParams().relu)
}

func NewParamsFull(learningRate float32, decay float32, regularization float32, cap float32, relu float32) Params {
        return Params{
                lr:      learningRate,
                decay:   decay,
                L2:      regularization,
                lowCap:  cap,
                relu:    relu,
            }
}

func initialise(inputSize int, hidden []int, outputSize int, params Params) *NeuralNetwork {
        rand.Seed(int64(seed(inputSize, hidden, outputSize)))

        nn := &NeuralNetwork{
                // input + hidden + output
                layers: make([]*Layer, len(hidden) + 2),
                params: params,
        }

        first := &Layer{
                neurons: make([]*Neuron, inputSize),
                activation: Linear{},
        }
        for j := 0; j < inputSize; j++ {
                first.neurons[j] = &Neuron{
                        weights: make([]float32, inputSize),
                        bias:    xavierInit(inputSize, hidden[0], nn.params),
                }
                // TODO: use gauss to init weights
                for k := range first.neurons[j].weights {
                        first.neurons[j].weights[k] = xavierInit(inputSize, hidden[0], nn.params)
                }
        }
        nn.layers[0] = first

        for i, size := range hidden {
                layer := &Layer{
                        neurons: make([]*Neuron, size),
                        activation:Linear{},
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
                }
                nn.layers[i + 1] = layer
        }
        output := &Layer{
                neurons: make([]*Neuron, outputSize),
                activation: Linear{},
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

func defaultParams() *Params {
        return &Params{
            lr:      0.01,
            decay:   0.8,
            L2:      0,
            lowCap:  0,
            relu:    0,
        }
}

func DefaultNeuralNetwork(inputSize int, hidden []int, outputSize int) *NeuralNetwork {
        params := defaultParams()
        return initialise(inputSize, hidden, outputSize, *params)
}

func (nn *NeuralNetwork) SetActivation(layerIndex int, activation ActivationFunction) {
        nn.layers[layerIndex].activation = activation
}

func seed(inputSize int, hidden []int, outputSize int) int {
        seed := inputSize
        for _, h := range(hidden) {
                seed = seed + h
        }
        return seed + outputSize
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

func (nn *NeuralNetwork) AccumulateLoss(target mat.VecDense) {
        props := nn.calculateProps()
        // Softmax - 1 or Softmax
        for i := 0; i < len(nn.loss); i++ {
                if (target.AtVec(i) == 1.0) {
                        nn.loss[i] += float32(props.At(i, 0)) - 1
                } else {
                        nn.loss[i] += float32(props.At(i, 0))
                }
                nn.loss[i] = capValue(nn.loss[i], nn.params)
        }
}

func (nn *NeuralNetwork) Backpropagate(dataPoints int, jacobian bool) {
        outputLayer := nn.layers[len(nn.layers) - 1]
        outputLayer.deltas = make([]float32, len(outputLayer.neurons))

        // use accumulated loss deravative as delta for the last layer
        for i := 0; i < len(outputLayer.neurons); i++ {
                outputLayer.deltas[i] = nn.loss[i] / float32(dataPoints)
                outputLayer.deltas[i] = capValue(outputLayer.deltas[i], nn.params)
        }
        for i := 0; i < len(outputLayer.neurons); i++ {
                nn.loss[i] = 0
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
                if (jacobian) {
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
                // TODO: here we got NaN
                activationDelta.MulVec(jac.T(), convertDeltasToDense(nextLayer, nn.params))
                deltas := mat.NewVecDense(activationDelta.Len(), nil)
                for i := 0; i < activationDelta.Len(); i++ {
                        delta := capValue(float32(activationDelta.AtVec(i)), nn.params)
                        derivative:= capValue(layer.activation.Derivative(float32(input[i])), nn.params)
                        deltas.SetVec(i, float64(delta * derivative))
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

func (nn *NeuralNetwork) UpdateWeights(learningRate float32) {
        // 1st layer update
        for j, neuron := range nn.layers[0].neurons {
                for k := range neuron.weights {
                        // Gradient descent: w_new = w_old + learningRate * delta * previous_layer_output
                        neuron.weights[k] -= learningRate * nn.layers[0].deltas[j] * nn.input[k]
                        // regulisation term
                        neuron.weights[k] -= nn.params.L2 * neuron.weights[k]
                        neuron.weights[k] = capValue(neuron.weights[k], nn.params)
                    }
        
                    // Update bias: bias_new = bias_old + learningRate * delta
                    neuron.bias -= learningRate * nn.layers[0].deltas[j]
        }
        
        for i := 1; i < len(nn.layers); i++ {
            currentLayer := nn.layers[i]
            previousLayer := nn.layers[i-1]

            for j, neuron := range currentLayer.neurons {
                for k := range neuron.weights {
                    // Gradient descent: w_new = w_old + learningRate * delta * previous_layer_output
                    neuron.weights[k] -= learningRate * currentLayer.deltas[j] * previousLayer.neurons[k].output
                    // regulisation term
                    neuron.weights[k] -= nn.params.L2 * neuron.weights[i]
                    neuron.weights[k] = capValue(neuron.weights[k], nn.params)
                }
    
                // Update bias: bias_new = bias_old + learningRate * delta
                neuron.bias -= learningRate * currentLayer.deltas[j]
            }
        }
}

func (nn *NeuralNetwork) TrainSGD(trainingData []mat.VecDense, expectedOutputs []mat.VecDense,  epochs int) {
        batchSize := 1
        for e := 0; e < epochs; e++ {
                nn.params.lr = nn.params.lr * nn.params.decay / 100
                var loss float32 = 0.0
                for b := 0; b < len(trainingData); b++ {
                        data, labels := selectSamples(trainingData, expectedOutputs, batchSize)
                        // SGD. Train on 1 random example
                        for i := 0; i < batchSize; i++ {
                                nn.FeedForward(data[i])
                                nn.AccumulateLoss(labels[i])
                                loss += nn.calculateLoss(labels[i])
                                nn.Backpropagate(1, false)
                        }
                }
                nn.params.lr = nn.params.lr * 100
                fmt.Println(fmt.Sprintf("Loss SGD %d = %.2f", e, loss))
        }
}

func (nn *NeuralNetwork) TrainMiniBatch(trainingData []mat.VecDense, expectedOutputs []mat.VecDense,  batchRatio int, epochs int) {
        for e := 0; e < epochs; e++ {
                var loss float32 = 0.0
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
                                perWorker = append(perWorker, tasks[min:max])
                        }
                        for _, task := range(perWorker) {
                                wg.Add(1)
                                go func(work []Task) {
                                        defer wg.Done()
                                        for _, w := range(work) {
                                                nn.FeedForward(w.data)
                                                nn.AccumulateLoss(w.output)
                                                loss += nn.calculateLoss(w.output)
                                        }
                                }(task)
                        }
                        wg.Wait()
                        nn.Backpropagate(len(currentData), false)
                        nn.params.lr = nn.params.lr * nn.params.decay
                }
                fmt.Printf("Time elapsed: %s\n", time.Since(start))
                fmt.Println(fmt.Sprintf("Loss MB %d = %.2f", e, loss / float32(len(trainingData))))
        }
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
                nn.Backpropagate(len(trainingData), false)
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
        outputNeurons := nn.layers[len(nn.layers) - 1].neurons
        var loss float32 = 0.0
        for i := 0; i < len(outputNeurons); i++ {
                if (target.AtVec(i) == 1.0) {
                        loss -= float32(math.Log(props.AtVec(i)))
                        if (math.IsInf(math.Log(props.AtVec(i)), -1)) {
                                math.Log(props.AtVec(i))
                        }
                }
        }
        // Add L2 regularization term
        for _, layer := range nn.layers {
                for _, neuron := range layer.neurons {
                        for _, weight := range neuron.weights {
                                loss += nn.params.L2 * weight * weight
                                loss = capValue(loss, nn.params)
                        }
                }
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

func convertDeltasToDense(layer *Layer, params Params) *mat.VecDense {
        dense := mat.NewVecDense(len(layer.deltas), nil)
        for i, d := range layer.deltas {
                dense.SetVec(i, float64(capValue(d, params)))
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

 // Debug
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