package neuralnet

import (
        "fmt"
	"strings"
        "math"
        "math/rand"
        "gonum.org/v1/gonum/diff/fd"
        "gonum.org/v1/gonum/mat"
)

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
        layers []*Layer
        input []float32
        lr float32
        L2 float32
}

func NewNeuralNetwork(inputSize int, hidden []int, outputSize int, learningRate float32, regularization float32) *NeuralNetwork {
        rand.Seed(int64(NNSeed(inputSize, hidden, outputSize)))

        nn := &NeuralNetwork{
                // input + hidden + output
                layers: make([]*Layer, len(hidden) + 2),
                lr: learningRate,
                L2: regularization,
        }

        first := &Layer{
                neurons: make([]*Neuron, inputSize),
                activation: ReLU{},
        }
        for j := 0; j < inputSize; j++ {
                first.neurons[j] = &Neuron{
                        weights: make([]float32, inputSize),
                        bias:    xavierInit(inputSize, hidden[0]),
                }
                // TODO: use gauss to init weights
                for k := range first.neurons[j].weights {
                        first.neurons[j].weights[k] = xavierInit(inputSize, hidden[0])
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
                                bias:    xavierInit(size, out),
                        }
                        // TODO: use gauss to init weights
                        for k := range layer.neurons[j].weights {
                                layer.neurons[j].weights[k] =  xavierInit(size, out)
                        }
                }
                nn.layers[i + 1] = layer
        }
        output := &Layer{
                neurons: make([]*Neuron, outputSize),
                activation: ReLU{},
        }
        for l := 0; l < outputSize; l++ {
                output.neurons[l] = &Neuron{
                        weights: make([]float32, len(nn.layers[len(nn.layers) - 2].neurons)),
                        bias:    xavierInit(outputSize, outputSize),
                }
                // TODO: use gauss to init weights
                for k := range output.neurons[l].weights {
                        output.neurons[l].weights[k] = xavierInit(outputSize, outputSize)
                }
        }
        nn.layers[len(hidden) + 1] = output
        return nn
}

func (nn *NeuralNetwork) SetActivation(layerIndex int, activation ActivationFunction) {
        nn.layers[layerIndex].activation = activation
}

func NNSeed(inputSize int, hidden []int, outputSize int) int {
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
        }
        nn.input = saveInput
        for i := 1; i < len(nn.layers); i++ {
                for _, neuron := range nn.layers[i].neurons {
                        neuron.output = neuron.bias
                        for j := 0; j < len(nn.layers[i - 1].neurons); j++ {
                                neuron.output += nn.layers[i - 1].neurons[j].output * neuron.weights[j]
                        }
                        neuron.output = nn.layers[i].activation.Activate(neuron.output)
                }
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

func (nn *NeuralNetwork) CalculateProps() *mat.VecDense {
        outputLayer := nn.layers[len(nn.layers) - 1].neurons
        output := make([]float32, len(outputLayer))
        for i, neuron := range outputLayer {
                output[i] = neuron.output
        }
        softmax := Softmax(output)
        softmaxFloat64 := make([]float64, len(softmax))
        for i, v := range softmax {
                softmaxFloat64[i] = float64(v)
        }
        return mat.NewVecDense(len(outputLayer), softmaxFloat64)
}

func (nn *NeuralNetwork) CalculateLoss(target mat.VecDense) float32 {
        props := nn.CalculateProps()
        outputNeurons := nn.layers[len(nn.layers) - 1].neurons
        var loss float32 = 0.0
        for i := 0; i < len(outputNeurons); i++ {
                if (target.AtVec(i) == 1.0) {
                        loss -= float32(math.Log(props.AtVec(i)))
                }
        }
        // Add L2 regularization term
        for _, layer := range nn.layers {
                for _, neuron := range layer.neurons {
                        for _, weight := range neuron.weights {
                                loss += nn.L2 * weight * weight
                        }
                }
        }
        return loss
}

func (nn *NeuralNetwork) Backpropagate(target mat.VecDense, jacobian bool) {
        outputLayer := nn.layers[len(nn.layers) - 1]
        outputLayer.deltas = make([]float32, len(outputLayer.neurons))
        props := nn.CalculateProps()
        // Softmax - 1 or Softmax
        for i := 0; i < len(outputLayer.neurons); i++ {
                if (target.AtVec(i) == 1.0) {
                        outputLayer.deltas[i] = float32(props.At(i, 0)) - 1
                } else {
                        outputLayer.deltas[i] = float32(props.At(i, 0))
                }
        }

        for i := len(nn.layers) - 2; i >= 0; i-- {
                // M neurons neurons to calculate deltas for    
                layer := nn.layers[i]
                m := len(layer.neurons)
                // N neurons to propagate error from
                nextLayer := nn.layers[i + 1]
                n := len(nextLayer.neurons)

                // N * M
                weights := ConvertWeightsDense(nextLayer.neurons)

                input := make([]float64, m)
                for i, neuron := range layer.neurons {
                        input[i] = float64(neuron.output)
                }

                jac := mat.NewDense(n, m, nil)
                if (jacobian) {
                        // N * 1
                        bias := ConvertBiasToDense(nextLayer.neurons)
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
                activationDelta.MulVec(jac.T(), ConvertDeltasToDense(nextLayer))

                deltas := mat.NewVecDense(activationDelta.Len(), nil)
                for i := 0; i < activationDelta.Len(); i++ {
                        deltas.SetVec(i, activationDelta.AtVec(i) * float64(layer.activation.Derivative(float32(input[i]))))
                }

                layer.deltas = make([]float32, m)
                for i, _ := range layer.deltas {
                        layer.deltas[i] = float32(deltas.AtVec(i))
                }
        }
        // TODO can be changed in real time based on schedule
        nn.UpdateWeights(nn.lr)
}

func (nn *NeuralNetwork) UpdateWeights(learningRate float32) {
        // 1st layer update
        for j, neuron := range nn.layers[0].neurons {
                for k := range neuron.weights {
                        // Gradient descent: w_new = w_old + learningRate * delta * previous_layer_output
                        neuron.weights[k] -= learningRate * nn.layers[0].deltas[j] * nn.input[k]
                        // regulisation term
                        neuron.weights[k] -= nn.L2 * neuron.weights[k]
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
                    neuron.weights[k] -= nn.L2 * neuron.weights[i]
                }
    
                // Update bias: bias_new = bias_old + learningRate * delta
                neuron.bias -= learningRate * currentLayer.deltas[j]
            }
        }
}

// func (nn *NeuralNetwork) Train(trainingData [][]float32, expectedOutputs [][]float32, learningRate float32, epochs int) {
//         for epoch := 0; epoch < epochs; epoch++ {
//                 for i := 0; i < len(trainingData); i++ {
//                         nn.FeedForward(trainingData[i])
//                         // TODO: Backpropagation
//                 }
//         }
// }

// N number of 2nd layer neurons on M number of prev layer so N * M
func ConvertWeightsDense(neurons []*Neuron) *mat.Dense {
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

func ConvertBiasToDense(neurons []*Neuron) *mat.VecDense {
        dense := mat.NewVecDense(len(neurons), nil)
        for i, neuron := range neurons {
                dense.SetVec(i, float64(neuron.bias))
        }
        return dense
}

func ConvertDeltasToDense(layer *Layer) *mat.VecDense {
        dense := mat.NewVecDense(len(layer.deltas), nil)
        for i, d := range layer.deltas {
                dense.SetVec(i, float64(d))
        }
        return dense
}

func Softmax(output []float32) []float32 {
        expValues := make([]float32, len(output))
        var sum float32 = 0.0

        for i, value := range output {
            expValues[i] = float32(math.Exp(float64(value)))
            sum += expValues[i]
        }

        for i := range expValues {
            expValues[i] /= sum
        }
    
        return expValues
 }

 // TODO : norm(0, 1/num_inputs)
 func xavierInit(numInputs int, numOutputs int) float32 {
        limit := math.Sqrt(6.0 / float64(numInputs + numOutputs))
        return float32(2 * rand.Float64() * limit - limit)
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

// Define the String() method for the NeuralNetwork type
func (nn *NeuralNetwork) String() string {
	var sb strings.Builder

	// Iterate through the layers and print them
	for i, layer := range nn.layers {
		sb.WriteString(fmt.Sprintf("Layer %d:\n%s\n", i, layer.String()))
	}

	return sb.String()
}