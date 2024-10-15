package neuralnet

import (
        "fmt"
	"strings"
        "math"
        "math/rand"
        "gonum.org/v1/gonum/diff/fd"
        "gonum.org/v1/gonum/mat"
)

type ActivationFunction func(float32) float32

type Neuron struct {
        weights []float32
        bias    float32
        output  float32
        activation ActivationFunction
}

type Layer struct {
        neurons []*Neuron
        deltas  []float32
}

// Represents of the simlest NN.
type NeuralNetwork struct {
        layers []*Layer
}

func NewNeuralNetwork(inputSize int, hidden []int, outputSize int) *NeuralNetwork {
        rand.Seed(int64(NNSeed(inputSize, hidden, outputSize)))

        nn := &NeuralNetwork{
                // input + hidden + output
                layers: make([]*Layer, len(hidden) + 2),
        }

        first := &Layer{
                neurons: make([]*Neuron, inputSize),
        }
        for j := 0; j < inputSize; j++ {
                first.neurons[j] = &Neuron{
                        weights: make([]float32, inputSize),
                        bias:    rand.Float32(),
                        activation: ReLU,
                }
                // TODO: use gauss to init weights
                for k := range first.neurons[j].weights {
                        first.neurons[j].weights[k] = rand.Float32()
                }
        }
        nn.layers[0] = first

        for i, size := range hidden {
                layer := &Layer{
                        neurons: make([]*Neuron, size),
                }
                for j := range layer.neurons {
                        layer.neurons[j] = &Neuron{
                                weights: make([]float32, len(nn.layers[i].neurons)),
                                bias:    rand.Float32(),
                                activation: ReLU,
                        }
                        // TODO: use gauss to init weights
                        for k := range layer.neurons[j].weights {
                                layer.neurons[j].weights[k] = rand.Float32()
                        }
                }
                nn.layers[i + 1] = layer
        }
        output := &Layer{
                neurons: make([]*Neuron, outputSize),
        }
        for l := 0; l < outputSize; l++ {
                output.neurons[l] = &Neuron{
                        weights: make([]float32, len(nn.layers[len(nn.layers) - 2].neurons)),
                        bias:    rand.Float32(),
                        activation: ReLU,
                }
                // TODO: use gauss to init weights
                for k := range output.neurons[l].weights {
                        output.neurons[l].weights[k] = rand.Float32()
                }
        }
        nn.layers[len(hidden) + 1] = output
        return nn
}

func (nn *NeuralNetwork) SetActivation(layerIndex int, activation ActivationFunction) {
        for _, neuron := range nn.layers[layerIndex].neurons {
                neuron.activation = activation
        }
}

func NNSeed(inputSize int, hidden []int, outputSize int) int {
        seed := inputSize
        for _, h := range(hidden) {
                seed = seed + h
        }
        return seed + outputSize
}

func (nn *NeuralNetwork) FeedForward(target *mat.VecDense) {
        // the first layer takes inputs as X
        for _, neuron := range nn.layers[0].neurons {
                neuron.output = neuron.bias
                for j := 0; j < target.Len(); j++ {
                        neuron.output += float32(target.AtVec(j)) * neuron.weights[j]
                }
                neuron.output = neuron.activation(neuron.output)
        }
        for i := 1; i < len(nn.layers); i++ {
                for _, neuron := range nn.layers[i].neurons {
                        neuron.output = neuron.bias
                        for j := 0; j < len(nn.layers[i - 1].neurons); j++ {
                                neuron.output += nn.layers[i - 1].neurons[j].output * neuron.weights[j]
                        }
                        neuron.output = neuron.activation(neuron.output)
                }
        }
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

func (nn *NeuralNetwork) CalculateLoss(target *mat.VecDense) float64 {
        props := nn.CalculateProps()
        outputNeurons := nn.layers[len(nn.layers) - 1].neurons
        for i := 0; i < len(outputNeurons); i++ {
                if (target.AtVec(i) == 1.0) {
                        return -math.Log(props.AtVec(i))
                }
        }
        panic("Correct label is not defined")
}

func (nn *NeuralNetwork) Backpropagate(target *mat.VecDense) {
        outputLayer := nn.layers[len(nn.layers)-1]
        outputLayer.deltas = make([]float32, len(outputLayer.neurons))
        props := nn.CalculateProps()
        // Softmax or 1 - Softmax
        for i := 0; i < len(outputLayer.neurons); i++ {
                if (target.AtVec(i) == 1.0) {
                        outputLayer.deltas[i] = float32(props.At(i, 0))
                }
                outputLayer.deltas[i] = 1 - float32(props.At(i, 0))
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
                // N * 1
                bias := ConvertBiasToDense(nextLayer.neurons)

                input := make([]float64, m)
                for i, neuron := range layer.neurons {
                        input[i] = float64(neuron.output)
                }

                // TODO: not sure it's even needed if we store activation
                // jac is delta(ActicatonN)/delta(OutputM)
                jac := mat.NewDense(n, m, nil)
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

                // [M * N] x [N * 1] => [M * 1]
                var activationDelta mat.VecDense
                activationDelta.MulVec(jac.T(), ConvertDeltasToDense(nextLayer))

                deltas := mat.NewVecDense(activationDelta.Len(), nil)
                for i := 0; i < activationDelta.Len(); i++ {
                        deltas.SetVec(i, activationDelta.AtVec(i) * ReLUPrime(input[i]))
                }

                layer.deltas = make([]float32, m)
                for i, _ := range layer.deltas {
                        layer.deltas[i] = float32(deltas.AtVec(i))
                }
        }
}

func (nn *NeuralNetwork) UpdateWeights(learningRate float32) {
        for i := 1; i < len(nn.layers); i++ {
            currentLayer := nn.layers[i]
            previousLayer := nn.layers[i-1]

            for j, neuron := range currentLayer.neurons {
                for k := range neuron.weights {
                    // Gradient descent: w_new = w_old + learningRate * delta * previous_layer_output
                    neuron.weights[k] -= learningRate * currentLayer.deltas[j] * previousLayer.neurons[k].output
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

func ReLU(x float32) float32 {
        return float32(math.Max(float64(x), 0))
}

func ReLUPrime(x float64) float64 {
        if x > 0 {
                return 1
        }
        return 0
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

 // Debug
 func (l *Layer) String() string {
	var sb strings.Builder

	// Print each neuron
	for i, neuron := range l.neurons {
		sb.WriteString(fmt.Sprintf("Neuron %d: weights=%v, bias=%.2f, output=%.2f\n", i, neuron.weights, neuron.bias, neuron.output))
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