package neuralnet

import (
        "math"
        "math/rand"
        "time"

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
        deltas  *mat.Dense // Added deltas field
}

// Represents of the simlest NN.
type NeuralNetwork struct {
        layers []*Layer
}

func NewNeuralNetwork(hidden []int, inputSize int) *NeuralNetwork {
        rand.Seed(time.Now().UnixNano())
        nn := &NeuralNetwork{
                layers: make([]*Layer, len(hidden) + 1),
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
        return nn
}

func (nn *NeuralNetwork) SetActivation(layerIndex int, activation ActivationFunction) {
        if layerIndex < 0 || layerIndex >= len(nn.layers) {
                panic("Invalid layer index")
        }

        for _, neuron := range nn.layers[layerIndex].neurons {
                neuron.activation = activation
        }
}

func (nn *NeuralNetwork) FeedForward(input []float32) {
        if len(input) != len(nn.layers[0].neurons) {
                panic("Input size mismatch")
        }

        for i := 0; i < len(nn.layers); i++ {
                if i == 0 {
                        for _, neuron := range nn.layers[i].neurons {
                                neuron.output = neuron.bias
                                for j := 0; j < len(input); j++ {
                                        neuron.output += input[j] * neuron.weights[j]
                                }
                                neuron.output = neuron.activation(neuron.output)
                        }
                } else {
                        for _, neuron := range nn.layers[i].neurons {
                                neuron.output = neuron.bias
                                for j := 0; j < len(nn.layers[i - 1].neurons); j++ {
                                        neuron.output += nn.layers[i - 1].neurons[j].output * neuron.weights[j]
                                }
                                neuron.output = neuron.activation(neuron.output)
                        }
                }
        }

        outputLayer := nn.layers[len(nn.layers) - 1].neurons
        output := make([]float32, len(outputLayer))
        for i, neuron := range outputLayer {
                output[i] = neuron.output
        }
        softmax := Softmax(output)
        for i, neuron := range outputLayer {
                neuron.output = softmax[i]
        }
}

func (nn *NeuralNetwork) Backpropagate(target *mat.Dense) {
        outputLayer := nn.layers[len(nn.layers)-1]
        outputLayer.deltas = ConvertOutputToDense(outputLayer.neurons)
        for i, neuron := range outputLayer.neurons {
                outputLayer.deltas.Set(i, 0, float64(neuron.output) - outputLayer.deltas.At(i, 1))
        }

        for i := len(nn.layers) - 2; i >= 0; i-- {
                layer := nn.layers[i]
                layer.deltas = ConvertOutputToDense(layer.neurons)
                previouseLayer := nn.layers[i+1]
                weights := ConvertWeightsDense(previouseLayer.neurons)
                bias := ConvertBiasToDense(previouseLayer.neurons)

                jac, err := fd.Jacobian(func(x []float64) []float64 {
                        previouseLayerOutputs := weights.Mul(mat.NewDense(previouseLayer.weights.Rows(), 1, x), nil)
                        previouseLayerOutputs.Add(bias, previouseLayerOutputs)
                        return previouseLayerOutputs.Raw()
                }, layer.outputs.Raw())
                if err != nil {
                        panic(err)
                }

                layer.deltas.Mul(jac.T(), nextLayer.deltas, layer.deltas)
                layer.deltas.Apply(func(i, j int, v float64) float64 {
                        return v * ReLUPrime(layer.outputs.At(i, j))
                }, layer.deltas)
        }
}

func ConvertOutputToDense(neurons []*Neuron) *mat.Dense {
        dense = mat.NewDense(len(neurons), 1, nil)
        for i, neuron := range neurons {
                for j, w := range neurons.weights {
                        dense.Set(i, j, float64(neuron.output))
                }
        }
        return dense
}

func ConvertWeightsDense(neurons []*Neuron) *mat.Dense {
        dense = mat.NewDense(len(neurons), 1, nil)
        for i, neuron := range neurons {
                dense.Set(i, 0, float64(neuron.weights))
        }
        return dense
}

func ConvertBiasToDense(neurons []*Neuron) *mat.Dense {
        dense = mat.NewDense(len(neurons), 1, nil)
        for i, neuron := range neurons {
                dense.Set(i, 0, float64(neuron.bias))
        }
        return dense
}

func (nn *NeuralNetwork) Train(trainingData [][]float32, expectedOutputs [][]float32, learningRate float32, epochs int) {
        for epoch := 0; epoch < epochs; epoch++ {
                for i := 0; i < len(trainingData); i++ {
                        nn.FeedForward(trainingData[i])
                        // TODO: Backpropagation
                }
        }
}

func ReLU(x float32) float32 {
        return float32(math.Max(float64(x), 0))
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