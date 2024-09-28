package neuralnet

import (
        "math"
        "math/rand"
        "time"
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

func (nn *NeuralNetwork) FeedForward(input []float32) []float32 {
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

        return Softmax(output)
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