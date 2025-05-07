package neuralnet

import "testing"

func TestSGDApplyInvalidBatchSize(t *testing.T) {
    sgd := &SGD{}
    nn := DefaultNeuralNetwork(1, []int{1}, 1)
    if err := sgd.Apply(nn, 0); err == nil {
        t.Error("SGD.Apply with batchSize=0 did not return error")
    }
}

func TestSGDApplyValidBatchSize(t *testing.T) {
    sgd := &SGD{}
    nn := DefaultNeuralNetwork(1, []int{1}, 1)
    if err := sgd.Apply(nn, 1); err != nil {
        t.Errorf("SGD.Apply with batchSize=1 returned error: %v", err)
    }
}
