package neuralnet

import (
    "math"
    "testing"
)

func TestCrossEntropyCompute(t *testing.T) {
    ce := &CrossEntropy{}
    output := []float32{0.5, 0.5}
    target := []float32{1.0, 0.0}
    loss := ce.Compute(output, target)
    want := float32(-math.Log(0.5))
    if diff := loss - want; diff < -1e-3 || diff > 1e-3 {
        t.Errorf("CrossEntropy.Compute = %v; want approx %v", loss, want)
    }
}

func TestCrossEntropyGradient(t *testing.T) {
    ce := &CrossEntropy{}
    output := []float32{0.5, 0.5}
    target := []float32{1.0, 0.0}
    grad := ce.Gradient(output, target)
    want := []float32{-0.5, 0.5}
    for i := range grad {
        if grad[i] != want[i] {
            t.Errorf("CrossEntropy.Gradient[%d] = %v; want %v", i, grad[i], want[i])
        }
    }
}
