package neuralnet

import "testing"

func TestReLUActivate(t *testing.T) {
    r := ReLU{}
    if got := r.Activate(-1); got != 0 {
        t.Errorf("ReLU.Activate(-1) = %v; want 0", got)
    }
    if got := r.Activate(2); got != 2 {
        t.Errorf("ReLU.Activate(2) = %v; want 2", got)
    }
}

func TestSigmoidActivate(t *testing.T) {
    s := Sigmoid{}
    got := s.Activate(0)
    want := float32(0.5)
    if diff := got - want; diff < -1e-3 || diff > 1e-3 {
        t.Errorf("Sigmoid.Activate(0) = %v; want approx %v", got, want)
    }
}

func TestLinearActivate(t *testing.T) {
    l := Linear{}
    input := float32(3.14)
    if got := l.Activate(input); got != input {
        t.Errorf("Linear.Activate(%v) = %v; want %v", input, got, input)
    }
}
