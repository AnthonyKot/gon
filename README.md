Sample Architecture:
Here’s a simple architecture you can start with:

Input layer: 576 neurons (for 24x24 grayscale image)
Hidden layer 1: 128 or 256 neurons, ReLU activation
Output layer: 10 neurons, softmax activation

TODO: optimisation algo
TODO: train/validation split
TODO: train + evolute
TODO: batches
TODO: compare float32 vs float64 perfomance
TODO: use colors
TODO: check other activations then RELU
TODO: gauss initialisation
TODO: save/load model
TODO: train progress

TODO?:
func init() {
	mat64.Register(goblas.Blas{})
}
