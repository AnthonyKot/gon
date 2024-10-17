Sample Architecture:
Here’s a simple architecture you can start with:

Input layer: 576 neurons (for 24x24 grayscale image)
Hidden layer 1: 128 or 256 neurons, ReLU activation
Output layer: 10 neurons, softmax activation

TODO: train + evolute
TODO: train/validation split
TODO: train progress
TODO: save/load model
TODO: batches
TODO: feed images
TODO: use colors of image
TODO: measure perfomance/time
TODO: compare float32 vs float64 perfomance

TODO?:
func init() {
	mat64.Register(goblas.Blas{})
}
