Sample Architecture:
Here’s a simple architecture you can start with:

Input layer: 576 neurons (for 24x24 grayscale image)
Hidden layer 1: 128 or 256 neurons, ReLU activation
Output layer: 10 neurons, softmax activation

Profile:
go build -o load  
./load -cpuprofile=cpu.prof
go tool pprof cpu.prof
>top10

TODO: batches
TODO: Use Leaky ReLU on the last (pre last) layer
TODO: learn more on Stochastic Gradient Descent (SGD) ?
TODO: train/validation split
TODO: train progress
TODO: evalution on valdation set
TODO: save/load model
TODO: feed images
TODO: use colors of image
TODO: measure perfomance/time
TODO: compare float32 vs float64 perfomance
TODO: consider "manual" GC
TODO: put it into Docker

TODO?:
func init() {
	mat64.Register(goblas.Blas{})
}
