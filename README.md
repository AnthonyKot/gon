Architecture:
Input layer: 1024 neurons (for 32x32 grayscale image)
Hidden layer 1: 256 neurons, ReLU activation
Hidden layer 2: 128 neurons, ReLU activation
Hidden layer 3: 10 neurons, Leaky ReLU activation
Output layer: softmax activation

"10 to 50 epochs for medium-sized datasets.
50 to 200 epochs for very large datasets."

TODO: learn more on Stochastic Gradient Descent (SGD) ?

Profile:
go build -o load  
./load -cpuprofile=cpu.prof
go tool pprof cpu.prof
>top10

TODO: batches
TODO: manage LR while training
TODO: better gradient descent (momentum)
TODO: 2nd order gradient descent ?


TODO: other weight init => norm(0, 1/num_inputs)
TODO: train progress
TODO: run on N cores in parallel (8 cores?)
TODO: Use Leaky ReLU on the last (pre last) layer

TODO: train/validation split
TODO: evalution on valdation set
TODO: save/load model

TODO: use colors of image
TODO: measure perfomance/time
TODO: compare float32 vs float64 perfomance
TODO: consider "manual" GC
TODO: put it into Docker

TODO?:
func init() {
	mat64.Register(goblas.Blas{})
}
