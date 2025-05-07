// Package main loads CIFAR-10 data, trains a simple neural network, and evaluates accuracy.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"

	"gon/neuralnet"
	"gonum.org/v1/gonum/mat"
)

const (
	LabelSize = 1
	Colors    = 3
	D         = 32
	Batch     = 10000
	Channel   = D * D
	ImageSize = Channel * Colors
	Row       = LabelSize + ImageSize
)

func loadCIFAR10(filePath string) ([][]mat.Dense, []int, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	images := make([][]mat.Dense, Batch)
	labels := make([]int, Batch)
	for {
		data := make([]byte, Batch*Row)
		_, err := io.ReadFull(file, data)
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, nil, err
		}
		for i := 0; i < Batch; i++ {
			row := data[i*Row : (i+1)*Row]
			labels[i] = int(row[0])

			pixels := row[LabelSize : LabelSize+ImageSize]
			norm := make([]float64, ImageSize)
			for i := 0; i < len(pixels); i++ { // Corrected loop to start from 0
				norm[i] = float64(pixels[i]) / 255.0
			}
			image := make([]mat.Dense, 3)
			for i := 0; i < Colors; i++ {
				channel := norm[i*Channel : (i+1)*Channel]
				image[i] = *mat.NewDense(32, 32, channel)
			}
			images[i] = image
		}
	}

	return images, labels, nil
}

func readLabels() []string {
	file, err := os.Open("data/batches.meta.txt")
	if err != nil {
		fmt.Println("Error opening file:", err)
		panic("Error opening file")
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var words []string
	for scanner.Scan() {
		words = append(words, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("Error reading file:", err)
		panic("Error reading file")
	}
	return words
}

func saveImg(ts [][]mat.Dense, ls []mat.VecDense, ws []string, sampleIdx int, predIdx int) {
	t := ts[sampleIdx] // Corrected: use sampleIdx instead of undefined i
	img := image.NewRGBA(image.Rect(0, 0, 32, 32))
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			r := t[0].At(y, x)
			r8 := uint8(r * 255.0)
			g := t[1].At(y, x)
			g8 := uint8(g * 255.0)
			b := t[2].At(y, x)
			b8 := uint8(b * 255.0)
			img.Set(x, y, color.RGBA{r8, g8, b8, 255})
		}
	}
	// Save the image to a PNG file
	trueIdx := 0
	for k := 0; k < ls[sampleIdx].Len(); k++ {
		if ls[sampleIdx].AtVec(k) == 1.0 {
			trueIdx = k
		}
	}
	label := fmt.Sprintf("file_%s_%d_%s.png", ws[trueIdx], sampleIdx, ws[predIdx])
	file, err := os.Create(label)
	defer file.Close()
	if err != nil {
		panic(err)
	}

	err = png.Encode(file, img)
	if err != nil {
		panic(err)
	}

	println(fmt.Sprintf("Image saved as %s", label))
}

func oneHotEncode(labels []int, numClasses int) []mat.VecDense {
	encoded := make([]mat.VecDense, len(labels))
	for i, label := range labels {
		vec := mat.NewVecDense(numClasses, nil)
		for j := 0; j < vec.Len(); j++ {
			if label == j {
				vec.SetVec(j, 1.0)
			} else {
				vec.SetVec(j, 0)
			}
		}
		encoded[i] = *vec
	}
	return encoded
}

func description(encoded mat.VecDense, desc []string) string {
	for i := 0; i < encoded.Len(); i++ {
		if encoded.AtVec(i) == 1.0 {
			return desc[i]
		}
	}
	panic("No suitable description")
}

func label(encoded mat.VecDense) int {
	for i := 0; i < encoded.Len(); i++ {
		if encoded.AtVec(i) == 1.0 {
			return i
		}
	}
	panic("No suitable idx")
}

func RGBToBlackWhite(rgbImage []mat.Dense) mat.Dense {
	// Create a new matrix to store the grayscale image
	rows, cols := rgbImage[0].Dims()
	grayImage := mat.NewDense(rows, cols, nil)

	// Iterate over each pixel in the RGB image
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			// Calculate the grayscale value using the weighted average of RGB components
			r, g, b := rgbImage[0].At(i, j), rgbImage[1].At(i, j), rgbImage[2].At(i, j)
			gray := 0.299*r + 0.587*g + 0.114*b // Calculate grayscale as float64
			grayImage.Set(i, j, gray)           // Set the normalized float64 value directly

		}
	}

	return *grayImage
}

func flatten(image mat.Dense) mat.VecDense {
	rows, cols := image.Dims()
	vec := mat.NewVecDense(rows*cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			vec.SetVec(i*cols+j, image.At(i, j))
		}
	}
	return *vec
}

func convertImagesToInputs(images [][]mat.Dense) []mat.VecDense {
	inputs := make([]mat.VecDense, len(images))
	for i := 0; i < len(inputs); i++ {
		bwImage := RGBToBlackWhite(images[i]) // Convert to BW first
		inputs[i] = flatten(bwImage)          // Then flatten
	}
	return inputs
}


func load() ([][]mat.Dense, []mat.VecDense) {
	images, labels, err := loadCIFAR10("data/data_batch_1.bin")
	if err != nil {
		fmt.Println("Error loading CIFAR-10:", err)
		panic("Error loading CIFAR-10")
	}
	out := oneHotEncode(labels, 10)
	return images, out
}

func accuracy(nn *neuralnet.NeuralNetwork, trainingData []mat.VecDense, expectedOutputs []mat.VecDense, from int, to int, numWorkers int) float32 {
	numSamplesToTest := to - from
	if numSamplesToTest <= 0 {
		return 0.0
	}

	correctPredictions := 0

	if numWorkers <= 1 || numSamplesToTest < numWorkers { // Fallback to single-threaded if not enough samples or workers=1
		for i := from; i < to; i++ {
			// For single-threaded, we can use the original nn directly,
			// but for consistency and to ensure no accidental shared state issues if nn.Predict were to change,
			// using a clone is safer, though slightly more overhead.
			// If performance is critical here for single thread, could use 'nn' directly.
			// currentNN := nn
			// For this implementation, let's be consistent with cloning for clarity.
			currentNN := nn.Clone()
			if label(expectedOutputs[i]) == currentNN.Predict(trainingData[i]) {
				correctPredictions++
			}
		}
	} else {
		var wg sync.WaitGroup
		mu := &sync.Mutex{} // Mutex to protect access to correctPredictions

		samplesPerWorker := (numSamplesToTest + numWorkers - 1) / numWorkers // Ceiling division

		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			workerStartOffset := w * samplesPerWorker
			workerEndOffset := workerStartOffset + samplesPerWorker
			if workerStartOffset >= numSamplesToTest {
				wg.Done()
				continue
			}
			if workerEndOffset > numSamplesToTest {
				workerEndOffset = numSamplesToTest
			}

			// Convert offsets to actual indices in trainingData/expectedOutputs
			actualWorkerStart := from + workerStartOffset
			actualWorkerEnd := from + workerEndOffset

			go func(startIdx int, endIdx int) {
				defer wg.Done()
				if startIdx >= endIdx {
					return
				}

				workerNN := nn.Clone() // Each worker gets its own clone
				workerCorrect := 0
				for i := startIdx; i < endIdx; i++ {
					if label(expectedOutputs[i]) == workerNN.Predict(trainingData[i]) {
						workerCorrect++
					}
				}
				mu.Lock()
				correctPredictions += workerCorrect
				mu.Unlock()
			}(actualWorkerStart, actualWorkerEnd)
		}
		wg.Wait()
	}

	return float32(correctPredictions) / float32(numSamplesToTest)
}

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

func runTrainingSession(
	useFloat64Calc bool,
	inputs []mat.VecDense,
	labels []mat.VecDense,
	imgs [][]mat.Dense,
	descr []string,
	from int,
	to int,
	epochs int,
	trainToValidationRatio int,
	miniBatchSize int,
	baseNumWorkers int,
	initialLR float32,
	initialDecay float32,
	initialL2 float32,
	initialLowCap float32,
	initialRelu float32,
	initialMomentum float32,
	initialBN float32,
) {
	fmt.Printf("\n--- Starting Training Session (UseFloat64 for Calculations: %t) ---\n", useFloat64Calc)

	// Create Params for this session
	// Note: NewParamsFull now correctly includes the useFloat64 flag
	currentParams := neuralnet.NewParamsFull(
		initialLR,
		initialDecay,
		initialL2,
		initialLowCap,
		initialRelu,
		initialMomentum,
		initialBN,    // bn
		useFloat64Calc, // UseFloat64
	)

	nn := neuralnet.NewNeuralNetwork(1024, []int{512, 256}, 10, currentParams)
	// Ensure the optimizer is set if NewNeuralNetwork doesn't set a default one
	// or if a specific one is desired. DefaultNeuralNetwork sets SGD.
	// If NewNeuralNetwork is used directly, optimizer might need to be set manually:
	// nn.SetOptimizer(&neuralnet.SGD{}) // Example if needed

	j := 0 // Used for saving sample images
	numWorkers := baseNumWorkers
	if numWorkers > neuralnet.MAX_WORKERS {
		numWorkers = neuralnet.MAX_WORKERS
	}
	fmt.Printf("Number of workers for mini-batch processing: %d\n", numWorkers)
	fmt.Println("---")

	totalTrainingStartTime := time.Now()
	var totalEpochsDuration time.Duration

	for i := 0; i < epochs; i++ {
		epochStartTime := time.Now()
		// Switched to TrainMiniBatch, now with numWorkers
		// The '1' for epochs in TrainMiniBatch means it processes the dataset once per outer loop iteration.
		nn.TrainMiniBatch(inputs[from:to], labels[from:to], miniBatchSize, 1, numWorkers)
		for sample := 0; sample < 3; sample++ {
			// Select a random validation sample to predict and save
			validationStart := to
			validationEnd := to + ((to - from) / trainToValidationRatio)
			if validationEnd > len(inputs) { // Ensure we don't go out of bounds
				validationEnd = len(inputs)
			}
			if validationStart >= validationEnd { // Skip if validation set is empty or invalid
				fmt.Println("Skipping image saving due to invalid validation range.")
				break
			}
			j = validationStart + rand.Intn(validationEnd-validationStart)

			pred := nn.Predict(inputs[j])
			saveImg(imgs, labels, descr, j, pred)
			fmt.Printf("Epoch %d, Sample %d, Output: %v\n", i, sample, nn.Output())
		}
		fmt.Printf("Train accuracy: %.2f, Validation accuracy: %.2f\n",
			accuracy(nn, inputs, labels, from, to, numWorkers),
			accuracy(nn, inputs, labels, to, to+((to-from)/trainToValidationRatio), numWorkers),
		)
		epochDuration := time.Since(epochStartTime)
		totalEpochsDuration += epochDuration
		fmt.Printf("Epoch %d duration: %s\n", i, epochDuration)
		fmt.Println()
	}

	totalTrainingDuration := time.Since(totalTrainingStartTime)
	averageEpochDuration := time.Duration(0)
	if epochs > 0 {
		averageEpochDuration = totalEpochsDuration / time.Duration(epochs)
	}
	fmt.Println("---")
	fmt.Printf("Total training duration for session (UseFloat64: %t): %s\n", useFloat64Calc, totalTrainingDuration)
	fmt.Printf("Average epoch duration for session (UseFloat64: %t): %s\n", useFloat64Calc, averageEpochDuration)
	fmt.Println("---")
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	flag.Parse()
	if *cpuprofile != "" {
		f, _ := os.Create(*cpuprofile)
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	imgs, labels := load()
	descr := readLabels()
	// input layer + 1 hidden layer + output layer
	// around 250k params ~ 1000*250
	inputs := convertImagesToInputs(imgs)

	// Shuffle the entire dataset (inputs and labels consistently) before splitting
	// to ensure training/validation batches are representative.
	if len(inputs) != len(labels) {
		panic("Mismatch between number of inputs and labels before shuffling.")
	}
	datasetSize := len(inputs)
	permutation := rand.Perm(datasetSize)

	shuffledInputs := make([]mat.VecDense, datasetSize)
	shuffledLabels := make([]mat.VecDense, datasetSize)
	shuffledColorImgs := make([][]mat.Dense, datasetSize) // For shuffling the original color images

	for i := 0; i < datasetSize; i++ {
		idx := permutation[i]            // Original index from permutation
		shuffledInputs[i] = inputs[idx]  // inputs here is original b/w flattened
		shuffledLabels[i] = labels[idx]  // labels here is original one-hot
		shuffledColorImgs[i] = imgs[idx] // imgs here is original color images
	}
	// Use the shuffled data from now on
	inputs = shuffledInputs
	labels = shuffledLabels
	imgs = shuffledColorImgs // Assign the shuffled color images back to imgs

	fmt.Printf("Total samples loaded: %d\n", len(inputs))

	// Tanh works the best without Jacobian calculation
	from := 0
	to := from + 8000
	epochs := 10
	train_to_validation := 4

	trainingSetSize := to - from
	validationSetSize := (to - from) / train_to_validation
	miniBatchSize := 64 // Define mini-batch size
	fmt.Printf("Training set size: %d samples\n", trainingSetSize)
	fmt.Printf("Validation set size: %d samples\n", validationSetSize)
	fmt.Printf("Number of main epochs: %d\n", epochs)
	fmt.Printf("Mini-batch size: %d\n", miniBatchSize)
	fmt.Println("---")

	// nn and j are no longer used directly in main after refactoring to runTrainingSession
	// numWorkers is now initialized inside runTrainingSession or passed to it.
	// The baseNumWorkers is now used for runTrainingSession.
	// numWorkers := runtime.NumCPU() // This specific variable is no longer needed here.
	// if numWorkers > neuralnet.MAX_WORKERS {
	// 	numWorkers = neuralnet.MAX_WORKERS
	// }
	// Fetch default parameter values to use as base for both sessions
	// We need a way to get these defaults. Let's assume neuralnet.DefaultParams() exists
	// or we hardcode them based on current defaults if a public getter isn't available.
	// For now, let's use the values from neuralnet.defaultParams() directly.
	// These are: lr: 0.01, decay: 0.95, L2: 1e-4, lowCap: 0, relu: 0, momentum: 0.9, bn: 0.0
	initialLR := float32(0.01)
	initialDecay := float32(0.95)
	initialL2 := float32(1e-4)
	initialLowCap := float32(0.0)
	initialRelu := float32(0.0)      // Assuming this is the default for LeakyReLU alpha if relu param is for that
	initialMomentum := float32(0.9)
	initialBN := float32(0.0)

	baseNumWorkers := runtime.NumCPU() // Use number of available CPUs for workers

	// Run session with UseFloat64 = false
	runTrainingSession(
		false, // useFloat64Calc
		inputs, labels, imgs, descr,
		from, to, epochs, train_to_validation, miniBatchSize, baseNumWorkers,
		initialLR, initialDecay, initialL2, initialLowCap, initialRelu, initialMomentum, initialBN,
	)

	// Run session with UseFloat64 = true
	runTrainingSession(
		true, // useFloat64Calc
		inputs, labels, imgs, descr,
		from, to, epochs, train_to_validation, miniBatchSize, baseNumWorkers,
		initialLR, initialDecay, initialL2, initialLowCap, initialRelu, initialMomentum, initialBN,
	)

	// Original commented out code
	// j := 0 // This j would be uninitialized if only runTrainingSession is called.
	// saveImg(imgs, labels, descr, j, nn.Predict(inputs[j]))
	// nn.TrainMiniBatch(inputs[from:to], labels[from:to], 100, 1)
	// j = to + rand.Intn(to - from / 2)
	// saveImg(imgs, labels, descr, j, nn.Predict(inputs[j]))
	// nn.TrainMiniBatch(inputs[from:to], labels[from:to], 1000, 1)

	// nn = neuralnet.NewNeuralNetwork(1024, []int{512, 256}, 10, lr * 10, L2 * 10)
	// for i := 0; i < epochs; i++ {
	// 	nn.TrainMiniBatch(inputs[from:to], labels[from:to], 1)
	// 	fmt.Println("train", accuracy(nn, inputs, labels, from, to))
	// 	fmt.Println("validation", accuracy(nn, inputs, labels, to, to + ((to - from)/2)))
	// 	fmt.Println()
	// }
	// nn = neuralnet.NewNeuralNetwork(1024, []int{512, 256}, 10, lr * 100, L2 * 100)
	// for i := 0; i < epochs; i++ {
	// 	nn.TrainBatch(inputs[from:to], labels[from:to], 1)
	// 	fmt.Println("train", accuracy(nn, inputs, labels, from, to))
	// 	fmt.Println("validation", accuracy(nn, inputs, labels, to, to + ((to - from)/2)))
	// 	fmt.Println()
	// }
	// profiling top 5
	// flat  flat%   sum%        cum   cum%
	// 24.42% (*NeuralNetwork).UpdateWeights
	// 20.00% gon/neuralnet.ConvertWeightsDense
	// 17.37% gon/neuralnet.(*NeuralNetwork).CalculateLoss
	// 11.85% gonum.org/v1/gonum/mat.(*VecDense).at (inline)
	// 11.51% gon/neuralnet.(*NeuralNetwork).FeedForward
}
