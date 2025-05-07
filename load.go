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
	"path/filepath" // Added for joining paths
)

const (
	TempDirName      = "temp" // Directory to save images
	TrainSplitRatio  = 0.8    // Proportion of data used for training
	LabelSize        = 1
	Colors           = 3
	D                = 32
	Batch            = 10000
	Channel          = D * D
	ImageSize        = Channel * Colors
	Row              = LabelSize + ImageSize
	// Training configuration constants
	NumClasses       = 10
	NumEpochs        = 10
	MiniBatchSize    = 64
	NumSamplesToSave = 3
)

// Returns: list of images [image][channel][pixel_data_float64], list of labels [label_int], error
func loadCIFAR10(filePath string) ([][][]float64, []int, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	images := make([][][]float64, Batch) // Changed type
	labels := make([]int, Batch)
	readCount := 0 // Keep track of how many images we've actually read
	for readCount < Batch {
		// Read one row (image + label) at a time for simplicity
		row := make([]byte, Row)
		n, err := io.ReadFull(file, row)
		if err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				break // End of file or incomplete row
			}
			return nil, nil, err
		}
		if n != Row {
			// Should not happen with io.ReadFull unless EOF
			break
		}

		labels[readCount] = int(row[0])

		pixels := row[LabelSize : LabelSize+ImageSize]
		norm := make([]float64, ImageSize)
		for k := 0; k < len(pixels); k++ { // Corrected loop to start from 0
			norm[k] = float64(pixels[k]) / 255.0
		}
		imageChannels := make([][]float64, Colors) // Changed type
		for k := 0; k < Colors; k++ {
			channel := norm[k*Channel : (k+1)*Channel]
			imageChannels[k] = channel // Store the slice directly
		}
		images[readCount] = imageChannels
		readCount++
	}
	// Trim slices if fewer than Batch images were read
	images = images[:readCount]
	labels = labels[:readCount]

	return images, labels, nil
}

// Removed duplicated block from loadCIFAR10

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

func saveImg(ts [][][]float64, ls [][]float32, ws []string, sampleIdx int, predIdx int) {
	t := ts[sampleIdx] // t is now [][]float64
	img := image.NewRGBA(image.Rect(0, 0, 32, 32))
	rows, cols := D, D // Assuming D=32
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			// Access pixel data directly from slices
			// Assuming t[0], t[1], t[2] are R, G, B channels flattened row-wise
			pixelIndex := y*cols + x
			r := t[0][pixelIndex]
			r8 := uint8(r * 255.0)
			g := t[1][pixelIndex]
			g8 := uint8(g * 255.0)
			b := t[2][pixelIndex]
			b8 := uint8(b * 255.0)
			img.Set(x, y, color.RGBA{r8, g8, b8, 255})
		}
	}
	// Save the image to a PNG file
	trueIdx := 0
	labelVec := ls[sampleIdx] // labelVec is []float32
	for k := 0; k < len(labelVec); k++ {
		if labelVec[k] == 1.0 {
			trueIdx = k
			break // Found the true label index
		}
	}

	// Ensure the temp directory exists
	err := os.MkdirAll(TempDirName, 0755) // 0755 are standard permissions
	if err != nil {
		fmt.Printf("Error creating directory %s: %v\n", TempDirName, err)
		// Decide if we should panic or just try saving in the current dir
		// For now, let's panic as saving might fail anyway.
		panic(err)
	}

	// Construct the full path including the directory
	baseFilename := fmt.Sprintf("file_%s_%d_%s.png", ws[trueIdx], sampleIdx, ws[predIdx])
	fullPath := filepath.Join(TempDirName, baseFilename)

	file, err := os.Create(fullPath)
	defer file.Close()
	if err != nil {
		panic(err)
	}

	err = png.Encode(file, img)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Image saved as %s\n", fullPath) // Use fmt.Printf for consistency
}

func oneHotEncode(labels []int, numClasses int) [][]float32 { // Return [][]float32
	encoded := make([][]float32, len(labels))
	for i, label := range labels {
		vec := make([]float32, numClasses) // Create slice
		if label >= 0 && label < numClasses {
			vec[label] = 1.0 // Set the corresponding index to 1.0
		} else {
			// Handle invalid label index if necessary
			fmt.Printf("Warning: Invalid label %d found during one-hot encoding.\n", label)
		}
		encoded[i] = vec
	}
	return encoded
}

func description(encoded []float32, desc []string) string { // Input is []float32
	for i := 0; i < len(encoded); i++ { // Use len() for slice
		if encoded[i] == 1.0 { // Access slice element directly
			return desc[i]
		}
	}
	panic("No suitable description")
}

func label(encoded []float32) int { // Input is []float32
	for i := 0; i < len(encoded); i++ {
		if encoded[i] == 1.0 {
			return i
		}
	}
	panic("No suitable idx")
}

// Converts RGB image (represented as [][]float64 channels) to a flattened B&W []float32 slice
func RGBToBlackWhiteFlattened(rgbImage [][]float64) []float32 {
	if len(rgbImage) != Colors {
		panic("RGBToBlackWhiteFlattened: Expected 3 color channels")
	}
	numPixels := len(rgbImage[0]) // Assuming all channels have same length
	if numPixels != Channel {     // Channel = D*D
		panic("RGBToBlackWhiteFlattened: Incorrect number of pixels per channel")
	}

	grayImage := make([]float32, numPixels)

	// Iterate over each pixel
	for i := 0; i < numPixels; i++ {
		r := rgbImage[0][i]
		g := rgbImage[1][i]
		b := rgbImage[2][i]
		gray := 0.299*r + 0.587*g + 0.114*b // Calculate grayscale as float64
		grayImage[i] = float32(gray)        // Store as float32
	}

	return grayImage
}

// flatten function removed as its logic is integrated into RGBToBlackWhiteFlattened

func convertImagesToInputs(images [][][]float64) [][]float32 { // Input/Output types changed
	inputs := make([][]float32, len(images))
	for i := 0; i < len(inputs); i++ {
		inputs[i] = RGBToBlackWhiteFlattened(images[i]) // Convert and flatten directly
	}
	return inputs
}


func load() ([][][]float64, [][]float32) { // Return types changed
	images, labels, err := loadCIFAR10("data/data_batch_1.bin")
	if err != nil {
		fmt.Println("Error loading CIFAR-10:", err)
		panic("Error loading CIFAR-10")
	}
	// Remove redundant declaration: out := oneHotEncode(labels, 10)
	out := oneHotEncode(labels, NumClasses)
	return images, out
}

func accuracy(nn *neuralnet.NeuralNetwork, trainingData [][]float32, expectedOutputs [][]float32, from int, to int, numWorkers int) float32 {
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

func calculateAndPrintPerClassAccuracy(nn *neuralnet.NeuralNetwork, trainingData [][]float32, expectedOutputs [][]float32, from int, to int, numWorkers int, descr []string) {
	numSamplesToTest := to - from
	if numSamplesToTest <= 0 {
		fmt.Println("No samples in validation set to calculate per-class accuracy.")
		return
	}

	classCorrectCounts := make([]int, NumClasses)
	classTotalCounts := make([]int, NumClasses)

	if numWorkers <= 1 || numSamplesToTest < numWorkers { // Fallback to single-threaded
		for i := from; i < to; i++ {
			currentNN := nn.Clone() // Use clone for consistency
			trueLabelIdx := label(expectedOutputs[i])
			predLabelIdx := currentNN.Predict(trainingData[i])

			if trueLabelIdx >= 0 && trueLabelIdx < NumClasses {
				classTotalCounts[trueLabelIdx]++
				if trueLabelIdx == predLabelIdx {
					classCorrectCounts[trueLabelIdx]++
				}
			}
		}
	} else {
		var wg sync.WaitGroup
		mu := &sync.Mutex{} // Mutex to protect access to shared counts

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

			actualWorkerStart := from + workerStartOffset
			actualWorkerEnd := from + workerEndOffset

			go func(startIdx int, endIdx int) {
				defer wg.Done()
				if startIdx >= endIdx {
					return
				}

				workerNN := nn.Clone() // Each worker gets its own clone
				// Local counts per worker to reduce mutex contention
				localClassCorrect := make([]int, NumClasses)
				localClassTotal := make([]int, NumClasses)

				for i := startIdx; i < endIdx; i++ {
					trueLabelIdx := label(expectedOutputs[i])
					predLabelIdx := workerNN.Predict(trainingData[i])

					if trueLabelIdx >= 0 && trueLabelIdx < NumClasses {
						localClassTotal[trueLabelIdx]++
						if trueLabelIdx == predLabelIdx {
							localClassCorrect[trueLabelIdx]++
						}
					}
				}
				// Aggregate local counts into shared counts under mutex
				mu.Lock()
				for k := 0; k < NumClasses; k++ {
					classCorrectCounts[k] += localClassCorrect[k]
					classTotalCounts[k] += localClassTotal[k]
				}
				mu.Unlock()
			}(actualWorkerStart, actualWorkerEnd)
		}
		wg.Wait()
	}

	fmt.Println("--- Per-Class Validation Accuracy ---")
	for k := 0; k < NumClasses; k++ {
		className := descr[k]
		if classTotalCounts[k] > 0 {
			acc := float32(classCorrectCounts[k]) / float32(classTotalCounts[k])
			fmt.Printf("Accuracy of %8s : %.0f %%\n", className, acc*100)
		} else {
			fmt.Printf("Accuracy of %8s : N/A (0 samples)\n", className)
		}
	}
	fmt.Println("------------------------------------")

}

var (
    flagLR      = flag.Float64("lr", 0.01, "learning rate")
    flagDecay   = flag.Float64("decay", 0.95, "learning rate decay")
    flagEpochs  = flag.Int("epochs", NumEpochs, "number of epochs")
    flagBatch   = flag.Int("batch", MiniBatchSize, "mini-batch size")
    flagWorkers = flag.Int("workers", runtime.NumCPU(), "number of workers")
)
var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

func runTrainingSession(
	// useFloat64Calc bool, // Parameter removed
	inputs [][]float32, // Changed type
	labels [][]float32, // Changed type
	imgs [][][]float64, // Changed type
	descr []string,
	from int,
	to int,
	epochs int,
	// trainToValidationRatio int, // Removed parameter
	miniBatchSize int,
	baseNumWorkers int,
	initialLR float32,
	initialDecay float32,
	initialL2 float32,
	initialLowCap float32,
	// initialRelu float32, // Removed
	initialMomentum float32,
	// initialBN float32, // Removed
) {
	fmt.Printf("\n--- Starting Training Session (Calculations use float64 internally) ---\n")

	// Create Params for this session
	// NewParamsFull no longer takes useFloat64Calc
	currentParams := neuralnet.NewParamsFull(
		initialLR,
		initialDecay,
		initialL2,
		// initialLowCap, // Removed
		initialMomentum,
	)

	// Use NumClasses constant and derive input size (D*D)
	inputSize := D * D
	nn := neuralnet.NewNeuralNetwork(inputSize, []int{512, 256}, NumClasses, currentParams)
	// Note: Optimizer/Loss logic is currently internal to training/loss functions.

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
		for sample := 0; sample < NumSamplesToSave; sample++ {
			// Select a random validation sample to predict and save
			validationStart := to         // Validation starts after training data
			validationEnd := len(inputs) // Validation ends at the end of the dataset
			if validationStart >= validationEnd { // Skip if validation set is empty
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
			accuracy(nn, inputs, labels, to, len(inputs), numWorkers), // Use full remaining data for validation
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
	fmt.Printf("Total training duration for session: %s\n", totalTrainingDuration)
	fmt.Printf("Average epoch duration for session: %s\n", averageEpochDuration)
	fmt.Println("---")

	// Calculate and print per-class accuracy on the validation set
	calculateAndPrintPerClassAccuracy(nn, inputs, labels, to, len(inputs), numWorkers, descr)
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

	shuffledInputs := make([][]float32, datasetSize)   // Changed type
	shuffledLabels := make([][]float32, datasetSize)   // Changed type
	shuffledColorImgs := make([][][]float64, datasetSize) // Changed type

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

	from := 0
	// Calculate training set size based on ratio
	trainingSetSize := int(float64(datasetSize) * TrainSplitRatio)
	to := trainingSetSize // 'to' is the end index (exclusive) for training data
	epochs := *flagEpochs

	validationSetSize := datasetSize - trainingSetSize // Use remaining data for validation
	miniBatchSize := *flagBatch
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
	initialLR := float32(*flagLR)
	initialDecay := float32(*flagDecay)
	initialL2 := float32(1e-4)
	// initialLowCap := float32(0.0) // Removed
	initialMomentum := float32(0.9)

	baseNumWorkers := *flagWorkers // Use number of workers from command-line flag

	// Run a single training session (UseFloat64 flag is now removed)
	runTrainingSession(
		inputs, labels, imgs, descr,
		from, to, epochs, miniBatchSize, baseNumWorkers,
		initialLR, initialDecay, initialL2, /*initialLowCap,*/ initialMomentum,
	)

	// End of main function
}
