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
   "math"
   "math/rand"
   "os"
   "runtime"
   "runtime/pprof"
   "strings"
   "strconv"
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

// loadCIFAR10 loads a single CIFAR-10 batch file (binary format).
// The binary format consists of 10000 records.
// Each record is 3073 bytes: 1 byte label (0-9), followed by 3072 bytes pixel data.
// Pixel data is 3072 unsigned bytes: 1024 Red, 1024 Green, 1024 Blue (row-major order).
// Returns:
// - [][][]float64: Slice of images. Each image is [][]float64 (3 channels), each channel is []float64 (1024 pixels, 0.0-1.0).
// - []int: Slice of labels (0-9).
// - error: Any error encountered during file reading or processing.
func loadCIFAR10(filePath string) ([][][]float64, []int, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	images := make([][][]float64, Batch) // Pre-allocate for the batch size
	labels := make([]int, Batch)
	data := make([]byte, Batch*Row) // Buffer to read the entire batch

	// Read the entire batch file content
	n, err := io.ReadFull(file, data)
	if err != nil && err != io.ErrUnexpectedEOF && err != io.EOF {
		// Handle unexpected errors during read
		return nil, nil, fmt.Errorf("error reading batch file: %w", err)
	}
	// Calculate how many full records were actually read
	numRecordsRead := n / Row
	if numRecordsRead == 0 && err == io.EOF {
		return nil, nil, fmt.Errorf("empty batch file or read error")
	}


	// Process the records read into the buffer
	for i := 0; i < numRecordsRead; i++ {
		row := data[i*Row : (i+1)*Row]
		labels[i] = int(row[0])

		pixels := row[LabelSize : LabelSize+ImageSize]
		norm := make([]float64, ImageSize)
		for k := 0; k < len(pixels); k++ {
			norm[k] = float64(pixels[k]) / 255.0
		}
		imageChannels := make([][]float64, Colors)
		for k := 0; k < Colors; k++ {
			channel := norm[k*Channel : (k+1)*Channel]
			imageChannels[k] = channel
		}
		images[i] = imageChannels
	}

	// Return slices trimmed to the actual number of records read
	return images[:numRecordsRead], labels[:numRecordsRead], nil
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

// description finds the string label corresponding to a one-hot encoded vector.
func description(encoded []float32, desc []string) string {
	for i := 0; i < len(encoded); i++ {
		if encoded[i] == 1.0 {
			return desc[i]
		}
	}
	panic("No suitable description")
}

// label finds the integer label index corresponding to a one-hot encoded vector.
func label(encoded []float32) int {
	for i := 0; i < len(encoded); i++ {
		if encoded[i] == 1.0 {
			return i
		}
	}
	panic("No suitable idx")
}

// convertImagesToInputs flattens the R, G, B channels of color images.
// Input: images [][][]float64 of shape [numImages][Colors (3)][Channel (1024)]
// Output: inputs [][]float32 of shape [numImages][ImageSize (3072)], where pixels are [R... G... B...]
func convertImagesToInputs(images [][][]float64) [][]float32 {
	numImages := len(images)
	if numImages == 0 {
		return [][]float32{}
	}

	// Validate structure of the first image, assuming consistency
	if len(images[0]) != Colors {
		panic(fmt.Sprintf("convertImagesToInputs: Expected %d color channels, got %d", Colors, len(images[0])))
	}
	if len(images[0][0]) != Channel {
		panic(fmt.Sprintf("convertImagesToInputs: Expected %d pixels per channel, got %d", Channel, len(images[0][0])))
	}

	inputs := make([][]float32, numImages)

	for i := 0; i < numImages; i++ {
		if len(images[i]) != Colors || len(images[i][0]) != Channel || len(images[i][1]) != Channel || len(images[i][2]) != Channel {
			// More robust check for each image
			panic(fmt.Sprintf("convertImagesToInputs: Image %d has inconsistent channel structure", i))
		}
		
		flattenedImage := make([]float32, ImageSize)
		offset := 0
		// R channel
		for _, val := range images[i][0] {
			flattenedImage[offset] = float32(val)
			offset++
		}
		// G channel
		for _, val := range images[i][1] {
			flattenedImage[offset] = float32(val)
			offset++
		}
		// B channel
		for _, val := range images[i][2] {
			flattenedImage[offset] = float32(val)
			offset++
		}
		inputs[i] = flattenedImage
	}
	return inputs
}


// load reads the first CIFAR-10 batch file, performs one-hot encoding on labels.
func load() ([][][]float64, [][]float32) {
	images, labels, err := loadCIFAR10("data/data_batch_1.bin")
	if err != nil {
		fmt.Println("Error loading CIFAR-10:", err)
		panic("Error loading CIFAR-10")
	}
	// Remove redundant declaration: out := oneHotEncode(labels, 10)
	out := oneHotEncode(labels, NumClasses)
	return images, out
}

// accuracy calculates the classification accuracy over a specified range of the dataset (e.g., training or validation set).
// It supports parallel calculation using goroutines and network cloning for thread safety.
func accuracy(nn *neuralnet.NeuralNetwork, allInputs [][]float32, allLabels [][]float32, from int, to int, numWorkers int) float32 {
	numSamplesToTest := to - from
	if numSamplesToTest <= 0 {
		return 0.0
	}

	correctPredictions := 0

	if numWorkers <= 1 || numSamplesToTest < numWorkers { // Fallback to single-threaded
		for i := from; i < to; i++ {
			// Use a clone even for single-threaded case for consistency and safety,
			// as Predict->FeedForward modifies internal network state (neuron outputs).
			currentNN := nn.Clone()
			if label(allLabels[i]) == currentNN.Predict(allInputs[i]) {
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
					if label(allLabels[i]) == workerNN.Predict(allInputs[i]) {
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

// Calculates and prints per-class accuracy for a given slice range (e.g., validation set)
func calculateAndPrintPerClassAccuracy(nn *neuralnet.NeuralNetwork, allInputs [][]float32, allLabels [][]float32, from int, to int, numWorkers int, descr []string) {
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
			trueLabelIdx := label(allLabels[i])
			predLabelIdx := currentNN.Predict(allInputs[i])

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
					trueLabelIdx := label(allLabels[i])
					// Predict modifies the clone's internal state (neuron outputs) via FeedForward.
					predLabelIdx := workerNN.Predict(allInputs[i])

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
    flagL2      = flag.Float64("l2", 1e-4, "L2 regularization factor")
    flagEpochs  = flag.Int("epochs", NumEpochs, "number of epochs")
    flagBatch   = flag.Int("batch", MiniBatchSize, "mini-batch size")
    flagWorkers    = flag.Int("workers", runtime.NumCPU(), "number of workers")
    flagMomentum   = flag.Float64("momentum", 0.9, "momentum coefficient")
    flagSaveModel  = flag.String("save", "", "path to save best model during training")
    flagLoadModel  = flag.String("load", "", "path to load saved (best) model for evaluation")
    flagPreprocess = flag.String("preprocess", "none", "data preprocessing method: none, normalize, standardize")
    flagHiddenLayers = flag.String("hiddenlayers", "512,256", "comma-separated list of hidden layer sizes")
    flagHiddenActivation = flag.String("hiddenactivation", "relu", "activation function for hidden layers (relu, sigmoid, tanh, leakyrelu, linear)")
    flagOutputActivation = flag.String("outputactivation", "linear", "activation function for output layer (relu, sigmoid, tanh, leakyrelu, linear)")
    flagDropoutRate = flag.Float64("dropoutrate", 0.0, "dropout rate for hidden layers (0.0 to disable)")
    flagBatchNorm = flag.Bool("batchnorm", false, "enable batch normalization for hidden layers")
)
var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

// --- Data Preprocessing Functions ---

// calculateMeanStddevPerFeature calculates the mean and standard deviation for each feature (pixel)
// across the training dataset.
// Input: trainingImages [][][]float64 (shape: [numImages][Colors (3)][Channel (1024)])
// Output: means []float64, stddevs []float64 (each of length ImageSize=3072)
func calculateMeanStddevPerFeature(trainingImages [][][]float64) ([]float64, []float64) {
	if len(trainingImages) == 0 {
		return nil, nil
	}
	numTrainImages := len(trainingImages)
	means := make([]float64, ImageSize)
	stddevs := make([]float64, ImageSize)
	epsilon := 1e-5 // To prevent division by zero in stddev

	// Accumulate sum and sum of squares
	for _, imgChannels := range trainingImages {
		// Flatten the image: R, G, B
		flattenedImg := make([]float64, ImageSize)
		offset := 0
		for c := 0; c < Colors; c++ {
			for p := 0; p < Channel; p++ {
				flattenedImg[offset] = imgChannels[c][p]
				offset++
			}
		}

		for i := 0; i < ImageSize; i++ {
			means[i] += flattenedImg[i]
		}
	}

	// Calculate mean
	for i := 0; i < ImageSize; i++ {
		means[i] /= float64(numTrainImages)
	}

	// Accumulate sum of squared differences from mean
	for _, imgChannels := range trainingImages {
		flattenedImg := make([]float64, ImageSize)
		offset := 0
		for c := 0; c < Colors; c++ {
			for p := 0; p < Channel; p++ {
				flattenedImg[offset] = imgChannels[c][p]
				offset++
			}
		}
		for i := 0; i < ImageSize; i++ {
			diff := flattenedImg[i] - means[i]
			stddevs[i] += diff * diff
		}
	}

	// Calculate standard deviation
	for i := 0; i < ImageSize; i++ {
		stddevs[i] = math.Sqrt(stddevs[i]/float64(numTrainImages)) + epsilon
	}

	return means, stddevs
}

// normalizePixelValues scales pixel values from [0, 1] to [-1, 1].
// Assumes input images are already in [0, 1] range.
func normalizePixelValues(images [][][]float64) [][][]float64 {
	processedImages := make([][][]float64, len(images))
	for i, imgChannels := range images {
		processedChannels := make([][]float64, Colors)
		for c, channelPixels := range imgChannels {
			processedPixelValues := make([]float64, len(channelPixels))
			for p, val := range channelPixels {
				processedPixelValues[p] = (val * 2.0) - 1.0
			}
			processedChannels[c] = processedPixelValues
		}
		processedImages[i] = processedChannels
	}
	return processedImages
}

// standardizePixelValues applies (pixel - mean) / stddev to each pixel feature.
func standardizePixelValues(images [][][]float64, means []float64, stddevs []float64) [][][]float64 {
	if means == nil || stddevs == nil || len(means) != ImageSize || len(stddevs) != ImageSize {
		panic("standardizePixelValues: Invalid means or stddevs provided")
	}
	processedImages := make([][][]float64, len(images))
	for i, imgChannels := range images {
		processedChannels := make([][]float64, Colors)
		featureIndex := 0
		for c := 0; c < Colors; c++ {
			processedPixelValues := make([]float64, Channel)
			for p := 0; p < Channel; p++ {
				val := imgChannels[c][p]
				processedPixelValues[p] = (val - means[featureIndex]) / stddevs[featureIndex]
				featureIndex++
			}
			processedChannels[c] = processedPixelValues
		}
		processedImages[i] = processedChannels
	}
	return processedImages
}

// applyPreprocessing dispatches to the appropriate preprocessing function.
func applyPreprocessing(allImages [][][]float64, trainingSplitEnd int, method string) [][][]float64 {
	fmt.Printf("Applying preprocessing method: %s\n", method)
	switch method {
	case "normalize":
		return normalizePixelValues(allImages)
	case "standardize":
		if trainingSplitEnd <= 0 || trainingSplitEnd > len(allImages) {
			panic("applyPreprocessing: Invalid trainingSplitEnd for standardization")
		}
		trainingImages := allImages[:trainingSplitEnd]
		means, stddevs := calculateMeanStddevPerFeature(trainingImages)
		fmt.Printf("Calculated means (first 3): %v, stddevs (first 3): %v for standardization.\n", means[:3], stddevs[:3])
		return standardizePixelValues(allImages, means, stddevs)
	case "none":
		return allImages
	default:
		fmt.Printf("Warning: Unknown preprocessing method '%s'. No preprocessing will be applied.\n", method)
		return allImages
	}
}

// --- End Data Preprocessing Functions ---

func runTrainingSession(
	inputs [][]float32,
	labels [][]float32,
	imgs [][][]float64,
	descr []string,
	from int,
	to int,
	epochs int,
	miniBatchSize int,
	baseNumWorkers int,
	initialLR float32,
	initialDecay float32,
	initialL2 float32,
	initialMomentum float32,
	dropoutRate float32,
	enableBatchNorm bool, // Added enableBatchNorm parameter
	savePath string,
	hiddenLayerConfigStr string,
	hiddenActivationStr string,
	outputActivationStr string,
) {
	fmt.Printf("\n--- Starting Training Session (Calculations use float64 internally) ---\n")
	bestValAcc := float32(0.0)

	// Get activation functions
	hiddenActivation, err := neuralnet.GetActivationFunction(hiddenActivationStr)
	if err != nil {
		fmt.Printf("Error getting hidden activation function '%s': %v. Using ReLU as default.\n", hiddenActivationStr, err)
		hiddenActivation, _ = neuralnet.GetActivationFunction("relu") // Fallback
	}
	outputActivation, err := neuralnet.GetActivationFunction(outputActivationStr)
	if err != nil {
		fmt.Printf("Error getting output activation function '%s': %v. Using Linear as default.\n", outputActivationStr, err)
		outputActivation, _ = neuralnet.GetActivationFunction("linear") // Fallback
	}
	fmt.Printf("Using Hidden Activation: %s, Output Activation: %s\n", hiddenActivationStr, outputActivationStr)

	// Parse hidden layer configuration
	hiddenLayerSizes := []int{}
	if hiddenLayerConfigStr != "" {
		parts := strings.Split(hiddenLayerConfigStr, ",")
		for _, part := range parts {
			size, err := strconv.Atoi(strings.TrimSpace(part))
			if err != nil || size <= 0 {
				// Handle error: log and fall back to a default or panic
				fmt.Printf("Error parsing hidden layer size '%s': %v. Using default [512, 256].\n", part, err)
				hiddenLayerSizes = []int{512, 256} // Default fallback
				break
			}
			hiddenLayerSizes = append(hiddenLayerSizes, size)
		}
	}
	if len(hiddenLayerSizes) == 0 { // If string was empty or all parts failed
		fmt.Println("No valid hidden layer sizes provided or string was empty, using default [512, 256].")
		hiddenLayerSizes = []int{512, 256}
	}
	fmt.Printf("Using hidden layers: %v\n", hiddenLayerSizes)

	// Create Params for this session
	// NewParamsFull no longer takes useFloat64Calc
	currentParams := neuralnet.NewParamsFull(
		initialLR,
		initialDecay,
		initialL2,
		initialMomentum,
		dropoutRate,
		enableBatchNorm, // Pass enableBatchNorm to NewParamsFull
	)

	// Use NumClasses constant and derive input size (ImageSize = D*D*Colors = 32*32*3 = 3072 for color)
	inputSize := ImageSize // This is 3072
	nn := neuralnet.NewNeuralNetwork(inputSize, hiddenLayerSizes, NumClasses, currentParams, hiddenActivation, outputActivation)

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
		// Set IsTraining to true for the training phase
		nn.Params.IsTraining = true
		nn.TrainMiniBatch(inputs[from:to], labels[from:to], miniBatchSize, 1, numWorkers)
		// Set IsTraining to false for evaluation/prediction phases
		nn.Params.IsTraining = false

		for sample := 0; sample < NumSamplesToSave; sample++ {
			// Select a random validation sample to predict and save
			validationStart := to         // Validation starts after training data
			validationEnd := len(inputs) // Validation ends at the end of the dataset
			if validationStart >= validationEnd { // Skip if validation set is empty
				fmt.Println("Skipping image saving due to invalid validation range.")
				break
			}
			j = validationStart + rand.Intn(validationEnd-validationStart)
			// IsTraining is already false here
			pred := nn.Predict(inputs[j])
			saveImg(imgs, labels, descr, j, pred)
		}
		// IsTraining is already false here
		trainAcc := accuracy(nn, inputs, labels, from, to, numWorkers)
		valAcc := accuracy(nn, inputs, labels, to, len(inputs), numWorkers) // Use full remaining data for validation
		fmt.Printf("Train accuracy: %.2f, Validation accuracy: %.2f\n", trainAcc, valAcc)
		if savePath != "" && valAcc > bestValAcc {
			// Save model with IsTraining = false
			nn.Save(savePath)
			bestValAcc = valAcc
		}
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
		f, err := os.Create(*cpuprofile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating CPU profile file %s: %v\n", *cpuprofile, err)
			os.Exit(1)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	if *flagLoadModel != "" {
		nn := neuralnet.LoadModel(*flagLoadModel)
		nn.Params.IsTraining = false // Ensure IsTraining is false for loaded model evaluation
		imgs, labels := load()
		descr := readLabels()
		// Apply preprocessing to loaded images if specified (or use defaults if not)
		// For simplicity, we'll assume if a model is loaded, preprocessing matching its training should be applied.
		// This might require saving preprocessing stats with the model or having a consistent setup.
		// For now, just apply the flag-specified preprocessing.
		preprocessMethodEval := *flagPreprocess // Use the same flag for now
		// Note: trainingSetSize for a loaded model context is ambiguous here.
		// For standardization, ideally, we'd use the stats from the original training.
		// Using the whole loaded dataset (len(imgs)) to calculate stats is incorrect if it includes test data.
		// Safest is to require 'none' or 'normalize' or have saved stats.
		// For this example, we'll proceed but acknowledge this limitation.
		if preprocessMethodEval == "standardize" {
			fmt.Println("Warning: Applying standardization to loaded model based on current dataset's stats. Ideally, use original training stats.")
		}
		imgs = applyPreprocessing(imgs, len(imgs), preprocessMethodEval)


		inputs := convertImagesToInputs(imgs)
		numWorkers := *flagWorkers
		if numWorkers > neuralnet.MAX_WORKERS {
			numWorkers = neuralnet.MAX_WORKERS
		}
		totalAcc := accuracy(nn, inputs, labels, 0, len(inputs), numWorkers)
		fmt.Printf("Overall accuracy: %.2f\n", totalAcc)
		calculateAndPrintPerClassAccuracy(nn, inputs, labels, 0, len(inputs), numWorkers, descr)
		return
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
	// inputs = shuffledInputs // This is grayscale, will be recalculated later
	labels = shuffledLabels
	imgs = shuffledColorImgs // This is [][][]float64 color images

	fmt.Printf("Total samples loaded: %d\n", len(imgs)) // Use len(imgs) as inputs is not yet defined

	from := 0
	// Calculate training set size based on ratio
	trainingSetSize := int(float64(datasetSize) * TrainSplitRatio)
	
	// Apply preprocessing
	// The 'imgs' variable (shuffledColorImgs) is [][][]float64
	// trainingSetSize is the count of training images, used as trainingSplitEnd for applyPreprocessing
	preprocessMethod := *flagPreprocess
	imgs = applyPreprocessing(imgs, trainingSetSize, preprocessMethod) // imgs is now preprocessed [][][]float64

	// Now convert preprocessed images to the format expected by the neural network
	inputs = convertImagesToInputs(imgs) // This converts [][][]float64 to [][]float32

	to := trainingSetSize // 'to' is the end index (exclusive) for training data
	epochs := *flagEpochs

	validationSetSize := datasetSize - trainingSetSize // Use remaining data for validation
	miniBatchSize := *flagBatch
	fmt.Printf("Training set size: %d samples\n", trainingSetSize)
	fmt.Printf("Validation set size: %d samples\n", validationSetSize)
	fmt.Printf("Number of main epochs: %d\n", epochs)
	fmt.Printf("Mini-batch size: %d\n", miniBatchSize)
	// Fetch default parameter values to use as base for both sessions
	// These are: lr: 0.01, decay: 0.95, L2: 1e-4, lowCap: 0, relu: 0, momentum: 0.9, bn: 0.0
	initialLR := float32(*flagLR)
	initialDecay := float32(*flagDecay)
	initialL2 := float32(*flagL2)
	initialMomentum := float32(*flagMomentum)
	initialDropoutRate := float32(*flagDropoutRate)
	enableBatchNorm := *flagBatchNorm // Get batch norm flag
	fmt.Printf("Initial LR: %.5f, Decay: %.5f, L2: %.6f, Momentum: %.5f, Dropout: %.2f, BatchNorm: %t\n", initialLR, initialDecay, initialL2, initialMomentum, initialDropoutRate, enableBatchNorm)


	baseNumWorkers := *flagWorkers // Use number of workers from command-line flag

	// Run a single training session
	// Note: 'inputs' is now the preprocessed data, 'imgs' is also preprocessed but still in [][][]float64 for saving
	runTrainingSession(
		inputs, labels, imgs, descr,
		from, to, epochs, miniBatchSize, baseNumWorkers,
		initialLR, initialDecay, initialL2, initialMomentum, initialDropoutRate, enableBatchNorm, *flagSaveModel, *flagHiddenLayers, *flagHiddenActivation, *flagOutputActivation,
	)

	// End of main function
}
