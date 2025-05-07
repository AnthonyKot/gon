package main

import (
	"gon/neuralnet"
	"bufio"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"
	"math/rand"

	"flag"
	"runtime"
	"runtime/pprof"

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
			row := data[i * Row : (i + 1) * Row]
			labels[i] = int(row[0])

			pixels := row[LabelSize : LabelSize + ImageSize]
			norm := make([]float64, ImageSize)
			for i := 0; i < len(pixels); i++ { // Corrected loop to start from 0
				norm[i] = float64(pixels[i]) / 255.0
			}
			image := make([]mat.Dense, 3)
			for i := 0; i < Colors; i++ {
				channel := norm[i * Channel : (i + 1) * Channel]
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

func saveImg(ts [][]mat.Dense, ls []mat.VecDense, ws []string, i int, j int) {
	t := ts[i]
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
	ws_idx := 0
	for j:= 0; j < ls[i].Len(); j++ {
		if ls[i].AtVec(j) == 1.0 {
			ws_idx = j
		}
	}
	label := fmt.Sprintf("file_%s_%d_%s.png", ws[ws_idx], i, ws[j])
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
            grayImage.Set(i, j, gray) // Set the normalized float64 value directly

        }
    }

    return *grayImage
}

func flaten(image mat.Dense) mat.VecDense {
	rows, cols := image.Dims()
	vec := mat.NewVecDense(rows * cols, nil)
	for i:= 0; i < rows; i++ {
		for j:= 0; j < cols; j++ {
			vec.SetVec(i * cols + j, image.At(i, j))
		}
	}
	return *vec
}

func converImagesToInputs(images [][]mat.Dense) []mat.VecDense {
	inputs := make([]mat.VecDense, len(images))
	for i := 0; i < len(inputs); i++ {
		bwImage := RGBToBlackWhite(images[i]) // Convert to BW first
		inputs[i] = flaten(bwImage)           // Then flatten
	}
	return inputs
}

func finMaxIdx(output []float32) int {
	index := 0
	max := output[0]
	for i := 0; i < len(output); i++ {
		if max < output[i] {
			max = output[i];
			index = i;
		}
	}
	return index
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

func accuracy(nn *neuralnet.NeuralNetwork, trainingData []mat.VecDense, expectedOutputs []mat.VecDense, from int, to int) float32 {
	accuracy := 0
	for i := from; i < to; i ++ {
		if label(expectedOutputs[i]) == nn.Predict(trainingData[i]) {
			accuracy++
		}
	}
	return float32(accuracy) / float32(to - from)
}

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

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
	inputs := converImagesToInputs(imgs)

	// Shuffle the entire dataset (inputs and labels consistently) before splitting
	// to ensure training/validation batches are representative.
	if len(inputs) != len(labels) {
		panic("Mismatch between number of inputs and labels before shuffling.")
	}
	datasetSize := len(inputs)
	permutation := rand.Perm(datasetSize)

	shuffledInputs := make([]mat.VecDense, datasetSize)
	shuffledLabels := make([]mat.VecDense, datasetSize)

	for i := 0; i < datasetSize; i++ {
		shuffledInputs[i] = inputs[permutation[i]]
		shuffledLabels[i] = labels[permutation[i]]
	}
	// Use the shuffled data from now on
	inputs = shuffledInputs
	labels = shuffledLabels

	fmt.Printf("Total samples loaded: %d\n", len(inputs))

	// Tanh works the best without Jacobian calculation
	from := 0
	to := from + 8000
	epochs := 10
	train_to_validation := 4

	trainingSetSize := to - from
	validationSetSize := (to - from) / train_to_validation
	fmt.Printf("Training set size: %d samples\n", trainingSetSize)
	fmt.Printf("Validation set size: %d samples\n", validationSetSize)
	fmt.Printf("Number of main epochs: %d\n", epochs)
	fmt.Printf("Samples processed per call to nn.TrainBatch (effectively the batch size for gradient update): %d\n", trainingSetSize)
	fmt.Println("---")

	nn := neuralnet.DefaultNeuralNetwork(1024, []int{512, 256}, 10)
	j := 0
	for i := 0; i < epochs; i++ {
		nn.TrainBatch(inputs[from:to], labels[from:to], 1)
		for sample := 0; sample < 3; sample++ {
			j = to + rand.Intn((to - from) / train_to_validation)
			pred := nn.Predict(inputs[j])
			saveImg(imgs, labels, descr, j, pred)
			fmt.Printf("Epoch %d, Sample %d, Output: %v\n", i, sample, nn.Output())
		}
		fmt.Printf("Train accuracy: %.2f, Validation accuracy: %.2f\n",
			accuracy(nn, inputs, labels, from, to),
			accuracy(nn, inputs, labels, to, to+((to-from)/train_to_validation)),
		)
		fmt.Println()
	}


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
