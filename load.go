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

	"flag"
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
			for i := 1; i < len(pixels); i++ {
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

func saveImg(ts [][]mat.Dense, ls []mat.VecDense, ws []string, i int) {
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
	label := fmt.Sprintf("file_%s_%d.png", ws[ws_idx], i)
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
            gray := 0.299*r + 0.587*g + 0.114*b
            grayImage.Set(i, j, gray)
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
		inputs[i] = flaten(RGBToBlackWhite(images[i]))
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
		nn.FeedForward(trainingData[i])
		if label(expectedOutputs[i]) == finMaxIdx(nn.Output()) {
			accuracy++
		}
	}
	return float32(accuracy) / float32(to - from)
}

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

func main() {
    flag.Parse()
    if *cpuprofile != "" {
        f, _ := os.Create(*cpuprofile)
        pprof.StartCPUProfile(f)
        defer pprof.StopCPUProfile()
    }
	imgs, labels := load()
	descr := readLabels()
	saveImg(imgs, labels, descr, 3)
	// input layer + 1 hidden layer + output layer
	// around 250k params ~ 1000*250
	nn := neuralnet.NewNeuralNetwork(1024, []int{512, 256}, 10, 0.001, 0.0001)
	// Tanh works the best without Jacobian calculation
	inputs := converImagesToInputs(imgs)
	// 1 * 10 epochs
	from := 10
	to := from + 100
	epochs := 5
	for i := 0; i < epochs; i ++ {
		nn.Train(inputs[from:to], labels[from:to], 10)
		fmt.Println("train", accuracy(nn, inputs, labels, from, to))
		fmt.Println("validation", accuracy(nn, inputs, labels, to, to + to - from))
	}
	// profiling top 5
	// flat  flat%   sum%        cum   cum%
    // 21.03s 24.42% 24.42%     21.98s 25.52%  gon/neuralnet.(*NeuralNetwork).UpdateWeights
    // 17.23s 20.00% 44.42%     18.41s 21.37%  gon/neuralnet.ConvertWeightsDense
    // 14.96s 17.37% 61.79%     15.27s 17.73%  gon/neuralnet.(*NeuralNetwork).CalculateLoss
    // 10.21s 11.85% 73.64%     10.22s 11.87%  gonum.org/v1/gonum/mat.(*VecDense).at (inline)
    //  9.91s 11.51% 85.15%     22.71s 26.37%  gon/neuralnet.(*NeuralNetwork).FeedForward
}