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

func load() ([][]mat.Dense, []mat.VecDense) {
	images, labels, err := loadCIFAR10("data/data_batch_1.bin")
	if err != nil {
		fmt.Println("Error loading CIFAR-10:", err)
		panic("Error loading CIFAR-10")
	}
	out := oneHotEncode(labels, 10)
	return images, out
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
	nn := neuralnet.NewNeuralNetwork(1024, []int{256, 128}, 10, 0.01, 0.001)
	// around 250k params ~ 1000*250
	nn.SetActivation(2, neuralnet.Tanh {})
	input := flaten(RGBToBlackWhite(imgs[0]))
	target := labels[0]
	fmt.Println(target)
	for i := 0; i < 2; i++ {
		nn.FeedForward(input)
		nn.Backpropagate(target, false)
	}
	fmt.Println(nn)

	// profiling top 10
	// 	  flat  flat%   sum%        cum   cum%
	//    620ms 23.57% 23.57%      630ms 23.95%  gon/neuralnet.(*NeuralNetwork).UpdateWeights
	//    570ms 21.67% 45.25%      580ms 22.05%  gon/neuralnet.(*NeuralNetwork).FeedForward
	//    490ms 18.63% 63.88%      550ms 20.91%  gon/neuralnet.ConvertWeightsDense
	//    260ms  9.89% 73.76%      260ms  9.89%  runtime.kevent
	//    220ms  8.37% 82.13%      220ms  8.37%  syscall.syscall
	//    140ms  5.32% 87.45%      140ms  5.32%  runtime.asyncPreempt
	// 	  60ms  2.28% 89.73%       60ms  2.28%  runtime.madvise
	// 	  50ms  1.90% 91.63%       50ms  1.90%  runtime.memclrNoHeapPointers
	// 	  30ms  1.14% 92.78%      130ms  4.94%  gon/neuralnet.(*NeuralNetwork).CalculateLoss
	// 	  30ms  1.14% 93.92%       30ms  1.14%  internal/runtime/atomic.(*UnsafePointer).Load (inline)
}