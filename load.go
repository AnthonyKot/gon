package main

import "gon/neuralnet"

import (
	"bufio"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"

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

func readLabels() ([]string, error) {
	file, err := os.Open("data/batches.meta.txt")
	if err != nil {
		fmt.Println("Error opening file:", err)
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var words []string
	for scanner.Scan() {
		words = append(words, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("Error reading file:", err)
	}
	return words, nil
}

func saveImg(ts [][]mat.Dense, ws []string, ls []int, i int) {
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
	label := fmt.Sprintf("file_%s_%d.png", ws[ls[i]], i)
	file, err := os.Create(label)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	err = png.Encode(file, img)
	if err != nil {
		panic(err)
	}

	println(fmt.Sprintf("Image saved as %s", label))
}

func oneHotEncode(labels []int, numClasses int) *mat.Dense {
	numLabels := len(labels)
	norm := make([]float64, numLabels*numClasses)

	for i, label := range labels {
		norm[i * numClasses + label] = 1.0
	}

	return mat.NewDense(numLabels, numClasses, norm)
}

func load() {
	images, labels, err := loadCIFAR10("data/data_batch_1.bin")
	if err != nil {
		fmt.Println("Error loading CIFAR-10:", err)
		return
	}
	words, err := readLabels()
	if err != nil {
		fmt.Println("Error loading labels:", err)
		return
	}
	saveImg(images, words, labels, 3)
	out := oneHotEncode(labels, 10)
	fmt.Println(out)
}

func main() {
	// input layer + 1 hidden layer + output layer
	nn := neuralnet.NewNeuralNetwork(2, []int{2}, 2)
	nn.FeedForward([]float32{1.0, 0})
	target := mat.NewVecDense(2, []float64{1.0, 0})
	fmt.Println("Output:", nn.CalculateLoss(target))
	fmt.Println("Before BP:", nn)
	nn.Backpropagate(target)
	fmt.Println("After BP:", nn)
}