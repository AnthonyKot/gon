package main

import (
    "fmt"
    "io"
    "os"
	"bufio"
    "image"
    "image/color"
	"image/png"

    "gorgonia.org/tensor"
)

const (
    ImageSize = 32 * 32 * 3
    LabelSize = 1
	Row = LabelSize + ImageSize
	Batch = 10000
)

func loadCIFAR10(filePath string) ([]tensor.Tensor, []int, error) {
    file, err := os.Open(filePath)
    if err != nil {
        return nil, nil, err
    }
    defer file.Close()

	images := make([]tensor.Tensor, 0)
	labels := make([]int, 0)
	for {
		data := make([]byte, Batch * Row)
		_, err := io.ReadFull(file, data)
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, nil, err
		}
		for i := 0; i < Batch; i++ {
			row := data[i * Row: (i + 1)* Row]
			labels = append(labels, int(row[0]))

			img := row[LabelSize : LabelSize + ImageSize]
			norm := make([]float32, ImageSize)
			for i := 1; i < len(img); i++ {
				norm[i] = float32(img[i]) / 255.0
			}
			t := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(3, 32, 32), tensor.WithBacking(norm))
			images = append(images, t)
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

func saveImg(ts []tensor.Tensor, ws []string, ls []int, i int) {
	t := ts[i]
	img := image.NewRGBA(image.Rect(0, 0, 32, 32))
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			r, _ := t.At(0, y, x)
			r8 := uint8(r.(float32) * 255.0)
			g, _ := t.At(1, y, x)
			g8 := uint8(g.(float32) * 255.0)
			b, _ := t.At(2, y, x)
			b8 := uint8(b.(float32) * 255.0)
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

func oneHotEncode(labels []int, numClasses int) tensor.Tensor {
    numLabels := len(labels)
    norm := make([]float32, numLabels * numClasses)

    for i, label := range labels {
        norm[i * numClasses + label] = 1.0
    }

    return tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(numLabels, numClasses), tensor.WithBacking(norm))
}

func main() {
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
	fmt.Println(oneHotLabels)
}
