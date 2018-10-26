package mnist

import (
	"bufio"
	"encoding/csv"
	"log"
	"os"
	"strconv"

	"github.com/haakonleg/neural/dataset"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type MnistDataset struct {
	f         *os.File
	csvReader *csv.Reader
}

func NewMnistDatasetFromFile(file string) *MnistDataset {
	md := &MnistDataset{}

	f, err := os.Open(file)
	if err != nil {
		log.Fatal(err)
	}

	md.f = f
	md.csvReader = csv.NewReader(bufio.NewReader(md.f))

	return md
}

func (md *MnistDataset) Close() {
	md.f.Close()
}

// There are 784 pixels in the images of handwritten numbers
func (md *MnistDataset) NumInputs() int {
	return 784
}

// There are 10 different labels, numbers from 0 to 9
func (md *MnistDataset) NumLabels() int {
	return 10
}

// Retrieve next sample data from the dataset
func (md *MnistDataset) NextSample() *dataset.DataSample {
	line, err := md.csvReader.Read()
	if line == nil || err != nil {
		return nil
	}

	// The first number is the label of the number
	label, _ := strconv.Atoi(line[0])

	// Rest of the numbers are grayscale pixels, convert to float64
	pixels := line[1:]
	pixelNums := make([]float64, len(pixels))
	for i := range pixels {
		pixelNums[i], _ = strconv.ParseFloat(pixels[i], 64)
	}

	// Rescale the pixel numbers so they are in the range 0.01-1.0
	for i := range pixelNums {
		pixelNums[i] = (pixelNums[i] / 255 * 0.99) + 0.01
	}

	return &dataset.DataSample{
		Label:     label,
		InputVec:  mat.NewVecDense(md.NumInputs(), pixelNums),
		TargetVec: mat.NewVecDense(md.NumLabels(), md.targetVec(label))}
}

func (md *MnistDataset) targetVec(label int) []float64 {
	vec := make([]float64, md.NumLabels())

	// Set all elements to 0.01
	floats.AddConst(0.01, vec)
	vec[label] = 0.99
	return vec
}

func (md *MnistDataset) Reset() {
	md.f.Seek(0, 0)
}
