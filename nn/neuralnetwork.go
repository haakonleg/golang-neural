package nn

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/haakonleg/neural/dataset"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type Settings struct {
	InputNodes   int
	OutputNodes  int
	LearningRate float64
	HiddenLayers []HiddenLayer
}

type HiddenLayer struct {
	Nodes          int
	ActivationFunc ActivationFunction
}

type Weights struct {
	LNodes  int
	RNodes  int
	ActFunc ActivationFunction

	m         *mat.Dense
	actFunc   activationFunction
	derivFunc activationFunctionD
}

func newRandomWeights(lNodes, rNodes int, activationFunc ActivationFunction) Weights {
	rand.Seed(time.Now().UnixNano())
	data := make([]float64, rNodes*lNodes)
	for i := range data {
		data[i] = rand.NormFloat64() * math.Sqrt(2/float64(lNodes+rNodes))
	}

	actFunc, derivFunc := activationFunc.Funcs()

	return Weights{
		LNodes:    lNodes,
		RNodes:    rNodes,
		ActFunc:   activationFunc,
		m:         mat.NewDense(rNodes, lNodes, data),
		actFunc:   actFunc,
		derivFunc: derivFunc}
}

type NeuralNetwork struct {
	InputNodes   int
	OutputNodes  int
	LearningRate float64

	// The neural network weights are represented in terms of matrices of size (n(l+1), n(l)) where n(l+1) is the number of nodes
	// in the next layer and n(l) is the number of nodes in the previous layer
	weights []Weights
}

// NewNeuralNetwork initializes a new neural network object with a specified number of input nodes, hidden nodes, output nodes,
// learning rate, and number of hidden layers. It then assigns random weights.
func NewNeuralNetwork(settings Settings) *NeuralNetwork {
	// Check values
	if settings.InputNodes < 1 {
		log.Fatal("Need at least 1 input node")
	}
	if settings.OutputNodes < 1 {
		log.Fatal("Need at least 1 output node")
	}
	if settings.LearningRate > 100 || settings.LearningRate < 0.000001 {
		log.Fatal("Learning rate cannot be more than 100 or less than 0.000001")
	}
	if len(settings.HiddenLayers) == 0 {
		log.Fatal("Need at least one hidden layer")
	}

	nn := &NeuralNetwork{
		InputNodes:   settings.InputNodes,
		OutputNodes:  settings.OutputNodes,
		LearningRate: settings.LearningRate,
		weights:      make([]Weights, len(settings.HiddenLayers)+1)}

	// init input-hidden weights
	nn.weights[0] = newRandomWeights(nn.InputNodes, settings.HiddenLayers[0].Nodes, settings.HiddenLayers[0].ActivationFunc)
	// init hidden-hidden weights
	for i := 0; i < len(settings.HiddenLayers)-1; i++ {
		lNodes := settings.HiddenLayers[i].Nodes
		rNodes := settings.HiddenLayers[i+1].Nodes
		actFunc := settings.HiddenLayers[i+1].ActivationFunc

		nn.weights[i+1] = newRandomWeights(lNodes, rNodes, actFunc)
	}
	// init hidden-output weights, use softmax
	nn.weights[len(nn.weights)-1] = newRandomWeights(settings.HiddenLayers[len(settings.HiddenLayers)-1].Nodes, nn.OutputNodes, SoftMax)

	return nn
}

// FeedForward sends an input vector through the neural network, and returns the output vector
func (nn *NeuralNetwork) FeedForward(input *mat.VecDense) *mat.VecDense {
	if input.Len() != nn.InputNodes {
		log.Fatal("Number of input values not equal to number of input nodes")
	}

	hiddenInput := mat.VecDenseCopyOf(input)

	// Pass through hidden layers
	for i := 0; i < len(nn.weights); i++ {
		hiddenOutput := mat.NewVecDense(nn.weights[i].RNodes, nil)
		hiddenOutput.MulVec(nn.weights[i].m, hiddenInput)
		nn.weights[i].actFunc(hiddenOutput)

		hiddenInput = hiddenOutput
	}

	return hiddenInput
}

// Train trains the neural network
func (nn *NeuralNetwork) Train(input, target *mat.VecDense, outputs, errors, adjVec []*mat.VecDense, weightDeltas []*mat.Dense) {
	last := len(nn.weights) - 1

	// Feed forward
	for i := 0; i < len(nn.weights); i++ {
		if i > 0 {
			outputs[i].MulVec(nn.weights[i].m, outputs[i-1])
		} else {
			outputs[i].MulVec(nn.weights[i].m, input)
		}
		nn.weights[i].actFunc(outputs[i])
	}

	// Backpropagation

	// Calulate errors
	errors[last].SubVec(target, outputs[last])
	for i := len(nn.weights) - 2; i >= 0; i-- {
		errors[i].MulVec(nn.weights[i+1].m.T(), errors[i+1])
	}

	// Update weights
	for i := len(nn.weights) - 1; i >= 0; i-- {
		for j := 0; j < outputs[i].Len(); j++ {
			adjVec[i].SetVec(j, errors[i].AtVec(j)*nn.weights[i].derivFunc(outputs[i].AtVec(j)))
		}

		if i > 0 {
			weightDeltas[i].Mul(adjVec[i], outputs[i-1].T())
		} else {
			weightDeltas[i].Mul(adjVec[i], input.T())
		}
		weightDeltas[i].Scale(nn.LearningRate, weightDeltas[i])
		nn.weights[i].m.Add(nn.weights[i].m, weightDeltas[i])
	}
}

// Test tests the neural network using a dataset object
func (nn *NeuralNetwork) Test(ds dataset.Dataset) []int {
	if nn.InputNodes != ds.NumInputs() {
		log.Fatal("Number of inputs does not equal number of input nodes")
	}
	if nn.OutputNodes != ds.NumLabels() {
		log.Fatal("Number of labels does not equal number of output nodes")
	}

	rightCnt := 0
	wrongCnt := 0

	results := make([]float64, nn.OutputNodes)
	labelResults := make([]int, 0)

	// Get data samples from the dataset as long as they are available
	sample := ds.NextSample()
	for sample != nil {
		// Query the network
		outputVec := nn.FeedForward(sample.InputVec)

		// Find which result we got and add it to our array
		mat.Col(results, 0, outputVec)
		result := floats.MaxIdx(results)
		labelResults = append(labelResults, result)

		if result == sample.Label {
			rightCnt++
		} else {
			wrongCnt++
		}

		sample = ds.NextSample()
	}

	// Print stats
	res := float64(rightCnt) / float64(rightCnt+wrongCnt) * 100
	fmt.Printf("Percent correct: %0.2f%%\n", res)

	ds.Close()
	return labelResults
}

// TrainWithDataset trains the neural network using a TrainingSet object
func (nn *NeuralNetwork) TrainWithDataset(ds dataset.Dataset, batchSize, iterations int) {
	if nn.InputNodes != ds.NumInputs() {
		log.Fatal("Number of inputs does not equal number of input nodes")
	}
	if nn.OutputNodes != ds.NumLabels() {
		log.Fatal("Number of labels does not equal number of output nodes")
	}
	if iterations < 1 {
		log.Fatal("Number of iterations must be more than 0")
	}
	tStart := time.Now()

	// Allocate data
	outputs := make([]*mat.VecDense, len(nn.weights))
	errors := make([]*mat.VecDense, len(nn.weights))
	adjVec := make([]*mat.VecDense, len(nn.weights))
	weightDeltas := make([]*mat.Dense, len(nn.weights))

	for i := 0; i < len(nn.weights); i++ {
		outputs[i] = mat.NewVecDense(nn.weights[i].RNodes, nil)
		errors[i] = mat.NewVecDense(nn.weights[i].RNodes, nil)
		adjVec[i] = mat.NewVecDense(nn.weights[i].RNodes, nil)
		weightDeltas[i] = mat.NewDense(nn.weights[i].RNodes, nn.weights[i].LNodes, nil)
	}
	errors[len(errors)-1] = mat.NewVecDense(nn.OutputNodes, nil)

	for it := 0; it < iterations; it++ {
		ds.Reset()
		dataChan := make(chan *dataset.DataSample, batchSize)
		go dataFetchAsync(ds, dataChan, batchSize)

		for sample := range dataChan {
			// Perform training
			nn.Train(sample.InputVec, sample.TargetVec, outputs, errors, adjVec, weightDeltas)
		}

		fmt.Printf("Status: epoch %d/%d done\n", it+1, iterations)
	}

	// Print summary
	fmt.Printf("Finished. Time elapsed: %s\n", time.Since(tStart).String())
	ds.Close()
}

// Fetches batches of data in the background and shuffles it before sending the batch back
func dataFetchAsync(ds dataset.Dataset, dataChan chan (*dataset.DataSample), size int) {
	samples := make([]*dataset.DataSample, 0, size)

	var sendSamples = func() {
		rand.Seed(time.Now().UnixNano())
		rand.Shuffle(len(samples), func(i, j int) {
			samples[i], samples[j] = samples[j], samples[i]
		})
		for _, send := range samples {
			dataChan <- send
		}
	}

	sample := ds.NextSample()
	for sample != nil {
		samples = append(samples, sample)

		if len(samples) == size {
			sendSamples()
			samples = samples[:0]
		}
		sample = ds.NextSample()
	}
	sendSamples()
	close(dataChan)
}
