package main

import (
	"github.com/haakonleg/neural/mnist"
	"github.com/haakonleg/neural/nn"
)

func main() {
	dsTrain := mnist.NewMnistDatasetFromFile("mnist_train.csv")
	dsTest := mnist.NewMnistDatasetFromFile("mnist_test.csv")

	hiddenLayers := []nn.HiddenLayer{
		{Nodes: 64, ActivationFunc: nn.LeakyRelu},
		{Nodes: 64, ActivationFunc: nn.LeakyRelu},
		{Nodes: 32, ActivationFunc: nn.LeakyRelu}}

	nn := nn.NewNeuralNetwork(nn.Settings{
		InputNodes:   dsTrain.NumInputs(),
		OutputNodes:  dsTrain.NumLabels(),
		LearningRate: 0.0001,
		HiddenLayers: hiddenLayers})

	nn.TrainWithDataset(dsTrain, 5000, 1)
	nn.Test(dsTest)

	/*
		nn := nn.FromFile("test.json")
		nn.TrainWithDataset(dsTrain, 10000, 2)
		nn.Test(dsTest)*/

	nn.ToFile("test.json")
}
