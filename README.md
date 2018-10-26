# golang-neural
Neural networks with golang

Implementation of artificial neural networks with Golang. I have tested this with the Mnist database of handwritten digits and managed to get 97.45% correct on the test set using a 784-128-64-10 network.

## Usage
Create a new neural network:
```
hiddenLayers := []nn.HiddenLayer{
		{Nodes: 64, ActivationFunc: nn.LeakyRelu},
		{Nodes: 64, ActivationFunc: nn.LeakyRelu},
		{Nodes: 32, ActivationFunc: nn.LeakyRelu}}

	nn := nn.NewNeuralNetwork(nn.Settings{
		InputNodes:   784,
		OutputNodes:  10,
		LearningRate: 0.0001,
		HiddenLayers: hiddenLayers})
```

You can also write the network to/from a file:
```
nn := nn.FromFile("network.json")
nn.ToFile("network.json")
```

For an example for training and testing the network with the Mnist digits dataset see "examples/mnist/main.go"
