package nn

import (
	"encoding/json"
	"io/ioutil"
	"log"

	"gonum.org/v1/gonum/mat"
)

type nnJSON struct {
	InputNodes   int     `json:"input_nodes"`
	OutputNodes  int     `json:"output_nodes"`
	LearningRate float64 `json:"learning_rate"`
	Weights      []wJSON `json:"weights"`
}

type wJSON struct {
	LNodes  int                `json:"l_nodes"`
	RNodes  int                `json:"r_nodes"`
	ActFunc ActivationFunction `json:"act_func"`
	M       []byte             `json:"m"`
}

func (nn *NeuralNetwork) ToFile(fileName string) {
	weights := make([]wJSON, len(nn.weights))

	for i := range weights {
		w, err := nn.weights[i].m.MarshalBinary()
		if err != nil {
			log.Fatal(err)
		}

		weights[i] = wJSON{
			LNodes:  nn.weights[i].LNodes,
			RNodes:  nn.weights[i].RNodes,
			ActFunc: nn.weights[i].ActFunc,
			M:       w}
	}

	nnJSON := &nnJSON{
		InputNodes:   nn.InputNodes,
		OutputNodes:  nn.OutputNodes,
		LearningRate: nn.LearningRate,
		Weights:      weights}

	jsonByt, err := json.Marshal(nnJSON)
	if err != nil {
		log.Fatal(err)
	}

	if err := ioutil.WriteFile(fileName, jsonByt, 0644); err != nil {
		log.Fatal(err)

	}
}

func FromFile(fileName string) *NeuralNetwork {
	jsonByt, err := ioutil.ReadFile(fileName)
	if err != nil {
		log.Fatal(err)
	}

	nnJSON := new(nnJSON)
	if err := json.Unmarshal(jsonByt, nnJSON); err != nil {
		log.Fatal(err)

	}

	weights := make([]Weights, len(nnJSON.Weights))
	for i := range weights {

		w := new(mat.Dense)
		if err := w.UnmarshalBinary(nnJSON.Weights[i].M); err != nil {
			log.Fatal(err)

		}

		weights[i] = Weights{
			LNodes:  nnJSON.Weights[i].LNodes,
			RNodes:  nnJSON.Weights[i].RNodes,
			ActFunc: nnJSON.Weights[i].ActFunc,
			m:       w}

		weights[i].actFunc, weights[i].derivFunc = nnJSON.Weights[i].ActFunc.Funcs()
	}

	return &NeuralNetwork{
		InputNodes:   nnJSON.InputNodes,
		OutputNodes:  nnJSON.OutputNodes,
		LearningRate: nnJSON.LearningRate,
		weights:      weights}
}
