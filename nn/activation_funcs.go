package nn

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type ActivationFunction int

const (
	Sigmoid ActivationFunction = iota
	Tanh
	Relu
	LeakyRelu
	SoftMax
)

// ActivationFunction is the interface for an activation function
type activationFunction func(*mat.VecDense)
type activationFunctionD func(float64) float64

func (af ActivationFunction) Funcs() (activationFunction, activationFunctionD) {
	switch af {
	case Sigmoid:
		return sigmoid, sigmoidD
	case Tanh:
		return tanh, tanhD
	case Relu:
		return relu, reluD
	case LeakyRelu:
		return lRelu, lReluD
	case SoftMax:
		return softMax, softMaxD
	default:
		return nil, nil
	}
}

// Sigmoid applies the sigmoid activation function to the given value
func sigmoid(vec *mat.VecDense) {
	for i := 0; i < vec.Len(); i++ {
		vec.SetVec(i, 1/(1+math.Exp(-vec.AtVec(i))))
	}
}

func sigmoidD(sig float64) float64 {
	return sig * (1 - sig)
}

// Tanh applies the tanh activation function to the given value
func tanh(vec *mat.VecDense) {
	for i := 0; i < vec.Len(); i++ {
		vec.SetVec(i, math.Sinh(vec.AtVec(i))/math.Cosh(vec.AtVec(i)))
	}
}

func tanhD(tan float64) float64 {
	return 1 - (tan * tan)
}

func relu(vec *mat.VecDense) {
	for i := 0; i < vec.Len(); i++ {
		if vec.AtVec(i) < 0 {
			vec.SetVec(i, 0)
		}
	}
}

func reluD(rel float64) float64 {
	if rel < 0 {
		return 0
	}
	return 1
}

func lRelu(vec *mat.VecDense) {
	for i := 0; i < vec.Len(); i++ {
		if vec.AtVec(i) < 0 {
			vec.SetVec(i, 0.01*vec.AtVec(i))
		}
	}
}

func lReluD(rel float64) float64 {
	if rel < 0 {
		return 0.01
	}
	return 1
}

func softMax(vec *mat.VecDense) {
	max := mat.Max(vec)
	sum := float64(0)

	for i := 0; i < vec.Len(); i++ {
		vec.SetVec(i, math.Exp(vec.AtVec(i)-max))
		sum += vec.AtVec(i)
	}
	for i := 0; i < vec.Len(); i++ {
		vec.SetVec(i, vec.AtVec(i)/sum)
	}
}

func softMaxD(sm float64) float64 {
	return sm * (1 - sm)
}
