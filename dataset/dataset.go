package dataset

import (
	"gonum.org/v1/gonum/mat"
)

type Dataset interface {
	NumInputs() int
	NumLabels() int
	NextSample() *DataSample
	Reset()
	Close()
}

type DataSample struct {
	Label     int
	InputVec  *mat.VecDense
	TargetVec *mat.VecDense
}
