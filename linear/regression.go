package linear

import (
	"github.com/vikashvverma/goml/linear"
)

type Regression interface {
	Learn() error
	Predict(x []float64, normalize ...bool) ([]float64, error)
	Persist(path string) error
	Restore(path string) error
}

type regression struct {
	model linear.LeastSquaresRegression
}

func NewRegression(model linear.LeastSquaresRegression) Regression {
	return &regression{
		model: model,
	}
}

func (r *regression) Learn() error {
	return r.model.Learn()
}

func (r *regression) Predict(x []float64, normalize ...bool) ([]float64, error) {
	return r.model.Predict(x, normalize...)
}

func (r *regression) Persist(path string) error {
	return r.model.PersistToFile(path)
}

func (r *regression) Restore(path string) error {
	return r.model.RestoreFromFile(path)
}
