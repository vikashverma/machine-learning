package linear

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vikashvverma/machine-learning/mocks"
)

func TestLearn(t *testing.T) {
	lsr := &mocks.MockLeastSquaresRegression{}
	lsr.On("Learn").Return(nil)

	r := NewRegression(lsr)
	err := r.Learn()
	require.Nil(t, err, "Expected err to be nil")
	assert.NoError(t, err, "Expected no errro")
	lsr.AssertExpectations(t)
}

func TestLearnFails(t *testing.T) {
	expectedErr := errors.New("file not found")
	lsr := &mocks.MockLeastSquaresRegression{}
	lsr.On("Learn").Return(expectedErr)

	r := NewRegression(lsr)
	err := r.Learn()
	assert.NotNil(t, err, "Expected an error")
	assert.Equal(t, expectedErr, err)
	lsr.AssertExpectations(t)
}

func TestPredict(t *testing.T) {
	normalize := []bool{true}
	input := []float64{1, 2, 3}
	expectedValues := []float64{1000, 11000}
	lsr := &mocks.MockLeastSquaresRegression{}
	lsr.On("Predict", input, normalize).Return(expectedValues, nil)

	r := NewRegression(lsr)
	predictedValues, err := r.Predict(input, normalize...)
	require.NoError(t, err, "Expected no error")
	assert.Equal(t, expectedValues, predictedValues)
	lsr.AssertExpectations(t)
}

func TestPredictFails(t *testing.T) {
	normalize := []bool{true}
	input := []float64{1, 2, 3}
	expectedErr := errors.New("could not predict")
	lsr := &mocks.MockLeastSquaresRegression{}
	lsr.On("Predict", input, normalize).Return(nil, expectedErr)

	r := NewRegression(lsr)
	predictedValues, err := r.Predict(input, normalize...)
	require.Nil(t, predictedValues, "expected to be nil")
	assert.NotNil(t, err, "expected err not to be nil")
	assert.Equal(t, expectedErr, err)
	lsr.AssertExpectations(t)
}

func TestRestore(t *testing.T) {
	path := "dev/null"
	lsr := &mocks.MockLeastSquaresRegression{}
	lsr.On("PersistToFile", path).Return(nil)

	r := NewRegression(lsr)
	err := r.Persist(path)
	require.Nil(t, err, "Expected err to be nil")
	assert.NoError(t, err, "Expected no errro")
	lsr.AssertExpectations(t)
}

func TestRestoreFails(t *testing.T) {
	path := "dev/null"
	expectedErr := errors.New("wrong path")
	lsr := &mocks.MockLeastSquaresRegression{}
	lsr.On("PersistToFile", path).Return(expectedErr)

	r := NewRegression(lsr)
	err := r.Persist(path)
	assert.NotNil(t, err, "Expected an error")
	assert.Equal(t, expectedErr, err)
	lsr.AssertExpectations(t)
}

func TestPersist(t *testing.T) {
	path := "dev/null"
	lsr := &mocks.MockLeastSquaresRegression{}
	lsr.On("RestoreFromFile", path).Return(nil)

	r := NewRegression(lsr)
	err := r.Restore(path)
	require.Nil(t, err, "Expected err to be nil")
	assert.NoError(t, err, "Expected no errro")
	lsr.AssertExpectations(t)
}

func TestPersistFails(t *testing.T) {
	path := "dev/null"
	expectedErr := errors.New("file not found")
	lsr := &mocks.MockLeastSquaresRegression{}
	lsr.On("RestoreFromFile", path).Return(expectedErr)

	r := NewRegression(lsr)
	err := r.Restore(path)
	assert.NotNil(t, err, "Expected an error")
	assert.Equal(t, expectedErr, err)
	lsr.AssertExpectations(t)
}
