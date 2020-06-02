package distances

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/blas/blas64"
	"math"
)

func Cosine(vec1 *mat.VecDense, vec2 *mat.VecDense) float64 {
	vec1Norm := math.Sqrt(mat.Dot(vec1, vec1))
	vec2Norm := math.Sqrt(mat.Dot(vec2, vec2))
	vecMul := mat.Dot(vec1, vec2)
	return 1 - vecMul/(vec1Norm * vec2Norm)
}

func L2Squared(vec1 *mat.VecDense, vec2 *mat.VecDense) float64 {
	subtractVector := mat.VecDenseCopyOf(vec1)
	subtractVector.SubVec(subtractVector, vec2)
	norm := blas64.Nrm2(subtractVector.RawVector())
	return norm
}
