# go-hnsw

go-hnsw is a GO implementation of the HNSW approximate nearest-neighbour search algorithm implemented in C++ in https://github.com/searchivarius/nmslib and described in https://arxiv.org/abs/1603.09320

NOTE: This version extends over the original one on the following:
 - Uses gonum/mat dense vectors to represent elements in the tree. No longer support for float32.
 - Also uses gonum BLAS functions. Available functions are Cosine and L2Squared
 - The HNSW object now support both Cosine and L2Squared metrics.
 - This code is NOT production ready!
 
## Usage

Simple usage example. See examples folder for more.
Note that both index building and searching can be safely done in parallel with multiple goroutines.
You can always extend the index, even while searching.

```go
package main

import (
	"fmt"
	"math/rand"
	"time"

	hnsw "github.com/tavoaqp/go-hnsw"
	"gonum.org/v1/gonum/mat"
)

func main() {

	const (
		M              = 32
		efConstruction = 400
		efSearch       = 100
		K              = 10
	)

	zero := mat.NewVecDense(128, nil)
	
	h := hnsw.New(M, efConstruction, zero, "l2")
	h.Grow(10000)

    // Note that added ID:s must start from 1
	for i := 1; i <= 10000; i++ {
		h.Add(randomPoint(), uint32(i))
		if (i)%1000 == 0 {
			fmt.Printf("%v points added\n", i)
		}
	}
	
	start := time.Now()
	for i := 0; i < 1000; i++ {
		Search(randomPoint, efSearch, K)
	}
	stop := time.Since(start)

	fmt.Printf("%v queries / second (single thread)\n", 1000.0/stop.Seconds())	
}

func randomPoint() *mat.VecDense {
	v := mat.NewVecDense(128, nil)
	for i := 0; i < 128; i++ {
		v.SetVec(i, rand.Float64())
	}
	return v
}

```
