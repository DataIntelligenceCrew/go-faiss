// Usage example for IndexIVFFlat.
// Based on tutorial/cpp/2-IVFFlat.cpp from the Faiss distribution.
// See https://github.com/facebookresearch/faiss/wiki/Faster-search for more
// information.
package main

import (
	"log"
	"math/rand"

	"github.com/blevesearch/go-faiss"
)

func main() {
	d := 64       // dimension
	nb := 1000000 // database size
	nq := 10000   // number of queries

	xb := make([]float32, d*nb)
	xq := make([]float32, d*nq)

	for i := 0; i < nb; i++ {
		for j := 0; j < d; j++ {
			xb[i*d+j] = rand.Float32()
		}
		xb[i*d] += float32(i) / 1000
	}

	for i := 0; i < nq; i++ {
		for j := 0; j < d; j++ {
			xq[i*d+j] = rand.Float32()
		}
		xq[i*d] += float32(i) / 1000
	}

	index, err := faiss.IndexFactory(d, "IDMap2,IVF100,SQ8", faiss.MetricInnerProduct)
	if err != nil {
		log.Fatal(err)
	}
	defer index.Close()

	_, err = index.GetSubIndex()
	if err != nil {
		log.Fatal(err)
	}
}
