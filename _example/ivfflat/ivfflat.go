// Usage example for IndexIVFFlat.
// Based on tutorial/cpp/2-IVFFlat.cpp from the Faiss distribution.
// See https://github.com/facebookresearch/faiss/wiki/Faster-search for more
// information.
package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/AnyVisionltd/go-faiss"
)

func main() {
	d := 64      // dimension
	nb := 100000 // database size
	nq := 10000  // number of queries

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

	index, err := faiss.IndexFactory(d, "IVF100,Flat", faiss.MetricL2)
	if err != nil {
		log.Fatal(err)
	}
	defer index.Delete()

	fmt.Println("IsTrained() =", index.IsTrained())
	index.Train(xb)
	fmt.Println("IsTrained() =", index.IsTrained())
	index.Add(xb)

	k := int64(4)

	// search xq

	_, ids, err := index.Search(xq, k)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("ids (last 5 results)=")
	for i := int64(nq) - 5; i < int64(nq); i++ {
		for j := int64(0); j < k; j++ {
			fmt.Printf("%5d ", ids[i*k+j])
		}
		fmt.Println()
	}

	// retry with nprobe=10 (default is 1)

	ps, err := faiss.NewParameterSpace()
	if err != nil {
		log.Fatal(err)
	}
	defer ps.Delete()

	if err := ps.SetIndexParameter(index, "nprobe", 10); err != nil {
		log.Fatal(err)
	}

	_, ids, err = index.Search(xq, k)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("ids (last 5 results)=")
	for i := int64(nq) - 5; i < int64(nq); i++ {
		for j := int64(0); j < k; j++ {
			fmt.Printf("%5d ", ids[i*k+j])
		}
		fmt.Println()
	}
}
