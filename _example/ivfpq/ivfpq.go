// Usage example for IndexIVFPQ.
// Based on tutorial/cpp/3-IVFPQ.cpp from the Faiss distribution.
// See https://github.com/facebookresearch/faiss/wiki/Lower-memory-footprint for
// more information.
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

	index, err := faiss.IndexFactory(d, "IVF100,PQ8", faiss.MetricL2)
	if err != nil {
		log.Fatal(err)
	}
	defer index.Delete()

	index.Train(xb)
	index.Add(xb)

	k := int64(4)

	// sanity check

	dist, ids, err := index.Search(xb[:5*d], k)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("ids=")
	for i := int64(0); i < 5; i++ {
		for j := int64(0); j < k; j++ {
			fmt.Printf("%5d ", ids[i*k+j])
		}
		fmt.Println()
	}

	fmt.Println("dist=")
	for i := int64(0); i < 5; i++ {
		for j := int64(0); j < k; j++ {
			fmt.Printf("%7.6g ", dist[i*k+j])
		}
		fmt.Println()
	}

	// search xq

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
