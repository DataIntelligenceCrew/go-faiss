package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/blevesearch/go-faiss"
)

func main() {
	d := 64      // dimension
	nb := 100000 // database size
	nq := 10000  // number of queries

	database_vecs := make([]float32, d*nb)
	query_vecs := make([]float32, d*nq)

	for i := 0; i < nb; i++ {
		for j := 0; j < d; j++ {
			database_vecs[i*d+j] = rand.Float32()
		}
		database_vecs[i*d] += float32(i) / 1000
	}

	for i := 0; i < nq; i++ {
		for j := 0; j < d; j++ {
			query_vecs[i*d+j] = rand.Float32()
		}
		query_vecs[i*d] += float32(i) / 1000
	}

	index, err := faiss.IndexFactory(d, "HNSW32,PQ4", faiss.MetricL2)
	if err != nil {
		log.Fatal(err)
	}
	defer index.Close()

	index.Train(database_vecs)
	index.Add(database_vecs)

	k := int64(4)

	// sanity check

	dist, ids, err := index.Search(database_vecs[:5*d], k)
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

	_, ids, err = index.Search(query_vecs, k)
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
