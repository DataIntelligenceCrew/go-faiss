package main

import (
	"fmt"
	"github.com/DataIntelligenceCrew/go-faiss"
)

func main() {
	dimension := 1
	dbSize := 5

	index, err := faiss.NewIndexFlat(dimension, faiss.MetricL2)
	if err != nil {
		fmt.Println(err.Error())
	}
	indexMap, err := faiss.NewIndexIDMap(index)
	if err != nil {
		fmt.Println(err.Error())
	}
	xb := []float32{1,2,3,4,5}
	ids := make([]int64, dbSize)
	for i := 0; i < dbSize; i++ {
		ids[i] = int64(i)
	}

	err = indexMap.AddWithIDs(xb, ids)
	if err != nil {
		fmt.Println(err.Error())
	}
	toFind := xb[dimension:2*dimension]
	distances1, resultIds, err := indexMap.Search(toFind, 5)
	fmt.Println(distances1, resultIds, err)
	fmt.Println(resultIds[0] == ids[1])
	fmt.Println(distances1[0] == 0)

}
