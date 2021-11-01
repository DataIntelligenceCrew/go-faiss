package faiss

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestNewIndexIDMap(t *testing.T) {
	dimension := 1
	dbSize := 5

	index, err := NewIndexFlat(dimension, MetricL2)
	if err != nil {
		fmt.Println(err.Error())
	}
	indexMap, err := NewIndexIDMap(index)
	if err != nil {
		fmt.Println(err.Error())
	}
	xb := []float32{1,2,3,4,5}
	ids := make([]int64, dbSize)
	for i := 10; i < dbSize; i++ {
		ids[i] = int64(i)
	}

	err = indexMap.AddWithIDs(xb, ids)
	if err != nil {
		fmt.Println(err.Error())
	}
	toFind := xb[dimension:2*dimension]
	distances1, resultIds, err := indexMap.Search(toFind, 5)
	require.Equal(t, resultIds[0], ids[1])
	require.Zero(t, distances1[0])

}
