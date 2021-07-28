package faiss

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestFlatIndexOnGpu(t *testing.T) {
	index, err := NewIndexFlatL2(1)
	require.Nil(t, err)

	gpuIdx, err := TransferToGpu(index)
	require.Nil(t, err)

	vectorsToAdd := []float32{1,2,3,4,5}
	err = gpuIdx.Add(vectorsToAdd)
	require.Nil(t, err)

	distances, resultIds, err := gpuIdx.Search(vectorsToAdd, 5)
	require.Nil(t, err)

	fmt.Println(distances, resultIds, err)
	for i := range vectorsToAdd {
		require.Equal(t, int64(i), resultIds[len(vectorsToAdd)*i])
		require.Zero(t, distances[len(vectorsToAdd)*i])
	}
}

func TestIndexIDMapOnGPU(t *testing.T) {
	index, err := NewIndexFlatL2(1)
	require.Nil(t, err)

	indexMap, err := NewIndexIDMap(index)
	require.Nil(t, err)

	gpuIndex, err := TransferToGpu(indexMap)
	require.Nil(t, err)

	vectorsToAdd := []float32{1,2,3,4,5}
	ids := make([]int64, len(vectorsToAdd))
	for i := 0; i < len(vectorsToAdd); i++ {
		ids[i] = int64(i)
	}

	err = gpuIndex.AddWithIDs(vectorsToAdd, ids)
	require.Nil(t, err)

	distances, resultIds, err := gpuIndex.Search(vectorsToAdd, 5)
	require.Nil(t, err)
	fmt.Println(gpuIndex.D(), gpuIndex.Ntotal())
	fmt.Println(distances, resultIds, err)
	for i := range vectorsToAdd {
		require.Equal(t, ids[i], resultIds[len(vectorsToAdd)*i])
		require.Zero(t, distances[len(vectorsToAdd)*i])
	}
}
