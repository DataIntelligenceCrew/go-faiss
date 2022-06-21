//go:build gpu
// +build gpu

package faiss

/*

#include <stddef.h>
#include <faiss/c_api/gpu/StandardGpuResources_c.h>
#include <faiss/c_api/gpu/GpuAutoTune_c.h>
#include <faiss/c_api/gpu/GpuIndicesOptions_c.h>
*/
import "C"
import (
	"errors"
	"unsafe"
)

func TransferToGpu(index Index) (Index, error) {
	var gpuResource *C.FaissStandardGpuResources
	var gpuIndex *C.FaissGpuIndex
	c := C.faiss_StandardGpuResources_new(&gpuResource)
	if c != 0 {
		return nil, errors.New("error on init gpu %v")
	}

	exitCode := C.faiss_index_cpu_to_gpu(gpuResource, 0, index.cPtr(), &gpuIndex)

	if exitCode != 0 {
		return nil, errors.New("error transferring to gpu")
	}

	var gpuResources []*C.FaissStandardGpuResources
	gpuResources = append(gpuResources, gpuResource)

	return &faissIndex{idx: gpuIndex, resources: gpuResources}, nil
}

// to use sharding use this version of faiss - git clone -b fix-cpi-gpu-faiss_index_cpu_to_gpu_multiple_with_options https://github.com/AviadHAv/faiss.git
// TransferToAllGPUs - gpuIndexes - which gpus to use [2,4,5] , index - flat or idmap , shard - should shard accross all gpus if not will put same data on all gpus
func TransferToAllGPUs(index Index, gpuIndexes []int, sharding bool) (Index, error) {
	amountOfGPUs := len(gpuIndexes)
	gpuResources := make([]*C.FaissStandardGpuResources, len(gpuIndexes))
	for i := 0; i < len(gpuIndexes); i++ {
		var resourceIndex *C.FaissStandardGpuResources
		gpuResources[i] = resourceIndex
	}

	var gpuIndex *C.FaissGpuIndex
	for i := 0; i < amountOfGPUs; i++ {
		c := C.faiss_StandardGpuResources_new(&gpuResources[i])
		if c != 0 {
			return nil, errors.New("error on init gpu %v")
		}
	}
	var exitCode C.int
	if sharding {
		exitCode = C.faiss_index_cpu_to_gpu_multiple_sharding(
			(**C.FaissStandardGpuResources)(unsafe.Pointer(&gpuResources[0])),
			C.size_t(len(gpuResources)),
			(*C.int)(unsafe.Pointer(&gpuIndexes[0])),
			C.size_t(len(gpuIndexes)),
			index.cPtr(),
			&gpuIndex)
	} else {
		exitCode = C.faiss_index_cpu_to_gpu_multiple(
			(**C.FaissStandardGpuResources)(unsafe.Pointer(&gpuResources[0])),
			(*C.int)(unsafe.Pointer(&gpuIndexes[0])),
			C.size_t(len(gpuIndexes)),
			index.cPtr(),
			&gpuIndex)
	}

	if exitCode != 0 {
		return nil, errors.New("error transferring to gpu")
	}

	return &faissIndex{idx: gpuIndex, resources: gpuResources}, nil
}

func TransferToCpu(gpuIndex Index) (Index, error) {
	var cpuIndex *C.FaissIndex

	exitCode := C.faiss_index_gpu_to_cpu(gpuIndex.cPtr(), &cpuIndex)
	if exitCode != 0 {
		return nil, errors.New("error transferring to gpu")
	}

	Free(gpuIndex)

	return &faissIndex{idx: cpuIndex}, nil
}

func Free(index Index) {
	gpuResources := index.cGpuResource()
	for _, gpuResource := range gpuResources {
		C.faiss_StandardGpuResources_free(gpuResource)
	}
	index.Delete()

}

func CreateGpuIndex() (Index, error) {
	var gpuResource *C.FaissStandardGpuResources
	var gpuIndex *C.FaissGpuIndex
	c := C.faiss_StandardGpuResources_new(&gpuResource)
	if c != 0 {
		return nil, errors.New("error on init gpu %v")
	}

	var gpuResources []*C.FaissStandardGpuResources
	gpuResources = append(gpuResources, gpuResource)

	return &faissIndex{idx: gpuIndex, resources: gpuResources}, nil
}
