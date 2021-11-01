//+build gpu

package faiss

/*
#include <faiss/c_api/gpu/StandardGpuResources_c.h>
#include <faiss/c_api/gpu/GpuAutoTune_c.h>
*/
import "C"
import (
	"errors"
)


func TransferToGpu(index Index) (Index, error) {
	var gpuResources *C.FaissStandardGpuResources
	var gpuIndex *C.FaissGpuIndex
	c := C.faiss_StandardGpuResources_new(&gpuResources)
	if c != 0 {
		return nil, errors.New("error on init gpu %v")
	}

	exitCode := C.faiss_index_cpu_to_gpu(gpuResources, 0, index.cPtr(), &gpuIndex)
	if exitCode != 0 {
		return nil, errors.New("error transferring to gpu")
	}

	return &faissIndex{idx: gpuIndex, resource: gpuResources}, nil
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
	var gpuResource *C.FaissStandardGpuResources
	gpuResource = index.cGpuResource()
	C.faiss_StandardGpuResources_free(gpuResource)
	index.Delete()
}

func CreateGpuIndex() (Index, error) {
	var gpuResource *C.FaissStandardGpuResources
	var gpuIndex *C.FaissGpuIndex
	c := C.faiss_StandardGpuResources_new(&gpuResource)
	if c != 0 {
		return nil, errors.New("error on init gpu %v")
	}

	return &faissIndex{idx: gpuIndex, resource: gpuResource}, nil
}
