//+build cpu

package faiss

import "C"
import "errors"

func TransferToGpu(index Index) (Index, error) {
	return nil, errors.New("Not supported when running in CPU mode..")
}

func TransferToCpu(index Index) (Index, error) {
	return nil, errors.New("Not supported when running in CPU mode..")
}

func Free(gpuIndex Index) error {
	return errors.New("Not supported when running in CPU mode..")
}

func CreateGpuIndex() (Index, error) {
	return nil, errors.New("Not supported when running in CPU mode..")
}

func TransferToAllGPUs(index Index,gpuIndexes []int) (Index, error) {
	return nil, errors.New("Not supported when running in CPU mode..")
}

func TransferToAllGPUsWithOptions(index Index,gpuIndexes []int) (Index, error) {
	return nil, errors.New("Not supported when running in CPU mode..")
}