package faiss

/*
#cgo LDFLAGS: -lfaiss_c

#include <stdlib.h>
#include <faiss/c_api/IndexFlat_c.h>
#include <faiss/c_api/Index_c.h>
#include <faiss/c_api/error_c.h>
#include <faiss/c_api/faiss_c.h>
#include <faiss/c_api/impl/AuxIndexStructures_c.h>
*/
import "C"

import (
	"errors"
	"unsafe"
)

// Index is the base structure for an index.
type Index struct {
	idx *C.FaissIndex
}

// IndexFlatIP is an index for maximum inner product search.
type IndexFlatIP struct {
	Index
}

// cVecMatrix allocates a len(x) * dim matrix in C memory that is a copy of x.
func cVecMatrix(x [][]float32, dim int) *C.float {
	cx := C.malloc(C.size_t(len(x)*dim) * C.sizeof_float)
	gx := (*[1 << 30]C.float)(cx)

	for i, vec := range x {
		for j := 0; j < dim; j++ {
			gx[dim*i+j] = C.float(vec[j])
		}
	}
	return (*C.float)(cx)
}

func getLastError() error {
	err := C.GoString(C.faiss_get_last_error())
	return errors.New(err)
}

// Add adds vectors to the index.
func (idx *Index) Add(x [][]float32) error {
	cx := cVecMatrix(x, int(C.faiss_Index_d(idx.idx)))
	defer C.free(unsafe.Pointer(cx))

	c := C.faiss_Index_add(idx.idx, C.idx_t(len(x)), cx)
	if c != 0 {
		return getLastError()
	}
	return nil
}

// Search queries the index with the vectors in x.
// Returns the IDs of the k nearest neighbors for each query vector in labels
// and the corresponding distances in dist.
func (idx *Index) Search(x [][]float32, k int) (
	dist [][]float32, labels [][]int, err error,
) {
	cx := cVecMatrix(x, int(C.faiss_Index_d(idx.idx)))
	defer C.free(unsafe.Pointer(cx))

	cl := (*C.idx_t)(C.malloc(C.size_t(len(x)*k) * C.sizeof_idx_t))
	defer C.free(unsafe.Pointer(cl))
	cd := (*C.float)(C.malloc(C.size_t(len(x)*k) * C.sizeof_float))
	defer C.free(unsafe.Pointer(cd))

	c := C.faiss_Index_search(idx.idx, C.idx_t(len(x)), cx, C.idx_t(k), cd, cl)
	if c != 0 {
		return nil, nil, getLastError()
	}

	gl := (*[1 << 30]C.idx_t)(unsafe.Pointer(cl))
	gd := (*[1 << 30]C.float)(unsafe.Pointer(cd))

	for i := 0; i < len(x); i++ {
		lrow := make([]int, k)
		drow := make([]float32, k)

		for j := 0; j < k; j++ {
			cidx := k*i + j
			lrow[j] = int(gl[cidx])
			drow[j] = float32(gd[cidx])
		}
		labels = append(labels, lrow)
		dist = append(dist, drow)
	}

	return dist, labels, nil
}

// RemoveIDs removes vectors with the given IDs from the index.
// Returns the number of elements removed and error or nil.
func (idx *Index) RemoveIDs(ids []int) (int, error) {
	ci := C.malloc(C.size_t(len(ids)) * C.sizeof_idx_t)
	defer C.free(ci)
	gi := (*[1 << 30]C.idx_t)(ci)

	for i, v := range ids {
		gi[i] = C.idx_t(v)
	}

	var sel *C.FaissIDSelectorBatch
	c := C.faiss_IDSelectorBatch_new(&sel, C.size_t(len(ids)), (*C.idx_t)(ci))
	if c != 0 {
		return 0, getLastError()
	}

	var nRemoved C.size_t
	c = C.faiss_Index_remove_ids(idx.idx, (*C.FaissIDSelector)(sel), &nRemoved)
	if c != 0 {
		return 0, getLastError()
	}
	return int(nRemoved), nil
}

// Delete frees the memory used by the index.
func (idx *Index) Delete() {
	C.faiss_Index_free(idx.idx)
}

// NewIndexFlatIP creates a new IndexFlatIP.
func NewIndexFlatIP(d int) (*IndexFlatIP, error) {
	var index Index
	c := C.faiss_IndexFlatIP_new_with(&index.idx, C.idx_t(d))
	if c != 0 {
		return nil, getLastError()
	}
	return &IndexFlatIP{index}, nil
}
