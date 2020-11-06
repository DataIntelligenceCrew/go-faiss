package faiss

/*
#cgo LDFLAGS: -lfaiss_c

#include <stdlib.h>
#include <faiss/c_api/IndexFlat_c.h>
#include <faiss/c_api/Index_c.h>
#include <faiss/c_api/error_c.h>
#include <faiss/c_api/faiss_c.h>
#include <faiss/c_api/impl/AuxIndexStructures_c.h>
#include <faiss/c_api/index_factory_c.h>
*/
import "C"

import (
	"errors"
	"unsafe"
)

// cVecMatrix allocates a len(x) * dim contiguous matrix in C memory that is a
// copy of x.
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

// cIdxArray allocates an array of idx_t in C memory that is a copy of a.
func cIdxArray(a []int) *C.idx_t {
	ca := C.malloc(C.size_t(len(a)) * C.sizeof_idx_t)
	ga := (*[1 << 30]C.idx_t)(ca)

	for i, v := range a {
		ga[i] = C.idx_t(v)
	}
	return (*C.idx_t)(ca)
}

func getLastError() error {
	err := C.GoString(C.faiss_get_last_error())
	return errors.New(err)
}

func indexAdd(idx *C.FaissIndex, x [][]float32) error {
	cx := cVecMatrix(x, int(C.faiss_Index_d(idx)))
	defer C.free(unsafe.Pointer(cx))

	c := C.faiss_Index_add(idx, C.idx_t(len(x)), cx)
	if c != 0 {
		return getLastError()
	}
	return nil
}

func indexAddWithIDs(idx *C.FaissIndex, x [][]float32, ids []int) error {
	cx := cVecMatrix(x, int(C.faiss_Index_d(idx)))
	defer C.free(unsafe.Pointer(cx))
	ci := cIdxArray(ids)
	defer C.free(unsafe.Pointer(ci))

	c := C.faiss_Index_add_with_ids(idx, C.idx_t(len(x)), cx, ci)
	if c != 0 {
		return getLastError()
	}
	return nil
}

func indexSearch(idx *C.FaissIndex, x [][]float32, k int) (
	dist [][]float32, labels [][]int, err error,
) {
	cx := cVecMatrix(x, int(C.faiss_Index_d(idx)))
	defer C.free(unsafe.Pointer(cx))

	cl := (*C.idx_t)(C.malloc(C.size_t(len(x)*k) * C.sizeof_idx_t))
	defer C.free(unsafe.Pointer(cl))
	cd := (*C.float)(C.malloc(C.size_t(len(x)*k) * C.sizeof_float))
	defer C.free(unsafe.Pointer(cd))

	c := C.faiss_Index_search(idx, C.idx_t(len(x)), cx, C.idx_t(k), cd, cl)
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

func indexRemoveIDs(idx *C.FaissIndex, ids []int) (int, error) {
	ci := cIdxArray(ids)
	defer C.free(unsafe.Pointer(ci))

	var sel *C.FaissIDSelectorBatch
	c := C.faiss_IDSelectorBatch_new(&sel, C.size_t(len(ids)), ci)
	if c != 0 {
		return 0, getLastError()
	}

	var nRemoved C.size_t
	c = C.faiss_Index_remove_ids(idx, (*C.FaissIDSelector)(sel), &nRemoved)
	if c != 0 {
		return 0, getLastError()
	}
	return int(nRemoved), nil
}

//--------------------------------------------------
// Index
//--------------------------------------------------

// Metric type
const (
	MetricInnerProduct  = int(C.METRIC_INNER_PRODUCT)
	MetricL2            = int(C.METRIC_L2)
	MetricL1            = int(C.METRIC_L1)
	MetricLinf          = int(C.METRIC_Linf)
	MetricLp            = int(C.METRIC_Lp)
	MetricCanberra      = int(C.METRIC_Canberra)
	MetricBrayCurtis    = int(C.METRIC_BrayCurtis)
	MetricJensenShannon = int(C.METRIC_JensenShannon)
)

// Index is the base structure for an index.
type Index struct {
	idx *C.FaissIndex
}

// IndexFactory builds a composite index.
// description is a comma-separated list of components.
func IndexFactory(d int, description string, metric int) (*Index, error) {
	cdesc := C.CString(description)
	defer C.free(unsafe.Pointer(cdesc))
	var index Index
	c := C.faiss_index_factory(&index.idx, C.int(d), cdesc, C.FaissMetricType(metric))
	if c != 0 {
		return nil, getLastError()
	}
	return &index, nil
}

// AsFlat casts idx to a flat index.
// AsFlat panics if idx is not a flat index.
func (idx *Index) AsFlat() *IndexFlat {
	ptr := C.faiss_IndexFlat_cast(idx.idx)
	if ptr == nil {
		panic("index is not a flat index")
	}
	return &IndexFlat{ptr}
}

// Add adds vectors to the index.
func (idx *Index) Add(x [][]float32) error {
	return indexAdd(idx.idx, x)
}

// AddWithIDs is like Add, but stores ids instead of sequential IDs.
func (idx *Index) AddWithIDs(x [][]float32, ids []int) error {
	return indexAddWithIDs(idx.idx, x, ids)
}

// Search queries the index with the vectors in x.
// Returns the IDs of the k nearest neighbors for each query vector in labels
// and the corresponding distances in dist.
func (idx *Index) Search(x [][]float32, k int) (
	dist [][]float32, labels [][]int, err error,
) {
	return indexSearch(idx.idx, x, k)
}

// RemoveIDs removes vectors with the given IDs from the index.
// Returns the number of elements removed and error or nil.
func (idx *Index) RemoveIDs(ids []int) (int, error) {
	return indexRemoveIDs(idx.idx, ids)
}

// Delete frees the memory used by the index.
func (idx *Index) Delete() {
	C.faiss_Index_free(idx.idx)
}

//--------------------------------------------------
// IndexFlat
//--------------------------------------------------

// IndexFlat is an index that stores the full vectors and performs exhaustive
// search.
type IndexFlat struct {
	idx *C.FaissIndexFlat
}

// NewIndexFlat creates a new IndexFlat.
func NewIndexFlat(d int, metric int) (*IndexFlat, error) {
	var index IndexFlat
	c := C.faiss_IndexFlat_new_with(&index.idx, C.idx_t(d), C.FaissMetricType(metric))
	if c != 0 {
		return nil, getLastError()
	}
	return &index, nil
}

// NewIndexFlatIP creates a new IndexFlat with inner product as the metric type.
func NewIndexFlatIP(d int) (*IndexFlat, error) {
	return NewIndexFlat(d, MetricInnerProduct)
}

func (idx *IndexFlat) Add(x [][]float32) error {
	return indexAdd(idx.idx, x)
}

func (idx *IndexFlat) AddWithIDs(x [][]float32, ids []int) error {
	return indexAddWithIDs(idx.idx, x, ids)
}

func (idx *IndexFlat) Search(x [][]float32, k int) (
	dist [][]float32, labels [][]int, err error,
) {
	return indexSearch(idx.idx, x, k)
}

func (idx *IndexFlat) RemoveIDs(ids []int) (int, error) {
	return indexRemoveIDs(idx.idx, ids)
}

func (idx *IndexFlat) Delete() {
	C.faiss_Index_free(idx.idx)
}
