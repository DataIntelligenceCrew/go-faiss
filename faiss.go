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

func getLastError() error {
	return errors.New(C.GoString(C.faiss_get_last_error()))
}

//--------------------------------------------------
// AuxIndexStructures
//--------------------------------------------------

// IDSelector represents a set of IDs to remove.
type IDSelector struct {
	sel *C.FaissIDSelector
}

// NewIDSelectorBatch creates a new batch selector.
func NewIDSelectorBatch(indices []int64) (*IDSelector, error) {
	var sel *C.FaissIDSelectorBatch
	if c := C.faiss_IDSelectorBatch_new(
		&sel,
		C.size_t(len(indices)),
		(*C.idx_t)(&indices[0]),
	); c != 0 {
		return nil, getLastError()
	}
	return &IDSelector{(*C.FaissIDSelector)(sel)}, nil
}

// Delete frees the memory associated with s.
func (s *IDSelector) Delete() {
	C.faiss_IDSelector_free(s.sel)
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

func indexD(idx *C.FaissIndex) int {
	return int(C.faiss_Index_d(idx))
}

func indexAdd(idx *C.FaissIndex, x []float32) error {
	n := len(x) / indexD(idx)
	if c := C.faiss_Index_add(idx, C.idx_t(n), (*C.float)(&x[0])); c != 0 {
		return getLastError()
	}
	return nil
}

func indexAddWithIDs(idx *C.FaissIndex, x []float32, xids []int64) error {
	n := len(x) / indexD(idx)
	if c := C.faiss_Index_add_with_ids(
		idx,
		C.idx_t(n),
		(*C.float)(&x[0]),
		(*C.idx_t)(&xids[0]),
	); c != 0 {
		return getLastError()
	}
	return nil
}

func indexSearch(idx *C.FaissIndex, x []float32, k int) (
	distances []float32, labels []int64, err error,
) {
	n := len(x) / indexD(idx)
	distances = make([]float32, n*k)
	labels = make([]int64, n*k)
	if c := C.faiss_Index_search(
		idx,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.idx_t(k),
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		err = getLastError()
	}
	return
}

func indexRemoveIDs(idx *C.FaissIndex, sel *C.FaissIDSelector) (int, error) {
	var nRemoved C.size_t
	if c := C.faiss_Index_remove_ids(idx, sel, &nRemoved); c != 0 {
		return 0, getLastError()
	}
	return int(nRemoved), nil
}

func indexDelete(idx *C.FaissIndex) {
	C.faiss_Index_free(idx)
}

// D returns the dimension of the indexed vectors.
func (idx *Index) D() int {
	return indexD(idx.idx)
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
func (idx *Index) Add(x []float32) error {
	return indexAdd(idx.idx, x)
}

// AddWithIDs is like Add, but stores xids instead of sequential IDs.
func (idx *Index) AddWithIDs(x []float32, xids []int64) error {
	return indexAddWithIDs(idx.idx, x, xids)
}

// Search queries the index with the vectors in x.
// Returns the IDs of the k nearest neighbors for each query vector and the
// corresponding distances.
func (idx *Index) Search(x []float32, k int) (
	distances []float32, labels []int64, err error,
) {
	return indexSearch(idx.idx, x, k)
}

// RemoveIDs removes the vectors specified by sel from the index.
// Returns the number of elements removed and error.
func (idx *Index) RemoveIDs(sel *IDSelector) (int, error) {
	return indexRemoveIDs(idx.idx, sel.sel)
}

// Delete frees the memory used by the index.
func (idx *Index) Delete() {
	indexDelete(idx.idx)
}

//--------------------------------------------------
// index_factory
//--------------------------------------------------

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

//--------------------------------------------------
// IndexFlat
//--------------------------------------------------

// IndexFlat is an index that stores the full vectors and performs exhaustive
// search.
type IndexFlat struct {
	idx *C.FaissIndexFlat
}

// NewIndexFlat creates a new flat index.
func NewIndexFlat(d int, metric int) (*IndexFlat, error) {
	var index IndexFlat
	if c := C.faiss_IndexFlat_new_with(
		&index.idx,
		C.idx_t(d),
		C.FaissMetricType(metric),
	); c != 0 {
		return nil, getLastError()
	}
	return &index, nil
}

// NewIndexFlatIP creates a new flat index with the inner product metric type.
func NewIndexFlatIP(d int) (*IndexFlat, error) {
	return NewIndexFlat(d, MetricInnerProduct)
}

func (idx *IndexFlat) D() int {
	return indexD(idx.idx)
}

func (idx *IndexFlat) Add(x []float32) error {
	return indexAdd(idx.idx, x)
}

func (idx *IndexFlat) AddWithIDs(x []float32, xids []int64) error {
	return indexAddWithIDs(idx.idx, x, xids)
}

func (idx *IndexFlat) Search(x []float32, k int) (
	distances []float32, labels []int64, err error,
) {
	return indexSearch(idx.idx, x, k)
}

func (idx *IndexFlat) RemoveIDs(sel *IDSelector) (int, error) {
	return indexRemoveIDs(idx.idx, sel.sel)
}

func (idx *IndexFlat) Delete() {
	indexDelete(idx.idx)
}
