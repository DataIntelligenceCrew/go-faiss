// Package faiss provides bindings to Faiss, a library for vector similarity
// search.
// More detailed documentation can be found at the Faiss wiki:
// https://github.com/facebookresearch/faiss/wiki.
package faiss

/*
#cgo LDFLAGS: -lfaiss_c

#include <stdlib.h>
#include <faiss/c_api/AutoTune_c.h>
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
// AutoTune
//--------------------------------------------------

type ParameterSpace struct {
	ps *C.FaissParameterSpace
}

// NewParameterSpace creates a new ParameterSpace.
func NewParameterSpace() (*ParameterSpace, error) {
	var ps *C.FaissParameterSpace
	if c := C.faiss_ParameterSpace_new(&ps); c != 0 {
		return nil, getLastError()
	}
	return &ParameterSpace{ps}, nil
}

// SetIndexParameter sets one of the parameters.
func (p *ParameterSpace) SetIndexParameter(idx Index, name string, val float64) error {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	c := C.faiss_ParameterSpace_set_index_parameter(
		p.ps, idx.cPtr(), cname, C.double(val))
	if c != 0 {
		return getLastError()
	}
	return nil
}

// Delete frees the memory associated with p.
func (p *ParameterSpace) Delete() {
	C.faiss_ParameterSpace_free(p.ps)
}

//--------------------------------------------------
// AuxIndexStructures
//--------------------------------------------------

// RangeSearchResult is the result of a range search.
type RangeSearchResult struct {
	rsr *C.FaissRangeSearchResult
}

// Nq returns the number of queries.
func (r *RangeSearchResult) Nq() int {
	return int(C.faiss_RangeSearchResult_nq(r.rsr))
}

// Lims returns a slice containing start and end indices for queries in the
// distances and labels slices returned by Labels.
func (r *RangeSearchResult) Lims() []int {
	var lims *C.size_t
	C.faiss_RangeSearchResult_lims(r.rsr, &lims)
	length := r.Nq() + 1
	return (*[1 << 30]int)(unsafe.Pointer(lims))[:length:length]
}

// Labels returns the unsorted IDs and respective distances for each query.
// The result for query i is labels[lims[i]:lims[i+1]].
func (r *RangeSearchResult) Labels() (labels []int64, distances []float32) {
	lims := r.Lims()
	length := lims[len(lims)-1]
	var clabels *C.idx_t
	var cdist *C.float
	C.faiss_RangeSearchResult_labels(r.rsr, &clabels, &cdist)
	labels = (*[1 << 30]int64)(unsafe.Pointer(clabels))[:length:length]
	distances = (*[1 << 30]float32)(unsafe.Pointer(cdist))[:length:length]
	return
}

// Delete frees the memory associated with r.
func (r *RangeSearchResult) Delete() {
	C.faiss_RangeSearchResult_free(r.rsr)
}

// IDSelector represents a set of IDs to remove.
type IDSelector struct {
	sel *C.FaissIDSelector
}

// NewIDSelectorRange creates a selector that removes IDs on [imin, imax).
func NewIDSelectorRange(imin, imax int64) (*IDSelector, error) {
	var sel *C.FaissIDSelectorRange
	c := C.faiss_IDSelectorRange_new(&sel, C.idx_t(imin), C.idx_t(imax))
	if c != 0 {
		return nil, getLastError()
	}
	return &IDSelector{(*C.FaissIDSelector)(sel)}, nil
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

// Index is a Faiss index.
//
// Note that some index implementations do not support all methods.
// Check the Faiss wiki to see what operations an index supports.
type Index interface {
	// D returns the dimension of the indexed vectors.
	D() int

	// IsTrained returns true if the index has been trained or does not require
	// training.
	IsTrained() bool

	// Ntotal returns the number of indexed vectors.
	Ntotal() int64

	// MetricType returns the metric type of the index.
	MetricType() int

	// Train trains the index on a representative set of vectors.
	Train(x []float32) error

	// Add adds vectors to the index.
	Add(x []float32) error

	// AddWithIDs is like Add, but stores xids instead of sequential IDs.
	AddWithIDs(x []float32, xids []int64) error

	// Search queries the index with the vectors in x.
	// Returns the IDs of the k nearest neighbors for each query vector and the
	// corresponding distances.
	Search(x []float32, k int64) (distances []float32, labels []int64, err error)

	// RangeSearch queries the index with the vectors in x.
	// Returns all vectors with distance < radius.
	RangeSearch(x []float32, radius float32) (*RangeSearchResult, error)

	// Reset removes all vectors from the index.
	Reset() error

	// RemoveIDs removes the vectors specified by sel from the index.
	// Returns the number of elements removed and error.
	RemoveIDs(sel *IDSelector) (int, error)

	// Delete frees the memory used by the index.
	Delete()

	cPtr() *C.FaissIndex
}

type faissIndex struct {
	idx *C.FaissIndex
}

func (idx *faissIndex) cPtr() *C.FaissIndex {
	return idx.idx
}

func (idx *faissIndex) D() int {
	return int(C.faiss_Index_d(idx.idx))
}

func (idx *faissIndex) IsTrained() bool {
	return C.faiss_Index_is_trained(idx.idx) != 0
}

func (idx *faissIndex) Ntotal() int64 {
	return int64(C.faiss_Index_ntotal(idx.idx))
}

func (idx *faissIndex) MetricType() int {
	return int(C.faiss_Index_metric_type(idx.idx))
}

func (idx *faissIndex) Train(x []float32) error {
	n := len(x) / idx.D()
	if c := C.faiss_Index_train(idx.idx, C.idx_t(n), (*C.float)(&x[0])); c != 0 {
		return getLastError()
	}
	return nil
}

func (idx *faissIndex) Add(x []float32) error {
	n := len(x) / idx.D()
	if c := C.faiss_Index_add(idx.idx, C.idx_t(n), (*C.float)(&x[0])); c != 0 {
		return getLastError()
	}
	return nil
}

func (idx *faissIndex) AddWithIDs(x []float32, xids []int64) error {
	n := len(x) / idx.D()
	if c := C.faiss_Index_add_with_ids(
		idx.idx,
		C.idx_t(n),
		(*C.float)(&x[0]),
		(*C.idx_t)(&xids[0]),
	); c != 0 {
		return getLastError()
	}
	return nil
}

func (idx *faissIndex) Search(x []float32, k int64) (
	distances []float32, labels []int64, err error,
) {
	n := len(x) / idx.D()
	distances = make([]float32, int64(n)*k)
	labels = make([]int64, int64(n)*k)
	if c := C.faiss_Index_search(
		idx.idx,
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

func (idx *faissIndex) RangeSearch(x []float32, radius float32) (
	*RangeSearchResult, error,
) {
	n := len(x) / idx.D()
	var rsr *C.FaissRangeSearchResult
	if c := C.faiss_RangeSearchResult_new(&rsr, C.idx_t(n)); c != 0 {
		return nil, getLastError()
	}
	if c := C.faiss_Index_range_search(
		idx.idx,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.float(radius),
		rsr,
	); c != 0 {
		return nil, getLastError()
	}
	return &RangeSearchResult{rsr}, nil
}

func (idx *faissIndex) Reset() error {
	if c := C.faiss_Index_reset(idx.idx); c != 0 {
		return getLastError()
	}
	return nil
}

func (idx *faissIndex) RemoveIDs(sel *IDSelector) (int, error) {
	var nRemoved C.size_t
	if c := C.faiss_Index_remove_ids(idx.idx, sel.sel, &nRemoved); c != 0 {
		return 0, getLastError()
	}
	return int(nRemoved), nil
}

func (idx *faissIndex) Delete() {
	C.faiss_Index_free(idx.idx)
}

// IndexImpl is an abstract structure for an index.
type IndexImpl struct {
	Index
}

// AsFlat casts idx to a flat index.
// AsFlat panics if idx is not a flat index.
func (idx *IndexImpl) AsFlat() *IndexFlat {
	ptr := C.faiss_IndexFlat_cast(idx.cPtr())
	if ptr == nil {
		panic("index is not a flat index")
	}
	return &IndexFlat{&faissIndex{ptr}}
}

//--------------------------------------------------
// index_factory
//--------------------------------------------------

// IndexFactory builds a composite index.
// description is a comma-separated list of components.
func IndexFactory(d int, description string, metric int) (*IndexImpl, error) {
	cdesc := C.CString(description)
	defer C.free(unsafe.Pointer(cdesc))
	var idx faissIndex
	c := C.faiss_index_factory(&idx.idx, C.int(d), cdesc, C.FaissMetricType(metric))
	if c != 0 {
		return nil, getLastError()
	}
	return &IndexImpl{&idx}, nil
}

//--------------------------------------------------
// IndexFlat
//--------------------------------------------------

// IndexFlat is an index that stores the full vectors and performs exhaustive
// search.
type IndexFlat struct {
	Index
}

// NewIndexFlat creates a new flat index.
func NewIndexFlat(d int, metric int) (*IndexFlat, error) {
	var idx faissIndex
	if c := C.faiss_IndexFlat_new_with(
		&idx.idx,
		C.idx_t(d),
		C.FaissMetricType(metric),
	); c != 0 {
		return nil, getLastError()
	}
	return &IndexFlat{&idx}, nil
}

// NewIndexFlatIP creates a new flat index with the inner product metric type.
func NewIndexFlatIP(d int) (*IndexFlat, error) {
	return NewIndexFlat(d, MetricInnerProduct)
}

// NewIndexFlatL2 creates a new flat index with the L2 metric type.
func NewIndexFlatL2(d int) (*IndexFlat, error) {
	return NewIndexFlat(d, MetricL2)
}

// Xb returns the index's vectors.
// The returned slice becomes invalid after any add or remove operation.
func (idx *IndexFlat) Xb() []float32 {
	var size C.size_t
	var ptr *C.float
	C.faiss_IndexFlat_xb(idx.cPtr(), &ptr, &size)
	return (*[1 << 30]float32)(unsafe.Pointer(ptr))[:size:size]
}
