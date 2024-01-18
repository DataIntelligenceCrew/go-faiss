package faiss

/*
#include <stdlib.h>
#include <stdio.h>
#include <faiss/c_api/index_io_c.h>
#include <faiss/c_api/index_io_c_ex.h>
*/
import "C"
import (
	"unsafe"
)

// WriteIndex writes an index to a file.
func WriteIndex(idx Index, filename string) error {
	cfname := C.CString(filename)
	defer C.free(unsafe.Pointer(cfname))
	if c := C.faiss_write_index_fname(idx.cPtr(), cfname); c != 0 {
		return getLastError()
	}
	return nil
}

func WriteIndexIntoBuffer(idx Index) ([]byte, error) {
	// the values to be returned by the faiss APIs
	tempBuf := (*C.uchar)(C.malloc(C.size_t(0)))
	bufSize := C.int(0)

	if c := C.faiss_write_index_buf(
		idx.cPtr(),
		&bufSize,
		&tempBuf,
	); c != 0 {
		return nil, getLastError()
	}

	// at this point, the idx has a valid ref count. furthermore, the index is
	// something that's present on the C memory space, so not available to go's
	// GC. needs to be freed when its of no more use.

	// todo: get a better upper bound.
	// todo: add checksum.
	// the content populated in the tempBuf is converted from *C.uchar to unsafe.Pointer
	// and then the pointer is casted into a large byte slice which is then sliced
	// to a length and capacity equal to bufSize returned across the cgo interface.
	// NOTE: it still points to the C memory though
	val := (*[1 << 32]byte)(unsafe.Pointer(tempBuf))[:int(bufSize):int(bufSize)]
	rv := make([]byte, len(val))
	// an explicit copy is necessary to free the memory on C heap and then return
	// the rv back to the caller which is definitely on goruntime space (which will
	// GC'd later on).
	//
	// an optimization over here - create buffer pool which can be used to make the
	// memory allocations cheaper. specifically two separate pools can be utilized,
	// one for C pointers and another for goruntime. within the faiss_write_index_buf
	// a cheaper calloc rather than malloc can be used to make any extra allocations
	// cheaper.
	copy(rv, val)
	// safe to free the c memory allocated while serializing the index, and the rv
	// is something that's present in go runtime so different address space altogether
	// p.s: no need to free "val" since the underlying memory is same for both the
	// vars
	C.free(unsafe.Pointer(tempBuf))
	return rv, nil
}

func ReadIndexFromBuffer(buf []byte, ioflags int) (*IndexImpl, error) {
	ptr := (*C.uchar)(unsafe.Pointer(&buf[0]))
	size := C.int(len(buf))

	// the idx var has C.FaissIndex within the struct which is nil as of now.
	var idx faissIndex
	if c := C.faiss_read_index_buf(ptr,
		size,
		C.int(ioflags),
		&idx.idx); c != 0 {
		return nil, getLastError()
	}

	ptr = nil

	// after exiting the faiss_read_index_buf, the ref count to the memory allocated
	// for the freshly created faiss::index becomes 1 (held by idx.idx of type C.FaissIndex)
	// this is allocated on the C heap, so not available for golang's GC. hence needs
	// to be cleaned up after the index is longer being used - to be done at zap layer.
	return &IndexImpl{&idx}, nil
}

const (
	IOFlagMmap     = C.FAISS_IO_FLAG_MMAP
	IOFlagReadOnly = C.FAISS_IO_FLAG_READ_ONLY
)

// ReadIndex reads an index from a file.
func ReadIndex(filename string, ioflags int) (*IndexImpl, error) {
	cfname := C.CString(filename)
	defer C.free(unsafe.Pointer(cfname))
	var idx faissIndex
	if c := C.faiss_read_index_fname(cfname, C.int(ioflags), &idx.idx); c != 0 {
		return nil, getLastError()
	}
	return &IndexImpl{&idx}, nil
}
