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

	// todo: get a better upper bound.
	// todo: add checksum.
	// the content populated in the tempBuf is converted from *C.uchar to unsafe.Pointer
	// and then the pointer is casted into a large byte slice which is then sliced
	// to a length and capacity equal to bufSize returned across the cgo interface.
	val := (*[1 << 32]byte)(unsafe.Pointer(tempBuf))[:int(bufSize):int(bufSize)]
	return val, nil
}

func ReadIndexFromBuffer(buf []byte, ioflags int) (*IndexImpl, error) {
	ptr := C.CBytes(buf)
	size := C.int(len(buf))
	var idx faissIndex
	if c := C.faiss_read_index_buf((*C.uchar)(ptr),
		size,
		C.int(ioflags),
		&idx.idx); c != 0 {
		return nil, getLastError()
	}
	return &IndexImpl{&idx}, nil
}

// IO flags
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
