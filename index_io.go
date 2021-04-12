package faiss

/*
#include <stdlib.h>
#include <faiss/c_api/index_io_c.h>
*/
import "C"
import "unsafe"

func WriteIndex(idx Index, filename string) error {
	fnamePointer := C.CString(filename)
	defer C.free(unsafe.Pointer(fnamePointer))
	if c := C.faiss_write_index_fname(idx.cPtr(), fnamePointer); c != 0 {
		return getLastError()
	}
	return nil
}

const (
	IoFlagMmap      = C.FAISS_IO_FLAG_MMAP
	IoFlagReadOnly  = C.FAISS_IO_FLAG_READ_ONLY
)

func ReadIndex(filename string, ioflag C.int) (*IndexImpl, error) {
	var idx faissIndex
	fnamePointer := C.CString(filename)
	defer C.free(unsafe.Pointer(fnamePointer))
	if c := C.faiss_read_index_fname(fnamePointer, ioflag, &idx.idx); c != 0 {
		return nil, getLastError()
	}
	return &IndexImpl{&idx}, nil
}
