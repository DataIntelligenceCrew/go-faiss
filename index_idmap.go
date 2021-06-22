package faiss

/*
#include <faiss/c_api/MetaIndexes_c.h>
*/
import "C"
import (
	"errors"
)

type IndexIDMapWrapper struct {
	Index
	cpointer **C.FaissIndex
}

func NewIndexIDMap(index *IndexFlat) (*IndexIDMapWrapper, error) {
	var indexMapPointer *C.FaissIndexIDMap
	var pointerToIndexMapPointer **C.FaissIndexIDMap
	pointerToIndexMapPointer = &indexMapPointer
	wrapper := IndexIDMapWrapper{cpointer: &indexMapPointer}
	if C.faiss_IndexIDMap_new(pointerToIndexMapPointer, index.cPtr()) != 0 {
		return nil, errors.New("Error occurred while initializing IndexIDMapWrapper")
	}
	wrapper.Index = &faissIndex{idx: *wrapper.cpointer}
	return &wrapper, nil
}
