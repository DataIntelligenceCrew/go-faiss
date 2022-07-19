package faiss

/*
#include <faiss/c_api/IndexIVF_c.h>
FaissIndexIVF* convert(FaissIndex *idx) {
	return (FaissIndexIVF*)(idx);
}
*/
import "C"
import "errors"

func SetNumProbes(index Index, numProbes int) error {
	var ivfidx *C.FaissIndexIVF
	ivfidx = C.convert(index.cPtr())
	if ivfidx == nil {
		return errors.New("Index cannot be converted to ivf")
	}
	C.faiss_IndexIVF_set_nprobe(ivfidx, C.size_t(numProbes))
	return nil
}
