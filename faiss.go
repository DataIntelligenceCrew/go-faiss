// Package faiss provides bindings to Faiss, a library for vector similarity
// search.
// More detailed documentation can be found at the Faiss wiki:
// https://github.com/facebookresearch/faiss/wiki.
package faiss

/*
#cgo LDFLAGS: -lfaiss_c

#include <faiss/c_api/Index_c.h>
#include <faiss/c_api/error_c.h>
*/
import "C"
import "errors"

func getLastError() error {
	return errors.New(C.GoString(C.faiss_get_last_error()))
}

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
