#include <cstdint>
#include <cublas_v2.h>

namespace mtk {
namespace cugemm {
template <class T>
cublasStatus_t gemm(
		cublasHandle_t cublas_handle,
		cublasOperation_t op_A,
		cublasOperation_t op_B,
		const unsigned m,
		const unsigned n,
		const unsigned k,
		const T* const alpha,
		const T* const A_ptr, const unsigned lda,
		const T* const B_ptr, const unsigned ldb,
		const T* const beta,
		T* const C_ptr, const unsigned ldc
		);
} // namespace cugemm
} // namespace mtk
