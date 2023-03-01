#include <cugemm.hpp>

template <class T>
__global__ void gemm_kernel(
		cublasOperation_t op_A,
		cublasOperation_t op_B,
		const unsigned m,
		const unsigned n,
		const unsigned k,
		const T alpha,
		const T* const A_ptr, const unsigned lda,
		const T* const B_ptr, const unsigned ldb,
		const T beta,
		T* const C_ptr, const unsigned ldc
		) {
	const auto tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= m * n) {
		return;
	}
	const auto mi = tid % m;
	const auto ni = tid / m;

	T sum = 0;
	for (unsigned ki_global = 0; ki_global < k; ki_global += 8) {

		T local_sum = 0;
		unsigned ki;
		for (unsigned ki_local = 0; ki_local < 8 && (ki = ki_global + ki_local) < k; ki_local++) {
			const std::size_t A_offset = (op_A == CUBLAS_OP_N ? (mi + ki * lda) : (ki + mi * lda));
			const std::size_t B_offset = (op_B == CUBLAS_OP_N ? (ki + ni * ldb) : (ni + ki * ldb));
			local_sum += A_ptr[A_offset] * B_ptr[B_offset];
		}
		sum += local_sum;
	}

	if (beta == static_cast<T>(0)) {
		C_ptr[mi + ni * ldc] = alpha * sum;
	} else {
		C_ptr[mi + ni * ldc] = alpha * sum + beta * C_ptr[mi + ni * ldc];
	}
}

template <class T>
cublasStatus_t mtk::cugemm::gemm(
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
		) {
	const auto block_size = 256lu;
	const auto grid_size = (m * n + block_size - 1) / block_size;

	cudaStream_t cuda_stream = 0;
	if (cublas_handle != nullptr) {
		cublasGetStream(cublas_handle, &cuda_stream);
	}

	gemm_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
			op_A, op_B,
			m, n, k,
			*alpha,
			A_ptr, lda,
			B_ptr, ldb,
			*beta,
			C_ptr, ldc
			);

	return CUBLAS_STATUS_SUCCESS;
}

#define GEMM_INSTANCE(T) \
template cublasStatus_t mtk::cugemm::gemm<T>(cublasHandle_t, cublasOperation_t, cublasOperation_t, const unsigned, const unsigned, const unsigned, const T* const, const T* const, const unsigned, const T* const, const unsigned, const T* const, T* const, const unsigned)
GEMM_INSTANCE(double);
GEMM_INSTANCE(float );
GEMM_INSTANCE(half  );
