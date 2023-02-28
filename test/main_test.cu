#include <iostream>
#include <cugemm.hpp>
#include <mateval/comparison_cuda.hpp>


template <class T> double get_error_threshold() {return 0;};
template <> double get_error_threshold<double>() {return 1e-14;};
template <> double get_error_threshold<float >() {return 1e-6;};
template <> double get_error_threshold<half  >() {return 1e-2;};

template <class T>
void eval_matmul(
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const unsigned m,
		const unsigned n,
		const unsigned k
		) {
	T *mat_a, *mat_b, *mat_c;
	cudaMallocManaged(&mat_a, sizeof(T) * m * k);
	cudaMallocManaged(&mat_b, sizeof(T) * n * k);
	cudaMallocManaged(&mat_c, sizeof(T) * m * n);

	for (unsigned i = 0; i < m * k; i++) {
		mat_a[i] = static_cast<double>(m * k / 2 - i) * 1e-3;
	}

	for (unsigned i = 0; i < n * k; i++) {
		mat_b[i] = static_cast<double>(n * k / 2 - i) * 1e-3;
	}

	const T alpha = static_cast<T>(1);
	const T beta  = static_cast<T>(0);

	mtk::cugemm::gemm<T>(
			nullptr,
			op_A,
			op_B,
			m, n, k,
			&alpha,
			mat_a, (op_A == CUBLAS_OP_N ? m : k),
			mat_b, (op_B == CUBLAS_OP_N ? k : n),
			&beta,
			mat_c, m
			);

	const auto error = mtk::mateval::cuda::get_error_AxB(
			mtk::mateval::avg_relative_error,
			m, n, k,
			(op_A == CUBLAS_OP_N ? mtk::mateval::col_major : mtk::mateval::row_major),
			(op_B == CUBLAS_OP_N ? mtk::mateval::col_major : mtk::mateval::row_major),
			mtk::mateval::col_major,
			mat_a, (op_A == CUBLAS_OP_N ? m : k),
			mat_b, (op_B == CUBLAS_OP_N ? k : n),
			mat_c, m
			);
	cudaDeviceSynchronize();
	std::printf("%u,%u,%u,%c,%c,%e,%s\n",
			m, n, k,
			(op_A == CUBLAS_OP_N ? 'N' : 'T'),
			(op_B == CUBLAS_OP_N ? 'N' : 'T'),
			error.at(mtk::mateval::avg_relative_error),
			error.at(mtk::mateval::avg_relative_error) < get_error_threshold<T>() ? "OK" : "NG"
			);

	cudaFree(mat_a);
	cudaFree(mat_b);
	cudaFree(mat_c);
}

int main() {
	eval_matmul<double>(CUBLAS_OP_N, CUBLAS_OP_N, 1024, 1024, 1024);
	eval_matmul<double>(CUBLAS_OP_T, CUBLAS_OP_N, 1024, 1024, 1024);
	eval_matmul<double>(CUBLAS_OP_N, CUBLAS_OP_T, 1024, 1024, 1024);
	eval_matmul<double>(CUBLAS_OP_T, CUBLAS_OP_T, 1024, 1024, 1024);
	eval_matmul<float >(CUBLAS_OP_N, CUBLAS_OP_N, 1024, 1024, 1024);
	eval_matmul<float >(CUBLAS_OP_T, CUBLAS_OP_N, 1024, 1024, 1024);
	eval_matmul<float >(CUBLAS_OP_N, CUBLAS_OP_T, 1024, 1024, 1024);
	eval_matmul<float >(CUBLAS_OP_T, CUBLAS_OP_T, 1024, 1024, 1024);
	eval_matmul<half  >(CUBLAS_OP_N, CUBLAS_OP_N, 1024, 1024, 1024);
	eval_matmul<half  >(CUBLAS_OP_T, CUBLAS_OP_N, 1024, 1024, 1024);
	eval_matmul<half  >(CUBLAS_OP_N, CUBLAS_OP_T, 1024, 1024, 1024);
	eval_matmul<half  >(CUBLAS_OP_T, CUBLAS_OP_T, 1024, 1024, 1024);
}
