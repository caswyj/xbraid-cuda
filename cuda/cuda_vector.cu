/*BHEADER**********************************************************************
 * Copyright (c) 2013, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * This file is part of XBraid. For support, post issues to the XBraid Github page.
 *
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE. See the terms and conditions of the GNU General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc., 59
 * Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 ***********************************************************************EHEADER*/

/**
 * \file cuda_vector.cu
 * \brief CUDA vector operations for XBraid.
 *
 * This file implements vector operations as CUDA kernels for efficient
 * GPU execution, including copy, axpy, and norm computations.
 */

#include "cuda_vector.h"
#include "cuda_util.h"
#include "_braid.h"

/*--------------------------------------------------------------------------
 * Kernel implementations
 *--------------------------------------------------------------------------*/

/* Vector copy kernel: dst[i] = src[i] */
__global__ void
cuda_kernel_vec_copy(braid_Real* dst, const braid_Real* src,
                     braid_Int n)
{
    braid_Int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        dst[idx] = src[idx];
    }
}

/* Vector fill (set to constant) kernel: dst[i] = alpha */
__global__ void
cuda_kernel_vec_fill(braid_Real* dst, braid_Real alpha, braid_Int n)
{
    braid_Int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        dst[idx] = alpha;
    }
}

/* Vector AXPY kernel: y[i] = alpha*x[i] + beta*y[i] */
__global__ void
cuda_kernel_vec_axpy(braid_Real* y, const braid_Real* x,
                     braid_Real alpha, braid_Real beta, braid_Int n)
{
    braid_Int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        y[idx] = alpha * x[idx] + beta * y[idx];
    }
}

/* Vector copy with index mapping */
__global__ void
cuda_kernel_vec_copy_indexed(braid_Real* dst, const braid_Real* src,
                             const braid_Int* idx_map, braid_Int n)
{
    braid_Int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        braid_Int src_idx = idx_map[idx];
        dst[idx] = src[src_idx];
    }
}

/* Block-level reduction kernel (sum) */
__global__ void
cuda_kernel_reduce_sum(braid_Real* input, braid_Real* output,
                       braid_Int n, braid_Real* block_sums)
{
    extern __shared__ braid_Real sdata[];
    braid_Int tid = threadIdx.x;
    braid_Int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Load into shared memory */
    braid_Real val = (idx < n) ? input[idx] : 0.0;
    sdata[tid] = val;
    __syncthreads();

    /* Reduce in shared memory */
    for (braid_Int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* Write block result */
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

/* Block-level reduction kernel (max) */
__global__ void
cuda_kernel_reduce_max(braid_Real* input, braid_Real* output,
                       braid_Int n, braid_Real* block_max)
{
    extern __shared__ braid_Real sdata[];
    braid_Int tid = threadIdx.x;
    braid_Int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Load into shared memory */
    braid_Real val = (idx < n) ? input[idx] : -1e300;
    sdata[tid] = val;
    __syncthreads();

    /* Reduce in shared memory */
    for (braid_Int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            braid_Real a = sdata[tid];
            braid_Real b = sdata[tid + s];
            sdata[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }

    /* Write block result */
    if (tid == 0) {
        block_max[blockIdx.x] = sdata[0];
    }
}

/* Spatial norm (L2) kernel */
__global__ void
cuda_kernel_norm_l2(braid_Real* vec, braid_Real* result, braid_Int n)
{
    extern __shared__ braid_Real sdata[];
    braid_Int tid = threadIdx.x;
    braid_Int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Load into shared memory */
    braid_Real val = (idx < n) ? vec[idx] * vec[idx] : 0.0;
    sdata[tid] = val;
    __syncthreads();

    /* Reduce in shared memory */
    for (braid_Int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* Write block result */
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

/* Spatial norm (L1) kernel */
__global__ void
cuda_kernel_norm_l1(braid_Real* vec, braid_Real* result, braid_Int n)
{
    extern __shared__ braid_Real sdata[];
    braid_Int tid = threadIdx.x;
    braid_Int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Load into shared memory */
    braid_Real val = (idx < n) ? fabs(vec[idx]) : 0.0;
    sdata[tid] = val;
    __syncthreads();

    /* Reduce in shared memory */
    for (braid_Int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* Write block result */
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

/* Spatial norm (L-inf) kernel */
__global__ void
cuda_kernel_norm_linf(braid_Real* vec, braid_Real* result, braid_Int n)
{
    extern __shared__ braid_Real sdata[];
    braid_Int tid = threadIdx.x;
    braid_Int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Load into shared memory */
    braid_Real val = (idx < n) ? fabs(vec[idx]) : 0.0;
    sdata[tid] = val;
    __syncthreads();

    /* Reduce in shared memory */
    for (braid_Int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            braid_Real a = sdata[tid];
            braid_Real b = sdata[tid + s];
            sdata[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }

    /* Write block result */
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

/* Vector dot product kernel */
__global__ void
cuda_kernel_vec_dot(const braid_Real* x, const braid_Real* y,
                    braid_Real* result, braid_Int n)
{
    extern __shared__ braid_Real sdata[];
    braid_Int tid = threadIdx.x;
    braid_Int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Load into shared memory */
    braid_Real val = (idx < n) ? x[idx] * y[idx] : 0.0;
    sdata[tid] = val;
    __syncthreads();

    /* Reduce in shared memory */
    for (braid_Int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* Write block result */
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

/*--------------------------------------------------------------------------
 * Host wrapper implementations
 *--------------------------------------------------------------------------*/

braid_Int
_cuda_VectorCopy(cuda_Vector cx, cuda_Vector cy)
{
    if (cx->size != cy->size) {
        fprintf(stderr, "Vector size mismatch in _cuda_VectorCopy\n");
        return 1;
    }

    braid_Int n = cx->size / sizeof(braid_Real);

    if (n <= 0) {
        return _braid_error_flag;
    }

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    cuda_kernel_vec_copy<<<grid_size, block_size>>>(
        (braid_Real*)cy->d_ptr,
        (const braid_Real*)cx->d_ptr,
        n
    );

    CUDA_CHECK(cudaGetLastError());

    return _braid_error_flag;
}

braid_Int
_cuda_VectorSet(cuda_Vector cx, braid_Real alpha)
{
    braid_Int n = cx->size / sizeof(braid_Real);

    if (n <= 0) {
        return _braid_error_flag;
    }

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    cuda_kernel_vec_fill<<<grid_size, block_size>>>(
        (braid_Real*)cx->d_ptr,
        alpha, n
    );

    CUDA_CHECK(cudaGetLastError());

    return _braid_error_flag;
}

braid_Int
_cuda_VectorAXPY(cuda_Vector cx, braid_Real alpha, cuda_Vector cy,
                 braid_Real beta)
{
    if (cx->size != cy->size) {
        fprintf(stderr, "Vector size mismatch in _cuda_VectorAXPY\n");
        return 1;
    }

    braid_Int n = cx->size / sizeof(braid_Real);

    if (n <= 0) {
        return _braid_error_flag;
    }

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    cuda_kernel_vec_axpy<<<grid_size, block_size>>>(
        (braid_Real*)cy->d_ptr,
        (const braid_Real*)cx->d_ptr,
        alpha, beta, n
    );

    CUDA_CHECK(cudaGetLastError());

    return _braid_error_flag;
}

braid_Int
_cuda_ReduceBlock(braid_Real* d_data, braid_Int n, braid_Real* d_block_results,
                  braid_Int* num_blocks)
{
    if (n <= 0) {
        *num_blocks = 0;
        return _braid_error_flag;
    }

    int block_size = 256;
    *num_blocks = (n + block_size - 1) / block_size;

    size_t shared_mem = block_size * sizeof(braid_Real);

    cuda_kernel_reduce_sum<<<*num_blocks, block_size, shared_mem>>>(
        d_data, NULL, n, d_block_results
    );

    CUDA_CHECK(cudaGetLastError());

    return _braid_error_flag;
}

braid_Int
_cuda_ReduceGrid(braid_Real* d_block_results, braid_Int num_blocks,
                 braid_Real* d_result)
{
    if (num_blocks <= 0) {
        return _braid_error_flag;
    }

    int block_size = (num_blocks < 1024) ? num_blocks : 1024;
    size_t shared_mem = block_size * sizeof(braid_Real);

    cuda_kernel_reduce_sum<<<1, block_size, shared_mem>>>(
        d_block_results, NULL, num_blocks, d_result
    );

    CUDA_CHECK(cudaGetLastError());

    return _braid_error_flag;
}

braid_Int
_cuda_ReduceCopyToHost(braid_Real* h_result, braid_Real* d_result)
{
    return _cuda_MemcpyD2H(h_result, d_result, sizeof(braid_Real));
}

braid_Int
_cuda_VectorNormL2(cuda_Vector cx, braid_Real* norm)
{
    braid_Int n = cx->size / sizeof(braid_Real);

    if (n <= 0) {
        *norm = 0.0;
        return _braid_error_flag;
    }

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    /* Allocate temporary storage for block sums */
    braid_Real* d_block_sums;
    _cuda_MemAlloc((void**)&d_block_sums, num_blocks * sizeof(braid_Real));

    /* Compute squared values and reduce */
    cuda_kernel_norm_l2<<<num_blocks, block_size>>>(
        (braid_Real*)cx->d_ptr, d_block_sums, n
    );

    CUDA_CHECK(cudaGetLastError());

    /* Reduce block sums to single value */
    braid_Real* d_result;
    _cuda_MemAlloc((void**)&d_result, sizeof(braid_Real));

    _cuda_ReduceGrid(d_block_sums, num_blocks, d_result);

    /* Copy result to host and take square root */
    braid_Real h_result;
    _cuda_ReduceCopyToHost(&h_result, d_result);

    *norm = sqrt(h_result);

    /* Free temporary storage */
    _cuda_MemFree(d_block_sums);
    _cuda_MemFree(d_result);

    return _braid_error_flag;
}

braid_Int
_cuda_VectorNormL1(cuda_Vector cx, braid_Real* norm)
{
    braid_Int n = cx->size / sizeof(braid_Real);

    if (n <= 0) {
        *norm = 0.0;
        return _braid_error_flag;
    }

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    braid_Real* d_block_sums;
    _cuda_MemAlloc((void**)&d_block_sums, num_blocks * sizeof(braid_Real));

    cuda_kernel_norm_l1<<<num_blocks, block_size>>>(
        (braid_Real*)cx->d_ptr, d_block_sums, n
    );

    CUDA_CHECK(cudaGetLastError());

    braid_Real* d_result;
    _cuda_MemAlloc((void**)&d_result, sizeof(braid_Real));

    _cuda_ReduceGrid(d_block_sums, num_blocks, d_result);

    braid_Real h_result;
    _cuda_ReduceCopyToHost(&h_result, d_result);

    *norm = h_result;

    _cuda_MemFree(d_block_sums);
    _cuda_MemFree(d_result);

    return _braid_error_flag;
}

braid_Int
_cuda_VectorNormLinf(cuda_Vector cx, braid_Real* norm)
{
    braid_Int n = cx->size / sizeof(braid_Real);

    if (n <= 0) {
        *norm = 0.0;
        return _braid_error_flag;
    }

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    braid_Real* d_block_max;
    _cuda_MemAlloc((void**)&d_block_max, num_blocks * sizeof(braid_Real));

    cuda_kernel_norm_linf<<<num_blocks, block_size>>>(
        (braid_Real*)cx->d_ptr, d_block_max, n
    );

    CUDA_CHECK(cudaGetLastError());

    braid_Real* d_result;
    _cuda_MemAlloc((void**)&d_result, sizeof(braid_Real));

    _cuda_ReduceGrid(d_block_max, num_blocks, d_result);

    braid_Real h_result;
    _cuda_ReduceCopyToHost(&h_result, d_result);

    *norm = h_result;

    _cuda_MemFree(d_block_max);
    _cuda_MemFree(d_result);

    return _braid_error_flag;
}

braid_Int
_cuda_VectorDot(cuda_Vector cx, cuda_Vector cy, braid_Real* dot)
{
    if (cx->size != cy->size) {
        fprintf(stderr, "Vector size mismatch in _cuda_VectorDot\n");
        return 1;
    }

    braid_Int n = cx->size / sizeof(braid_Real);

    if (n <= 0) {
        *dot = 0.0;
        return _braid_error_flag;
    }

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    braid_Real* d_block_sums;
    _cuda_MemAlloc((void**)&d_block_sums, num_blocks * sizeof(braid_Real));

    cuda_kernel_vec_dot<<<num_blocks, block_size>>>(
        (const braid_Real*)cx->d_ptr,
        (const braid_Real*)cy->d_ptr,
        d_block_sums, n
    );

    CUDA_CHECK(cudaGetLastError());

    braid_Real* d_result;
    _cuda_MemAlloc((void**)&d_result, sizeof(braid_Real));

    _cuda_ReduceGrid(d_block_sums, num_blocks, d_result);

    braid_Real h_result;
    _cuda_ReduceCopyToHost(&h_result, d_result);

    *dot = h_result;

    _cuda_MemFree(d_block_sums);
    _cuda_MemFree(d_result);

    return _braid_error_flag;
}
