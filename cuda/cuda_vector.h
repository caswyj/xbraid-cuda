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
 * \file cuda_vector.h
 * \brief CUDA vector operations for XBraid.
 *
 * This file defines vector operations implemented as CUDA kernels
 * for efficient GPU execution.
 */

#ifndef CUDA_VECTOR_H
#define CUDA_VECTOR_H

#include "cuda_braid.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Kernel prototypes
 *--------------------------------------------------------------------------*/

/* Vector copy kernel */
__global__ void
cuda_kernel_vec_copy(braid_Real* dst, const braid_Real* src,
                     braid_Int n, braid_Int stream_idx);

/* Vector fill (set to constant) kernel */
__global__ void
cuda_kernel_vec_fill(braid_Real* dst, braid_Real alpha, braid_Int n);

/* Vector AXPY kernel: y = alpha*x + beta*y */
__global__ void
cuda_kernel_vec_axpy(braid_Real* y, const braid_Real* x,
                     braid_Real alpha, braid_Real beta, braid_Int n);

/* Vector copy with index mapping */
__global__ void
cuda_kernel_vec_copy_indexed(braid_Real* dst, const braid_Real* src,
                             const braid_Int* idx_map, braid_Int n);

/* Reduction kernel (sum) */
__global__ void
cuda_kernel_reduce_sum(braid_Real* input, braid_Real* output,
                       braid_Int n, braid_Real* block_sums);

/* Reduction kernel (max) */
__global__ void
cuda_kernel_reduce_max(braid_Real* input, braid_Real* output,
                       braid_Int n, braid_Real* block_max);

/* Spatial norm (L2) kernel */
__global__ void
cuda_kernel_norm_l2(braid_Real* vec, braid_Real* result, braid_Int n);

/* Spatial norm (L1) kernel */
__global__ void
cuda_kernel_norm_l1(braid_Real* vec, braid_Real* result, braid_Int n);

/* Spatial norm (L-inf) kernel */
__global__ void
cuda_kernel_norm_linf(braid_Real* vec, braid_Real* result, braid_Int n);

/*--------------------------------------------------------------------------
 * Host wrapper functions
 *--------------------------------------------------------------------------*/

/* Vector copy */
braid_Int
_cuda_VectorCopy(cuda_Vector cx, cuda_Vector cy);

/* Vector set to constant */
braid_Int
_cuda_VectorSet(cuda_Vector cx, braid_Real alpha);

/* Vector AXPY: y = alpha*x + beta*y */
braid_Int
_cuda_VectorAXPY(cuda_Vector cx, braid_Real alpha, cuda_Vector cy,
                 braid_Real beta);

/* Vector L2 norm */
braid_Int
_cuda_VectorNormL2(cuda_Vector cx, braid_Real* norm);

/* Vector L1 norm */
braid_Int
_cuda_VectorNormL1(cuda_Vector cx, braid_Real* norm);

/* Vector L-inf norm */
braid_Int
_cuda_VectorNormLinf(cuda_Vector cx, braid_Real* norm);

/* Vector dot product */
braid_Int
_cuda_VectorDot(cuda_Vector cx, cuda_Vector cy, braid_Real* dot);

/*--------------------------------------------------------------------------
 * Reduction utilities
 *--------------------------------------------------------------------------*/

/* Perform block-level reduction */
braid_Int
_cuda_ReduceBlock(braid_Real* d_data, braid_Int n, braid_Real* d_block_results,
                  braid_Int* num_blocks);

/* Perform grid-level reduction */
braid_Int
_cuda_ReduceGrid(braid_Real* d_block_results, braid_Int num_blocks,
                 braid_Real* d_result);

/* Copy result to host */
braid_Int
_cuda_ReduceCopyToHost(braid_Real* h_result, braid_Real* d_result);

/*--------------------------------------------------------------------------
 * Memory utilities
 *--------------------------------------------------------------------------*/

/* Allocate device memory for vector */
braid_Int
_cuda_MemAlloc(void** d_ptr, braid_Int nbytes);

/* Free device memory */
braid_Int
_cuda_MemFree(void* d_ptr);

/* Copy host to device */
braid_Int
_cuda_MemcpyH2D(void* d_ptr, const void* h_ptr, braid_Int nbytes);

/* Copy device to host */
braid_Int
_cuda_MemcpyD2H(void* h_ptr, const void* d_ptr, braid_Int nbytes);

/* Set device memory to zero */
braid_Int
_cuda_Memset(void* d_ptr, int value, braid_Int nbytes);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_VECTOR_H */
