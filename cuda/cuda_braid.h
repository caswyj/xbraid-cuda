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
 * \file cuda_braid.h
 * \brief CUDA version of XBraid core structure and utility functions.
 *
 * This file defines the CUDA version of the core XBraid structure
 * and utility functions for GPU-based parallel-in-time computation.
 */

#ifndef CUDA_BRAID_H
#define CUDA_BRAID_H

#include "_braid.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * CUDA vector structure - wraps user vector with CUDA memory management
 */
typedef struct _cuda_Vector_struct {
    void* d_ptr;        /* Device pointer to vector data */
    void* h_ptr;        /* Host pointer (for staging) */
    braid_Int size;     /* Size in bytes */
    braid_Int owned;    /* Flag indicating if we own the memory */
} _cuda_Vector;

typedef _cuda_Vector* cuda_Vector;

/**
 * CUDA grid structure - mirrors _braid_Grid for GPU execution
 */
typedef struct {
    braid_Int level;
    braid_Int ilower;
    braid_Int iupper;
    braid_Int clower;
    braid_Int cupper;
    braid_Int gupper;
    braid_Int cfactor;
    braid_Int ncpoints;

    /* Device pointers to vectors and time values */
    cuda_Vector* ua_d;      /* Solution vectors on device */
    cuda_Vector* va_d;      /* Restricted unknown vectors */
    cuda_Vector* fa_d;      /* RHS vectors */
    braid_Real* ta_d;       /* Time values on device */

    /* Host pointers for staging */
    cuda_Vector* ua_h;
    cuda_Vector* va_h;
    cuda_Vector* fa_h;
    braid_Real* ta_h;

    /* Communication handles */
    braid_Int recv_index;
    braid_Int send_index;

    /* Memory allocation tracking */
    braid_Int ua_alloc_size;
    braid_Int ta_alloc_size;
} _cuda_Grid;

typedef _cuda_Grid* cuda_Grid;

/* Type definition for CUDA stream (must be before cuda_Core) */
typedef cudaStream_t cuda_Stream;

/**
 * CUDA core structure - mirrors _braid_Core for GPU execution
 */
typedef struct _cuda_Core_struct {
    /* MPI info (kept for compatibility) */
    MPI_Comm comm_world;
    MPI_Comm comm;
    braid_Int myid_world;
    braid_Int myid;

    /* Time domain */
    braid_Real tstart;
    braid_Real tstop;
    braid_Int ntime;

    /* User callbacks (host function pointers) */
    braid_PtFcnStep step;
    braid_PtFcnInit init;
    braid_PtFcnClone clone;
    braid_PtFcnFree free;
    braid_PtFcnSum sum;
    braid_PtFcnSpatialNorm spatialnorm;
    braid_PtFcnAccess access;
    braid_PtFcnBufSize bufsize;
    braid_PtFcnBufPack bufpack;
    braid_PtFcnBufUnpack bufunpack;
    braid_PtFcnResidual residual;
    braid_PtFcnInitBasis init_basis;
    braid_PtFcnInnerProd inner_prod;

    /* Configuration */
    braid_Int max_levels;
    braid_Int nlevels;
    braid_Int skip;
    braid_Int min_coarse;
    braid_Int relax_only_cg;
    braid_Int max_iter;
    braid_Int niter;
    braid_Int tnorm;
    braid_Real tol;
    braid_Int rtol;
    braid_Int access_level;
    braid_Int print_level;
    braid_Int seq_soln;
    braid_Int storage;
    braid_Int periodic;
    braid_Int initiali;
    braid_Int gupper;
    braid_Int nrefine;

    /* CF-relaxation parameters */
    braid_Int* nrels;
    braid_Real* CWts;
    braid_Int* cfactors;
    braid_Int nrdefault;
    braid_Real CWt_default;
    braid_Int cfdefault;

    /* Richardson error estimation */
    braid_Int richardson;
    braid_Int est_error;
    braid_Int order;
    braid_Real* dtk;
    braid_Real* estimate;

    /* Delta correction */
    braid_Int delta_correct;
    braid_Int delta_rank;
    braid_Int delta_defer_lvl;
    braid_Int delta_defer_iter;

    /* Adjoint/optimization */
    braid_Optim optim;
    braid_Int adjoint;
    braid_Int record;
    braid_Int obj_only;

    /* Timing */
    braid_Int timings;
    braid_Real timer_user_step;
    braid_Real timer_user_init;
    braid_Real timer_user_clone;
    braid_Real timer_user_free;
    braid_Real timer_user_sum;
    braid_Real timer_user_spatialnorm;

    /* Print file */
    FILE* printfile;

    /* CUDA-specific state */
    cuda_Stream stream;       /* CUDA stream for async operations */
    braid_Int device_id;      /* GPU device ID */

    /* CUDA grids - array of size nlevels */
    cuda_Grid* cuda_grids;

    /* Residual norm history */
    braid_Real* rnorms;
    braid_Real  rnorm0;
    braid_Real* full_rnorms;
    braid_Real  full_rnorm0;

} _cuda_Core;

typedef _cuda_Core* cuda_Core;

/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * Core management
 *--------------------------------------------------------------------------*/

/* Initialize CUDA core from regular core */
cuda_Core
_cuda_CoreInit(braid_Core core, braid_Int device_id);

/* Destroy CUDA core */
void
_cuda_CoreDestroy(cuda_Core ccore);

/*--------------------------------------------------------------------------
 * Grid management
 *--------------------------------------------------------------------------*/

/* Create CUDA grid from regular grid */
cuda_Grid
_cuda_GridInit(braid_Core core, braid_Int level);

/* Destroy CUDA grid */
void
_cuda_GridDestroy(cuda_Grid cgrid);

/* Upload grid data to device */
void
_cuda_GridUpload(cuda_Grid cgrid);

/* Download grid data from device */
void
_cuda_GridDownload(cuda_Grid cgrid);

/*--------------------------------------------------------------------------
 * Vector management
 *--------------------------------------------------------------------------*/

/* Initialize CUDA vector from user vector */
cuda_Vector
_cuda_VectorInit(braid_Core core, braid_App app, braid_Vector u);

/* Free CUDA vector */
void
_cuda_VectorFree(cuda_Vector cvec);

/* Upload vector to device */
void
_cuda_VectorUpload(cuda_Vector cvec);

/* Download vector from device */
void
_cuda_VectorDownload(cuda_Vector cvec);

/*--------------------------------------------------------------------------
 * Kernels
 *--------------------------------------------------------------------------*/

/* FCRelax on GPU */
braid_Int
_cuda_FCRelax(cuda_Core ccore, braid_Int level);

/* Compute global norm on GPU */
braid_Int
_cuda_NormGlobal(cuda_Core ccore, braid_Real* local_val, braid_Real* global_val,
                 braid_Int tnorm, MPI_Comm comm);

/* Vector sum (AXPY) on GPU */
braid_Int
_cuda_VectorSum(cuda_Vector cx, braid_Real alpha, cuda_Vector cy,
                braid_Real beta, cuda_Vector cz);

/* Spatial norm on GPU */
braid_Int
_cuda_SpatialNorm(cuda_Vector cx, braid_Real* norm);

/*--------------------------------------------------------------------------
 * Utility functions
 *--------------------------------------------------------------------------*/

/* Get CUDA stream for core */
cuda_Stream
_cuda_CoreStream(cuda_Core ccore);

/* Get device ID for core */
braid_Int
_cuda_CoreDevice(cuda_Core ccore);

/* Synchronize core's CUDA stream */
void
_cuda_CoreSync(cuda_Core ccore);

/* Error checking */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/* Stream synchronization */
#define CUDA_SYNC() CUDA_CHECK(cudaStreamSynchronize(0))

/*--------------------------------------------------------------------------
 * Public API - callable from user code
 *--------------------------------------------------------------------------*/

/* Initialize CUDA-aware XBraid */
braid_Int
braid_CUDAInit(braid_Core core, braid_Int device_id, cuda_Core *ccore_ptr);

/* Destroy CUDA core */
braid_Int
braid_CUDADestroy(cuda_Core ccore);

/* Perform FC-relaxation on GPU */
braid_Int
braid_CUDAFCRelax(cuda_Core ccore, braid_Int level);

/* Compute global norm on GPU */
braid_Int
braid_CUDANormGlobal(cuda_Core ccore, braid_Real* local_val, braid_Real* global_val,
                     braid_Int tnorm, MPI_Comm comm);

/* Vector operations on GPU */
braid_Int
braid_CUDAVecCopy(cuda_Core ccore, cuda_Vector cx, cuda_Vector cy);

braid_Int
braid_CUDAVecAXPY(cuda_Core ccore, braid_Real alpha, cuda_Vector cx,
                  braid_Real beta, cuda_Vector cy);

braid_Int
braid_CUDAVecNorm(cuda_Core ccore, cuda_Vector cx, braid_Real* norm);

/* Synchronize with GPU */
braid_Int
braid_CUDASync(cuda_Core ccore);

/* Upload data to GPU */
braid_Int
braid_CUDAGridUpload(cuda_Core ccore, braid_Int level);

/* Download data from GPU */
braid_Int
braid_CUDAGridDownload(cuda_Core ccore, braid_Int level);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_BRAID_H */
