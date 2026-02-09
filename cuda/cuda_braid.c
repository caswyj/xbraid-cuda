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
 * \file cuda_braid.c
 * \brief Main CUDA interface implementation for XBraid.
 *
 * This file provides the main interface functions that allow XBraid to use
 * CUDA for parallel-in-time computation on GPUs.
 */

#include "cuda_braid.h"
#include "cuda_util.h"
#include "cuda_vector.h"
#include "cuda_relax.h"
#include "_braid.h"

/* Stream type is defined in cuda_braid.h via typedef cudaStream_t cuda_Stream */

/*--------------------------------------------------------------------------
 * Public API functions
 *--------------------------------------------------------------------------*/

/* Initialize CUDA-aware XBraid from regular core */
braid_Int
braid_CUDAInit(braid_Core core, braid_Int device_id, cuda_Core *ccore_ptr)
{
    cuda_Core ccore;

    /* Initialize CUDA */
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: unable to set device %d: %s\n",
                device_id, cudaGetErrorString(err));
        return 1;
    }

    /* Create CUDA core */
    ccore = _cuda_CoreInit(core, device_id);
    if (ccore == NULL) {
        return 1;
    }

    /* Initialize CUDA grids */
    for (braid_Int level = 0; level < ccore->nlevels; level++) {
        ccore->cuda_grids[level] = _cuda_GridInit(core, level);
        if (ccore->cuda_grids[level] == NULL) {
            _cuda_CoreDestroy(ccore);
            return 1;
        }
    }

    *ccore_ptr = ccore;
    return 0;
}

/* Destroy CUDA core and cleanup */
braid_Int
braid_CUDADestroy(cuda_Core ccore)
{
    if (ccore != NULL) {
        _cuda_CoreDestroy(ccore);
    }
    return 0;
}

/* Perform FC-relaxation on GPU */
braid_Int
braid_CUDAFCRelax(cuda_Core ccore, braid_Int level)
{
    return _cuda_FCRelax(ccore, level);
}

/* Compute global norm on GPU */
braid_Int
braid_CUDANormGlobal(cuda_Core ccore, braid_Real* local_val, braid_Real* global_val,
                     braid_Int tnorm, MPI_Comm comm)
{
    /* This is a placeholder - full implementation would use CUDA reductions */
    /* For now, fall back to MPI reduction */
    if (tnorm == 1) {  /* one-norm */
        MPI_Allreduce(local_val, global_val, 1, braid_MPI_REAL, MPI_SUM, comm);
    } else if (tnorm == 3) {  /* inf-norm */
        MPI_Allreduce(local_val, global_val, 1, braid_MPI_REAL, MPI_MAX, comm);
    } else {  /* default two-norm */
        MPI_Allreduce(local_val, global_val, 1, braid_MPI_REAL, MPI_SUM, comm);
        *global_val = sqrt(*global_val);
    }
    return 0;
}

/* Vector operations on GPU */
braid_Int
braid_CUDAVecCopy(cuda_Core ccore, cuda_Vector cx, cuda_Vector cy)
{
    return _cuda_VectorCopy(cx, cy);
}

braid_Int
braid_CUDAVecAXPY(cuda_Core ccore, braid_Real alpha, cuda_Vector cx,
                  braid_Real beta, cuda_Vector cy)
{
    return _cuda_VectorAXPY(cx, alpha, cy, beta);
}

braid_Int
braid_CUDAVecNorm(cuda_Core ccore, cuda_Vector cx, braid_Real* norm)
{
    return _cuda_VectorNormL2(cx, norm);
}

/* Synchronize with GPU */
braid_Int
braid_CUDASync(cuda_Core ccore)
{
    _cuda_CoreSync(ccore);
    return 0;
}

/* Upload data to GPU */
braid_Int
braid_CUDAGridUpload(cuda_Core ccore, braid_Int level)
{
    if (ccore->cuda_grids[level] != NULL) {
        _cuda_GridUpload(ccore->cuda_grids[level]);
    }
    return 0;
}

/* Download data from GPU */
braid_Int
braid_CUDAGridDownload(cuda_Core ccore, braid_Int level)
{
    if (ccore->cuda_grids[level] != NULL) {
        _cuda_GridDownload(ccore->cuda_grids[level]);
    }
    return 0;
}
