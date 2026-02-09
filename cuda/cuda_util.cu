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
 * \file cuda_util.cu
 * \brief CUDA utility functions for XBraid.
 *
 * This file implements utility functions for CUDA memory management,
 * error checking, and core/grid initialization.
 */

#include "cuda_util.h"
#include "cuda_braid.h"
#include "cuda_vector.h"
#include "_braid.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*--------------------------------------------------------------------------
 * Memory management
 *--------------------------------------------------------------------------*/

braid_Int
_cuda_MemAlloc(void** d_ptr, braid_Int nbytes)
{
    if (nbytes <= 0) {
        *d_ptr = NULL;
        return _braid_error_flag;
    }

    cudaError_t err = cudaMalloc(d_ptr, nbytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return _braid_error_flag;
}

braid_Int
_cuda_MemFree(void* d_ptr)
{
    if (d_ptr == NULL) {
        return _braid_error_flag;
    }

    cudaError_t err = cudaFree(d_ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA free failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return _braid_error_flag;
}

braid_Int
_cuda_MemcpyH2D(void* d_ptr, const void* h_ptr, braid_Int nbytes)
{
    if (nbytes <= 0) {
        return _braid_error_flag;
    }

    cudaError_t err = cudaMemcpy(d_ptr, h_ptr, nbytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA H2D copy failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return _braid_error_flag;
}

braid_Int
_cuda_MemcpyD2H(void* h_ptr, const void* d_ptr, braid_Int nbytes)
{
    if (nbytes <= 0) {
        return _braid_error_flag;
    }

    cudaError_t err = cudaMemcpy(h_ptr, d_ptr, nbytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA D2H copy failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return _braid_error_flag;
}

braid_Int
_cuda_Memset(void* d_ptr, int value, braid_Int nbytes)
{
    if (nbytes <= 0) {
        return _braid_error_flag;
    }

    cudaError_t err = cudaMemset(d_ptr, value, nbytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memset failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return _braid_error_flag;
}

/*--------------------------------------------------------------------------
 * Core management
 *--------------------------------------------------------------------------*/

cuda_Core
_cuda_CoreInit(braid_Core core, braid_Int device_id)
{
    cuda_Core ccore;

    /* Allocate CUDA core structure */
    ccore = (_cuda_Core*)malloc(sizeof(_cuda_Core));
    if (ccore == NULL) {
        fprintf(stderr, "Failed to allocate CUDA core\n");
        return NULL;
    }

    /* Initialize CUDA device */
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA device %d: %s\n",
                device_id, cudaGetErrorString(err));
        free(ccore);
        return NULL;
    }

    /* Create CUDA stream */
    err = cudaStreamCreate(&ccore->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to create CUDA stream: %s\n", cudaGetErrorString(err));
        free(ccore);
        return NULL;
    }

    /* Copy core configuration */
    ccore->comm_world = _braid_CoreElt(core, comm_world);
    ccore->comm = _braid_CoreElt(core, comm);
    ccore->myid_world = _braid_CoreElt(core, myid_world);
    ccore->myid = _braid_CoreElt(core, myid);

    ccore->tstart = _braid_CoreElt(core, tstart);
    ccore->tstop = _braid_CoreElt(core, tstop);
    ccore->ntime = _braid_CoreElt(core, ntime);

    /* Copy function pointers */
    ccore->step = _braid_CoreElt(core, step);
    ccore->init = _braid_CoreElt(core, init);
    ccore->clone = _braid_CoreElt(core, clone);
    ccore->free = _braid_CoreElt(core, free);
    ccore->sum = _braid_CoreElt(core, sum);
    ccore->spatialnorm = _braid_CoreElt(core, spatialnorm);
    ccore->access = _braid_CoreElt(core, access);
    ccore->bufsize = _braid_CoreElt(core, bufsize);
    ccore->bufpack = _braid_CoreElt(core, bufpack);
    ccore->bufunpack = _braid_CoreElt(core, bufunpack);
    ccore->residual = _braid_CoreElt(core, residual);
    ccore->init_basis = _braid_CoreElt(core, init_basis);
    ccore->inner_prod = _braid_CoreElt(core, inner_prod);

    /* Copy configuration parameters */
    ccore->max_levels = _braid_CoreElt(core, max_levels);
    ccore->nlevels = _braid_CoreElt(core, nlevels);
    ccore->skip = _braid_CoreElt(core, skip);
    ccore->min_coarse = _braid_CoreElt(core, min_coarse);
    ccore->relax_only_cg = _braid_CoreElt(core, relax_only_cg);
    ccore->max_iter = _braid_CoreElt(core, max_iter);
    ccore->niter = _braid_CoreElt(core, niter);
    ccore->tnorm = _braid_CoreElt(core, tnorm);
    ccore->tol = _braid_CoreElt(core, tol);
    ccore->rtol = _braid_CoreElt(core, rtol);
    ccore->access_level = _braid_CoreElt(core, access_level);
    ccore->print_level = _braid_CoreElt(core, print_level);
    ccore->seq_soln = _braid_CoreElt(core, seq_soln);
    ccore->storage = _braid_CoreElt(core, storage);
    ccore->periodic = _braid_CoreElt(core, periodic);
    ccore->initiali = _braid_CoreElt(core, initiali);
    ccore->gupper = _braid_CoreElt(core, gupper);
    ccore->nrefine = _braid_CoreElt(core, nrefine);

    /* Copy relaxation parameters */
    ccore->nrels = _braid_TAlloc(braid_Int, ccore->max_levels);
    memcpy(ccore->nrels, _braid_CoreElt(core, nrels),
           ccore->max_levels * sizeof(braid_Int));

    ccore->CWts = _braid_TAlloc(braid_Real, ccore->max_levels);
    memcpy(ccore->CWts, _braid_CoreElt(core, CWts),
           ccore->max_levels * sizeof(braid_Real));

    ccore->cfactors = _braid_TAlloc(braid_Int, ccore->max_levels);
    memcpy(ccore->cfactors, _braid_CoreElt(core, cfactors),
           ccore->max_levels * sizeof(braid_Int));

    ccore->nrdefault = _braid_CoreElt(core, nrdefault);
    ccore->CWt_default = _braid_CoreElt(core, CWt_default);
    ccore->cfdefault = _braid_CoreElt(core, cfdefault);

    /* Copy Richardson parameters */
    ccore->richardson = _braid_CoreElt(core, richardson);
    ccore->est_error = _braid_CoreElt(core, est_error);
    ccore->order = _braid_CoreElt(core, order);
    ccore->dtk = _braid_CoreElt(core, dtk);
    ccore->estimate = _braid_CoreElt(core, estimate);

    /* Copy Delta correction parameters */
    ccore->delta_correct = _braid_CoreElt(core, delta_correct);
    ccore->delta_rank = _braid_CoreElt(core, delta_rank);
    ccore->delta_defer_lvl = _braid_CoreElt(core, delta_defer_lvl);
    ccore->delta_defer_iter = _braid_CoreElt(core, delta_defer_iter);

    /* Copy adjoint/optimization parameters */
    ccore->optim = _braid_CoreElt(core, optim);
    ccore->adjoint = _braid_CoreElt(core, adjoint);
    ccore->record = _braid_CoreElt(core, record);
    ccore->obj_only = _braid_CoreElt(core, obj_only);

    /* Copy timing parameters */
    ccore->timings = _braid_CoreElt(core, timings);
    ccore->timer_user_step = 0.0;
    ccore->timer_user_init = 0.0;
    ccore->timer_user_clone = 0.0;
    ccore->timer_user_free = 0.0;
    ccore->timer_user_sum = 0.0;
    ccore->timer_user_spatialnorm = 0.0;

    /* Copy print file */
    ccore->printfile = _braid_printfile;

    /* Store device ID */
    ccore->device_id = device_id;

    /* Initialize CUDA grids to NULL */
    ccore->cuda_grids = (_cuda_Grid**)calloc(ccore->nlevels, sizeof(_cuda_Grid*));

    /* Copy residual norm history */
    ccore->rnorms = _braid_CoreElt(core, rnorms);
    ccore->rnorm0 = _braid_CoreElt(core, rnorm0);
    ccore->full_rnorms = _braid_CoreElt(core, full_rnorms);
    ccore->full_rnorm0 = _braid_CoreElt(core, full_rnorm0);

    return ccore;
}

void
_cuda_CoreDestroy(cuda_Core ccore)
{
    if (ccore == NULL) {
        return;
    }

    /* Destroy CUDA grids */
    if (ccore->cuda_grids != NULL) {
        for (braid_Int level = 0; level < ccore->nlevels; level++) {
            if (ccore->cuda_grids[level] != NULL) {
                _cuda_GridDestroy(ccore->cuda_grids[level]);
            }
        }
        free(ccore->cuda_grids);
    }

    /* Destroy CUDA stream */
    if (ccore->stream != 0) {
        cudaStreamDestroy(ccore->stream);
    }

    /* Free allocated arrays */
    if (ccore->nrels != NULL) {
        free(ccore->nrels);
    }
    if (ccore->CWts != NULL) {
        free(ccore->CWts);
    }
    if (ccore->cfactors != NULL) {
        free(ccore->cfactors);
    }

    /* Free core structure */
    free(ccore);
}

/*--------------------------------------------------------------------------
 * Grid management
 *--------------------------------------------------------------------------*/

cuda_Grid
_cuda_GridInit(braid_Core core, braid_Int level)
{
    _braid_Grid **grids = _braid_CoreElt(core, grids);
    _braid_Grid *grid = grids[level];

    cuda_Grid cgrid = (_cuda_Grid*)malloc(sizeof(_cuda_Grid));
    if (cgrid == NULL) {
        fprintf(stderr, "Failed to allocate CUDA grid\n");
        return NULL;
    }

    cgrid->level = grid->level;
    cgrid->ilower = grid->ilower;
    cgrid->iupper = grid->iupper;
    cgrid->clower = grid->clower;
    cgrid->cupper = grid->cupper;
    cgrid->gupper = grid->gupper;
    cgrid->cfactor = grid->cfactor;
    cgrid->ncpoints = grid->ncpoints;

    /* Initialize device pointers to NULL */
    cgrid->ua_d = NULL;
    cgrid->va_d = NULL;
    cgrid->fa_d = NULL;
    cgrid->ta_d = NULL;

    /* Initialize host pointers */
    cgrid->ua_h = NULL;
    cgrid->va_h = NULL;
    cgrid->fa_h = NULL;
    cgrid->ta_h = NULL;

    cgrid->recv_index = grid->recv_index;
    cgrid->send_index = grid->send_index;

    /* Allocate device memory for time values */
    braid_Int nta = grid->iupper - grid->ilower + 3;  /* +2 for boundary values */
    cgrid->ta_alloc_size = nta;

    cudaError_t err = cudaMalloc((void**)&cgrid->ta_d, nta * sizeof(braid_Real));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc for time values failed: %s\n",
                cudaGetErrorString(err));
        free(cgrid);
        return NULL;
    }

    /* Initialize allocation sizes */
    cgrid->ua_alloc_size = 0;

    return cgrid;
}

void
_cuda_GridDestroy(cuda_Grid cgrid)
{
    if (cgrid == NULL) {
        return;
    }

    /* Free device memory */
    if (cgrid->ua_d != NULL) {
        cudaFree(cgrid->ua_d);
    }
    if (cgrid->va_d != NULL) {
        cudaFree(cgrid->va_d);
    }
    if (cgrid->fa_d != NULL) {
        cudaFree(cgrid->fa_d);
    }
    if (cgrid->ta_d != NULL) {
        cudaFree(cgrid->ta_d);
    }

    /* Free host memory */
    if (cgrid->ua_h != NULL) {
        /* Free individual vectors */
        braid_Int n = cgrid->iupper - cgrid->ilower + 1;
        for (braid_Int i = 0; i <= n; i++) {
            if (cgrid->ua_h[i] != NULL) {
                _cuda_VectorFree(cgrid->ua_h[i]);
            }
        }
        free(cgrid->ua_h);
    }
    if (cgrid->va_h != NULL) {
        free(cgrid->va_h);
    }
    if (cgrid->fa_h != NULL) {
        free(cgrid->fa_h);
    }
    if (cgrid->ta_h != NULL) {
        free(cgrid->ta_h);
    }

    /* Free grid structure */
    free(cgrid);
}

void
_cuda_GridUpload(cuda_Grid cgrid)
{
    if (cgrid->ta_h != NULL && cgrid->ta_d != NULL) {
        braid_Int nta = cgrid->ta_alloc_size;
        _cuda_MemcpyH2D(cgrid->ta_d, cgrid->ta_h, nta * sizeof(braid_Real));
    }
}

void
_cuda_GridDownload(cuda_Grid cgrid)
{
    if (cgrid->ta_d != NULL && cgrid->ta_h != NULL) {
        braid_Int nta = cgrid->ta_alloc_size;
        _cuda_MemcpyD2H(cgrid->ta_h, cgrid->ta_d, nta * sizeof(braid_Real));
    }
}

/*--------------------------------------------------------------------------
 * Vector management
 *--------------------------------------------------------------------------*/

cuda_Vector
_cuda_VectorInit(braid_Core core, braid_App app, braid_Vector u)
{
    cuda_Vector cvec = (_cuda_Vector*)malloc(sizeof(_cuda_Vector));
    if (cvec == NULL) {
        fprintf(stderr, "Failed to allocate CUDA vector\n");
        return NULL;
    }

    /* Get buffer size from user */
    braid_BufferStatus bstatus = (braid_BufferStatus)core;
    braid_Int size;
    _braid_BufferStatusInit(0, 0, 0, 0, bstatus);
    _braid_BaseBufSize(core, app, &size, bstatus);

    cvec->size = size;
    cvec->owned = 1;

    /* Allocate device memory */
    cudaError_t err = cudaMalloc(&cvec->d_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc for vector failed: %s\n", cudaGetErrorString(err));
        free(cvec);
        return NULL;
    }

    /* Allocate host buffer for staging */
    cvec->h_ptr = malloc(size);
    if (cvec->h_ptr == NULL) {
        cudaFree(cvec->d_ptr);
        free(cvec);
        return NULL;
    }

    return cvec;
}

void
_cuda_VectorFree(cuda_Vector cvec)
{
    if (cvec == NULL) {
        return;
    }

    if (cvec->d_ptr != NULL && cvec->owned) {
        cudaFree(cvec->d_ptr);
    }
    if (cvec->h_ptr != NULL) {
        free(cvec->h_ptr);
    }
    free(cvec);
}

void
_cuda_VectorUpload(cuda_Vector cvec)
{
    if (cvec->h_ptr != NULL && cvec->d_ptr != NULL) {
        _cuda_MemcpyH2D(cvec->d_ptr, cvec->h_ptr, cvec->size);
    }
}

void
_cuda_VectorDownload(cuda_Vector cvec)
{
    if (cvec->d_ptr != NULL && cvec->h_ptr != NULL) {
        _cuda_MemcpyD2H(cvec->h_ptr, cvec->d_ptr, cvec->size);
    }
}

/*--------------------------------------------------------------------------
 * Core utilities
 *--------------------------------------------------------------------------*/

cuda_Stream
_cuda_CoreStream(cuda_Core ccore)
{
    return ccore->stream;
}

braid_Int
_cuda_CoreDevice(cuda_Core ccore)
{
    return ccore->device_id;
}

void
_cuda_CoreSync(cuda_Core ccore)
{
    if (ccore->stream != 0) {
        cudaStreamSynchronize(ccore->stream);
    }
}
