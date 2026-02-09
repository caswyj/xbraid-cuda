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
 * \file cuda_util.h
 * \brief Header for CUDA utility functions.
 */

#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include "cuda_braid.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Memory management
 *--------------------------------------------------------------------------*/

/* Allocate device memory */
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

/* Set device memory */
braid_Int
_cuda_Memset(void* d_ptr, int value, braid_Int nbytes);

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
 * Core utilities
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

#ifdef __cplusplus
}
#endif

#endif /* CUDA_UTIL_H */
