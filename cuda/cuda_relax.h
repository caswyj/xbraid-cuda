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
 * \file cuda_relax.h
 * \brief Header for CUDA relaxation operations.
 */

#ifndef CUDA_RELAX_H
#define CUDA_RELAX_H

#include "cuda_braid.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * F-relaxation: Apply step function to all F-points
 *--------------------------------------------------------------------------*/

/* Apply F-relaxation on a grid */
braid_Int
_cuda_FRelax(cuda_Core ccore, cuda_Grid cgrid, braid_Int level);

/*--------------------------------------------------------------------------
 * C-relaxation: Apply step function to C-points with possible weighting
 *--------------------------------------------------------------------------*/

/* Apply C-relaxation with weighted Jacobi */
braid_Int
_cuda_CRelax(cuda_Core ccore, cuda_Grid cgrid, braid_Int level, braid_Real omega);

/*--------------------------------------------------------------------------
 * Combined FC-relaxation
 *--------------------------------------------------------------------------*/

/* Apply nrelax sweeps of F-then-C relaxation */
braid_Int
_cuda_FCRelax(cuda_Core ccore, braid_Int level);

/*--------------------------------------------------------------------------
 * Richardson-enhanced relaxation
 *--------------------------------------------------------------------------*/

/* Apply F-relaxation with Richardson weighting */
braid_Int
_cuda_FRelax_Richardson(cuda_Core ccore, cuda_Grid cgrid, braid_Int level);

/* Apply C-relaxation with Richardson weighting */
braid_Int
_cuda_CRelax_Richardson(cuda_Core ccore, cuda_Grid cgrid, braid_Int level,
                        braid_Real a, braid_Real b);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_RELAX_H */
