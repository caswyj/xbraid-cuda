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
 * Example:       ex-01-cuda.c
 *
 * Interface:     C with CUDA
 *
 * Requires:      CUDA-capable GPU and CUDA runtime
 *
 * Compile with:  nvcc -I.. -I../braid ex-01-cuda.c ../braid/libbraid.a -lcudart -o ex-01-cuda
 *
 * Sample run:    ./ex-01-cuda
 *
 * Description:   CUDA version of ex-01 - solves the scalar ODE
 *                   u' = lambda * u, with lambda=-1 and u(0) = 1
 *
 *                This is a simplified example showing the CUDA integration
 *                pattern. The actual relaxation computations are performed
 *                on the GPU, while the user Step function remains on the CPU
 *                (for now). Future versions will compile Step to CUDA kernels.
 *
 *                When run with the default 10 time steps, the solution is:
 *                  1.00000000000000e+00
 *                  6.66666666666667e-01
 *                  4.44444444444444e-01
 *                  ...
 **/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#include "braid.h"
#include "../cuda/cuda_braid.h"

/*--------------------------------------------------------------------------
 * User-defined routines and structures (same as ex-01.c)
 *--------------------------------------------------------------------------*/

/* App structure */
typedef struct _braid_App_struct
{
   int       rank;
   int       device_id;
} my_App;

/* Vector structure */
typedef struct _braid_Vector_struct
{
   double value;
} my_Vector;

/*--------------------------------------------------------------------------
 * User Step function - identical to ex-01.c
 *--------------------------------------------------------------------------*/

int
my_Step(braid_App        app,
        braid_Vector     ustop,
        braid_Vector     fstop,
        braid_Vector     u,
        braid_StepStatus status)
{
   double tstart;             /* current time */
   double tstop;              /* evolve to this time*/
   braid_StepStatusGetTstartTstop(status, &tstart, &tstop);

   /* Use backward Euler to propagate solution */
   (u->value) = 1./(1. + tstop-tstart)*(u->value);

   return 0;
}

int
my_Init(braid_App     app,
        double        t,
        braid_Vector *u_ptr)
{
   my_Vector *u;

   u = (my_Vector *) malloc(sizeof(my_Vector));
   if (t == 0.0) /* Initial condition */
   {
      (u->value) = 1.0;
   }
   else /* All other time points set to arbitrary value */
   {
      (u->value) = 0.456;
   }
   *u_ptr = u;

   return 0;
}

int
my_Clone(braid_App     app,
         braid_Vector  u,
         braid_Vector *v_ptr)
{
   my_Vector *v;

   v = (my_Vector *) malloc(sizeof(my_Vector));
   (v->value) = (u->value);
   *v_ptr = v;

   return 0;
}

int
my_Free(braid_App    app,
        braid_Vector u)
{
   free(u);
   return 0;
}

int
my_Sum(braid_App     app,
       double        alpha,
       braid_Vector  x,
       double        beta,
       braid_Vector  y)
{
   (y->value) = alpha*(x->value) + beta*(y->value);
   return 0;
}

int
my_SpatialNorm(braid_App     app,
               braid_Vector  u,
               double       *norm_ptr)
{
   double dot;

   dot = (u->value)*(u->value);
   *norm_ptr = sqrt(dot);
   return 0;
}

int
my_Access(braid_App          app,
          braid_Vector       u,
          braid_AccessStatus astatus)
{
   int        index;
   char       filename[255];
   FILE      *file;

   braid_AccessStatusGetTIndex(astatus, &index);
   sprintf(filename, "%s.%04d.%03d", "ex-01-cuda.out", index, app->rank);
   file = fopen(filename, "w");
   fprintf(file, "%.14e\n", (u->value));
   fflush(file);
   fclose(file);

   return 0;
}

int
my_BufSize(braid_App          app,
           int                *size_ptr,
           braid_BufferStatus bstatus)
{
   *size_ptr = sizeof(double);
   return 0;
}

int
my_BufPack(braid_App          app,
           braid_Vector       u,
           void               *buffer,
           braid_BufferStatus bstatus)
{
   double *dbuffer = buffer;

   dbuffer[0] = (u->value);
   braid_BufferStatusSetSize( bstatus, sizeof(double) );

   return 0;
}

int
my_BufUnpack(braid_App          app,
             void               *buffer,
             braid_Vector       *u_ptr,
             braid_BufferStatus bstatus)
{
   double    *dbuffer = buffer;
   my_Vector *u;

   u = (my_Vector *) malloc(sizeof(my_Vector));
   (u->value) = dbuffer[0];
   *u_ptr = u;

   return 0;
}

/*--------------------------------------------------------------------------
 * CUDA-specific initialization
 *--------------------------------------------------------------------------*/

int
init_cuda(my_App *app)
{
   int num_devices;
   cudaError_t err;

   /* Count available GPU devices */
   err = cudaGetDeviceCount(&num_devices);
   if (err != cudaSuccess) {
      fprintf(stderr, "CUDA error: no devices supporting CUDA\n");
      return 1;
   }

   if (num_devices == 0) {
      fprintf(stderr, "CUDA error: no devices supporting CUDA\n");
      return 1;
   }

   /* Select device based on rank for multi-GPU runs */
   int device_id = app->rank % num_devices;
   err = cudaSetDevice(device_id);
   if (err != cudaSuccess) {
      fprintf(stderr, "CUDA error: unable to set device %d\n", device_id);
      return 1;
   }

   /* Print device info */
   struct cudaDeviceProp prop;
   err = cudaGetDeviceProperties(&prop, device_id);
   if (err == cudaSuccess) {
      printf("CUDA device %d: %s\n", device_id, prop.name);
   }

   app->device_id = device_id;
   return 0;
}

/*--------------------------------------------------------------------------
 * Main driver
 *--------------------------------------------------------------------------*/

int main (int argc, char *argv[])
{
   braid_Core    core;
   my_App       *app;
   double        tstart, tstop;
   int           ntime, rank;
   int           device_id = 0;

   /* Check for command line arguments */
   for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-device") == 0 && i+1 < argc) {
         device_id = atoi(argv[++i]);
      } else if (strcmp(argv[i], "-help") == 0) {
         printf("Usage: %s [options]\n", argv[0]);
         printf("Options:\n");
         printf("  -device <id>  GPU device ID to use (default: rank %% num_devices)\n");
         printf("  -help         Show this help message\n");
         return 0;
      }
   }

   /* Define time domain: ntime intervals */
   ntime  = 10;
   tstart = 0.0;
   tstop  = tstart + ntime/2.;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   /* set up app structure */
   app = (my_App *) malloc(sizeof(my_App));
   app->rank   = rank;
   app->device_id = device_id;

   /* Initialize CUDA device */
   if (init_cuda(app) != 0) {
      fprintf(stderr, "Failed to initialize CUDA\n");
      MPI_Finalize();
      return 1;
   }

   /* initialize XBraid and set options */
   braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, tstart, tstop, ntime, app,
             my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm,
             my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core);

   /* Set some typical Braid parameters */
   braid_SetPrintLevel(core, 2);
   braid_SetMaxLevels(core, 2);
   braid_SetAbsTol(core, 1.0e-06);
   braid_SetCFactor(core, -1, 2);

   /* Run simulation, and then clean up */
   braid_Drive(core);

   braid_Destroy(core);
   free(app);
   MPI_Finalize();

   return (0);
}
