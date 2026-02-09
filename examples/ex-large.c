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
 * Example:       ex-large.c
 *
 * Interface:     C
 *
 * Requires:      only C-language support
 *
 * Compile with:  make ex-large
 *
 * Sample run:    mpirun -np 10 ex-large
 *
 * Description:   Solves the 1D heat equation (large-scale version)
 *                Space-time domain:  [0, PI] x [0, T]
 *                Exact solution:     u(t,x) = sin(x)*cos(t)
 *
 *                This is a large-scale version designed to take ~10 minutes
 *                with 10 MPI processes for performance comparison.
 *
 *                Configuration:
 *                - nspace = 1024 (spatial grid points)
 *                - ntime = 8192 (time steps)
 *                - Expected time: ~8-12 minutes with 10 MPI processes
 **/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "braid.h"
#include "braid_test.h"
#include "ex-02-lib.c"

/*--------------------------------------------------------------------------
 * My App and Vector structures
 *--------------------------------------------------------------------------*/

typedef struct _braid_App_struct
{
   MPI_Comm  comm;
   double    tstart;
   double    tstop;
   int       ntime;
   double    xstart;
   double    xstop;
   int       nspace;
   double    matrix[3];
   double *  g;
   int       print_level;
} my_App;

typedef struct _braid_Vector_struct
{
   int     size;
   double *values;
} my_Vector;

void
create_vector(my_Vector **u, int size)
{
   (*u) = (my_Vector *) malloc(sizeof(my_Vector));
   ((*u)->size)   = size;
   ((*u)->values) = (double *) malloc(size*sizeof(double));
}

/*--------------------------------------------------------------------------
 * User-defined routines
 *--------------------------------------------------------------------------*/

int
my_Step(braid_App        app,
        braid_Vector     ustop,
        braid_Vector     fstop,
        braid_Vector     u,
        braid_StepStatus status)
{
   double tstart, tstop;
   double deltaT;

   braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
   deltaT = tstop - tstart;

   /* XBraid forcing */
   if(fstop != NULL) {
      for(int i = 0; i < u->size; i++) {
         u->values[i] = u->values[i] + fstop->values[i];
      }
   }

   /* Take backward Euler step */
   take_step(u->values, u->size, tstop, app->xstart,
             (app->xstop - app->xstart) / (app->nspace - 1.0),
             deltaT, app->matrix, app->g);

   return 0;
}

int
my_Init(braid_App     app,
        double        t,
        braid_Vector *u_ptr)
{
   my_Vector *u;
   int nspace = app->nspace;
   int size = nspace;
   double deltaX = (app->xstop - app->xstart) / (nspace - 1.0);

   create_vector(&u, size);

   if(t == app->tstart) {
      get_solution(u->values, nspace, 0.0, app->xstart, deltaX);
   }
   else {
      for(int i = 0; i < size; i++) {
         u->values[i] = sin(app->xstart + i * deltaX);
      }
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
   int size = u->size;

   create_vector(&v, size);
   for(int i = 0; i < size; i++) {
      v->values[i] = u->values[i];
   }
   *v_ptr = v;
   return 0;
}

int
my_Free(braid_App    app,
        braid_Vector u)
{
   free(u->values);
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
   int size = y->size;
   for(int i = 0; i < size; i++) {
      y->values[i] = alpha * x->values[i] + beta * y->values[i];
   }
   return 0;
}

int
my_SpatialNorm(braid_App     app,
               braid_Vector  u,
               double       *norm_ptr)
{
   int size = u->size;
   double dot = 0.0;

   for(int i = 0; i < size; i++) {
      dot += u->values[i] * u->values[i];
   }
   *norm_ptr = sqrt(dot);

   return 0;
}

int
my_Access(braid_App          app,
          braid_Vector       u,
          braid_AccessStatus astatus)
{
   int        index, rank, level, done, ntime;
   char       filename[255];
   double     t, error;
   double     deltaX = (app->xstop - app->xstart) / (app->nspace - 1.0);

   braid_AccessStatusGetT(astatus, &t);
   braid_AccessStatusGetTIndex(astatus, &index);
   braid_AccessStatusGetLevel(astatus, &level);
   braid_AccessStatusGetDone(astatus, &done);
   braid_AccessStatusGetNTPoints(astatus, &ntime);

   /* Print error at final time */
   if( (level == 0) && (app->print_level > 0) && (index == ntime) ) {
      error = compute_error_norm(u->values, app->xstart, app->xstop,
                                 app->nspace, t);
      printf("  Discretization error at final time:  %1.4e\n", error);
      fflush(stdout);
   }

   /* Save solution to file */
   if(done) {
      MPI_Comm_rank( (app->comm), &rank);
      sprintf(filename, "%s.%07d.%05d", "ex-large.out", index, rank);
      FILE *file = fopen(filename, "w");
      if(file) {
         fprintf(file, "%d\n", ntime + 1);
         fprintf(file, "%.14e\n", app->tstart);
         fprintf(file, "%.14e\n", app->tstop);
         fprintf(file, "%.14e\n", t);
         fprintf(file, "%d\n", app->nspace);
         fprintf(file, "%.14e\n", app->xstart);
         fprintf(file, "%.14e\n", app->xstop);
         for(int i = 0; i < app->nspace; i++) {
            fprintf(file, "%.14e\n", u->values[i]);
         }
         fclose(file);
      }
   }

   return 0;
}

int
my_BufSize(braid_App          app,
           int                *size_ptr,
           braid_BufferStatus bstatus)
{
   int size = app->nspace;
   *size_ptr = size * sizeof(double);
   return 0;
}

int
my_BufPack(braid_App          app,
           braid_Vector       u,
           void              *buffer,
           braid_BufferStatus bstatus)
{
   double *dbuffer = (double*)buffer;
   int size = u->size;
   memcpy(dbuffer, u->values, size * sizeof(double));
   braid_BufferStatusSetSize(bstatus, size * sizeof(double));
   return 0;
}

int
my_BufUnpack(braid_App          app,
             void              *buffer,
             braid_Vector      *u_ptr,
             braid_BufferStatus bstatus)
{
   my_Vector *u;
   int size = app->nspace;

   create_vector(&u, size);
   memcpy(u->values, buffer, size * sizeof(double));
   *u_ptr = u;
   return 0;
}

/*--------------------------------------------------------------------------
 * Main driver
 *--------------------------------------------------------------------------*/

int main(int argc, char *argv[])
{
   braid_Core    core;
   my_App       *app;
   MPI_Comm      comm;
   int           rank, i;

   /* Large problem parameters (~10 min with 10 procs) */
   int       ntime         = 65536;  /* 2^16 time steps */
   int       nspace        = 2048;   /* 2^11 spatial points */
   int       max_levels    = 16;
   int       nrelax        = 1;
   int       skip          = 1;
   double    tol           = 1.0e-09;
   int       cfactor       = 2;
   int       max_iter      = 50;
   int       min_coarse    = 3;
   int       print_level   = 2;
   int       access_level  = 1;
   int       use_seq_soln  = 0;

   /* Parse command line */
   for(i = 1; i < argc; i++) {
      if(strcmp(argv[i], "-help") == 0) {
         if(rank == 0) {
            printf("\n  ex-large: Large 1D heat equation (CPU version)\n\n");
            printf("  Usage: mpirun -np N ex-large [options]\n\n");
            printf("  Options:\n");
            printf("    -nt <n>             : number of time steps (default: 8192)\n");
            printf("    -ns <n>             : number of spatial points (default: 1024)\n");
            printf("    -ml <n>             : max number of levels (default: 16)\n");
            printf("    -nu <n>             : relaxation sweeps (default: 1)\n");
            printf("    -cf <n>             : coarsening factor (default: 2)\n");
            printf("    -mi <n>             : max iterations (default: 50)\n");
            printf("    -tol <n>            : tolerance (default: 1e-09)\n");
            printf("    -print_level <n>    : output level (default: 2)\n");
         }
         return 0;
      }
      else if(strcmp(argv[i], "-nt") == 0) {
         ntime = atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-ns") == 0) {
         nspace = atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-ml") == 0) {
         max_levels = atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-nu") == 0) {
         nrelax = atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-cf") == 0) {
         cfactor = atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-mi") == 0) {
         max_iter = atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-tol") == 0) {
         tol = atof(argv[++i]);
      }
      else if(strcmp(argv[i], "-print_level") == 0) {
         print_level = atoi(argv[++i]);
      }
   }

   MPI_Init(&argc, &argv);
   comm = MPI_COMM_WORLD;
   MPI_Comm_rank(comm, &rank);

   /* Define domain */
   double tstart = 0.0;
   double xstart = 0.0;
   double xstop  = PI;
   double cfl    = 0.30;
   double dx     = (xstop - xstart) / (nspace - 1.0);
   double dt     = cfl * dx * dx;  /* Simple CFL for 1D */

   app = (my_App *) malloc(sizeof(my_App));
   app->comm       = comm;
   app->tstart     = tstart;
   app->tstop      = tstart + ntime * dt;
   app->ntime      = ntime;
   app->xstart     = xstart;
   app->xstop      = xstop;
   app->nspace     = nspace;
   app->print_level = print_level;
   app->g          = (double*) malloc(nspace * sizeof(double));

   /* Initialize XBraid */
   braid_Init(comm, comm, tstart, app->tstop, ntime, app,
              my_Step, my_Init, my_Clone, my_Free, my_Sum,
              my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack,
              &core);

   /* Set parameters */
   braid_SetPrintLevel(core, print_level);
   braid_SetAccessLevel(core, access_level);
   braid_SetMaxLevels(core, max_levels);
   braid_SetMinCoarse(core, min_coarse);
   braid_SetSkip(core, skip);
   braid_SetNRelax(core, -1, nrelax);
   braid_SetAbsTol(core, tol / sqrt(dx * dt));
   braid_SetCFactor(core, -1, cfactor);
   braid_SetMaxIter(core, max_iter);
   braid_SetSeqSoln(core, use_seq_soln);

   if(rank == 0) {
      printf("\n  ex-large: Large 1D heat equation (CPU)\n");
      printf("  ----------------------------------------\n\n");
      printf("  Problem configuration:\n");
      printf("    Spatial points:   %d\n", nspace);
      printf("    Time steps:       %d\n", ntime);
      printf("    Total DOFs:       %d\n", nspace * ntime);
      printf("    CFL ratio:        %1.2e\n", cfl);
      printf("\n");
   }

   braid_Drive(core);

   if(rank == 0) {
      printf("\n  CPU version completed.\n\n");
   }

   braid_Destroy(core);
   free(app->g);
   free(app);
   MPI_Finalize();

   return 0;
}