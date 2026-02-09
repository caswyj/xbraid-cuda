/*BHEADER**********************************************************************
 * Copyright (c) 2013, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory. Written by
 * Jacob Schroder, Rob Falgout, Tzanio Kolev, Ulrike Yang, Veselin
 * Dobrev, et al. LLNL-CODE-660355. All rights reserved.
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
 * Example:       ex-08.c
 *
 * Interface:     C
 *
 * Requires:      hypre
 *
 * Compile with:  make ex-08
 *
 * Help with:     ex-08 -help
 *
 * Sample run:    mpirun -np 10 ex-08 -pgrid 2 1 5 -ml 16 -nt 2048 -nx 129 129 -mi 100
 *
 * Description:   Solves the 2D heat equation on a regular grid in space and time,
 *                using backward Euler in time and classic second order
 *                finite-differencing in space.
 *
 *                This is a LARGE-SCALE example designed to take ~10 minutes
 *                with 10 MPI processes for performance comparison with CUDA version.
 *
 *                Problem configuration for ~10 min CPU time (with 10 procs):
 *                - nx = 129, ny = 129 (16,384 spatial DOFs per time step)
 *                - nt = 2048 time steps
 *                - pgrid = 2 x 1 x 5 (2x1 space, 5 time)
 *
 *                Total spatial DOFs: ~16K per step
 *                Total time DOFs: 2048
 *                Expected time: ~8-12 minutes with 10 MPI processes
 **/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE_sstruct_ls.h"
#include "_hypre_sstruct_mv.h"

#include "braid.h"
#include "braid_test.h"
#include "ex-03-lib.c"

/* --------------------------------------------------------------------
 * XBraid app struct
 * -------------------------------------------------------------------- */
typedef struct _braid_App_struct {
   MPI_Comm                comm;             /* global communicator */
   MPI_Comm                comm_t;           /* communicator for parallelizing in time  */
   MPI_Comm                comm_x;           /* communicator for parallelizing in space  */
   int                     pt;               /* number of processors in time  */
   simulation_manager     *man;              /* user's simulation manager structure */
   HYPRE_SStructVector     e;                /* temporary vector used for error computations */
   int                     nA;               /* number of discr. matrices that have been created */
   int                     max_nA;           /* max nA value allowed */
   HYPRE_SStructMatrix    *A;                /* nA sized array of discr. matrices (one per time level) */
   double                 *dt_A;             /* nA sized array of time step sizes for each  matrix  */
   HYPRE_StructSolver     *solver;           /* nA sized array of solvers (one per time level) */
   int                     use_rand;         /* binary, use random initial guess (1) or zero initial guess (0) */
   int                    *runtime_max_iter; /* runtime info for the max number of spatial solve iterations at each level */
   int                    *max_iter_x;       /* length 2 array of expensive and cheap max PFMG iters for spatial solves*/
} my_App;

int print_app(my_App * app)
{
   int myid,i;
   MPI_Comm_rank( app->comm, &myid );
   printf("\n\nmyid:  %d,  App contents:\n", myid);
   printf("myid:  %d,  pt:            %d\n", myid, app->pt);
   printf("myid:  %d,  use_rand:      %d\n", myid, app->use_rand);
   printf("myid:  %d,  nA:            %d\n", myid, app->nA);
   printf("myid:  %d,  max_iter_x[0]: %d\n", myid, app->max_iter_x[0]);
   printf("myid:  %d,  max_iter_x[1]: %d\n", myid, app->max_iter_x[1]);
   for(i = 0; i < app->nA; i++){
      printf("myid:  %d,  runtime_max_iter[%d]: %d\n", myid, i, app->runtime_max_iter[i]);
   }
   for(i = 0; i < app->nA; i++){
      printf("myid:  %d,  dt_A[%d]:           %1.2e\n", myid, i, app->dt_A[i]);
   }
   printf("\nmyid:  %d,  Note that some object members like comm, comm_t, comm_x, man, A and solver cannot be printed\n\n", myid);
   return 0;
}

/* --------------------------------------------------------------------
 * XBraid vector
 * Stores the state of the simulation for a given time step
 * -------------------------------------------------------------------- */
typedef struct _braid_Vector_struct {
   HYPRE_SStructVector   x;
} my_Vector;

/* --------------------------------------------------------------------
 * Time integrator routine that performs the update
 *   u_i = Phi_i(u_{i-1}) + g_i
 *
 * When Phi is called, u is u_{i-1}.
 * The return value is that u is set to u_i upon completion
 * -------------------------------------------------------------------- */
int
my_Step(braid_App        app,
        braid_Vector     ustop,
        braid_Vector     fstop,
        braid_Vector     u,
        braid_StepStatus status)
{
   double tstart;             /* current time */
   double tstop;              /* evolve u to this time*/
   HYPRE_SStructVector  bstop;
   int level;
   int iters_taken = -1;

   /* Grab status of current time step */
   braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
   braid_StepStatusGetLevel(status, &level);

   /* Now, set up the discretization matrix.  Use the XBraid level to index
    * into the matrix lookup table */

   /* We need to "trick" the user's manager with the new dt */
   app->man->dt = tstop - tstart;

   /* Set up a new matrix */
   if( app->dt_A[level] == -1.0 ){
      app->nA++;
      app->dt_A[level] = tstop-tstart;

      setUpImplicitMatrix( app->man );
      app->A[level] = app->man->A;

      /* Set up the PFMG solver using u->x as dummy vectors. */
      setUpStructSolver( app->man, u->x, u->x );
      app->solver[level] = app->man->solver;
   }

   /* Time integration to next time point: Solve the system Ax = b.
    * First, "trick" the user's manager with the right matrix and solver */
   app->man->A = app->A[level];
   app->man->solver = app->solver[level];

   /* Use level specific max_iter */
   if( level == 0 )
      app->man->max_iter = app->max_iter_x[0];
   else
      app->man->max_iter = app->max_iter_x[1];

   /* Take step */
   if (fstop == NULL)
   {
      bstop = NULL;
   }
   else
   {
      bstop = fstop->x;
   }
   take_step(app->man, ustop->x, bstop, u->x, tstart, tstop, &iters_taken);

   /* Store iterations taken */
   app->runtime_max_iter[level] = max_i( (app->runtime_max_iter[level]), iters_taken);

   /* Tell XBraid no refinement */
   braid_StepStatusSetRFactor(status, 1);

   return 0;
}

/* --------------------------------------------------------------------
 * -------------------------------------------------------------------- */
int
my_Residual(braid_App        app,
            braid_Vector     ustop,
            braid_Vector     r,
            braid_StepStatus status)
{
   double tstart;             /* current time */
   double tstop;              /* evolve u to this time*/
   int level;

   /* Grab status of current time step */
   braid_StepStatusGetTstartTstop(status, &tstart, &tstop);

   /* Grab level */
   braid_StepStatusGetLevel(status, &level);

   /* We need to "trick" the user's manager with the new dt */
   app->man->dt = tstop - tstart;

   /* Now, set up the discretization matrix.  Use the XBraid level to index
    * into the matrix lookup table */
   if( app->dt_A[level] == -1.0 ){
      app->nA++;
      app->dt_A[level] = tstop-tstart;

      setUpImplicitMatrix( app->man );
      app->A[level] = app->man->A;

      /* Set up the PFMG solver using r->x as dummy vectors. */
      setUpStructSolver( app->man, r->x, r->x );
      app->solver[level] = app->man->solver;
   }

   /* Compute residual Ax */
   app->man->A = app->A[level];
   comp_res(app->man, ustop->x, r->x, tstart, tstop);

   return 0;
}

/* --------------------------------------------------------------------
 * Create a vector object for a given time point.
 * This function is only called on the finest level.
 * -------------------------------------------------------------------- */
int
my_Init(braid_App     app,
        double        t,
        braid_Vector *u_ptr)
{

   my_Vector * u = (my_Vector *) malloc( sizeof(my_Vector) );

   if( t == app->man->tstart ){
      /* Sets u_ptr as the initial condition */
      t = 0.0;
   }
   else if (app->use_rand){
      /* This t-value will tell set_initial_condition() below to make u_ptr uniformly random */
      t = -1.0;
   }
   else{
      /* Sets u_ptr as an all zero vector*/
      t = 1.0;
   }

   set_initial_condition(app->man, &(u->x), t);
   *u_ptr = u;
   return 0;
}

/* --------------------------------------------------------------------
 * Create a copy of a vector object.
 * -------------------------------------------------------------------- */
int
my_Clone(braid_App     app,
         braid_Vector  u,
         braid_Vector *v_ptr)
{
   my_Vector *v = (my_Vector *) malloc(sizeof(my_Vector));
   double    *values;
   initialize_vector(app->man, &(v->x));

   /* Set the values. */
   values = (double *) malloc( (app->man->nlx)*(app->man->nly)*sizeof(double) );
   HYPRE_SStructVectorGather( u->x );
   HYPRE_SStructVectorGetBoxValues( u->x, 0, app->man->ilower, app->man->iupper, 0, values );
   HYPRE_SStructVectorSetBoxValues( v->x, 0, app->man->ilower, app->man->iupper, 0, values );
   free( values );
   HYPRE_SStructVectorAssemble( v->x );

   *v_ptr = v;
   return 0;
}

/* --------------------------------------------------------------------
 * Destroy vector object.
 * -------------------------------------------------------------------- */
int
my_Free(braid_App    app,
        braid_Vector u)
{
   HYPRE_SStructVectorDestroy( u->x );
   free( u );

   return 0;
}

/* --------------------------------------------------------------------
 * Compute vector sum y = alpha*x + beta*y.
 * -------------------------------------------------------------------- */
int
my_Sum(braid_App    app,
       double       alpha,
       braid_Vector x,
       double       beta,
       braid_Vector y)
{
   int i;
   double *values_x, *values_y;

   values_x = (double *) malloc( (app->man->nlx)*(app->man->nly)*sizeof(double) );
   values_y = (double *) malloc( (app->man->nlx)*(app->man->nly)*sizeof(double) );

   HYPRE_SStructVectorGather( x->x );
   HYPRE_SStructVectorGetBoxValues( x->x, 0, (app->man->ilower), (app->man->iupper), 0, values_x );
   HYPRE_SStructVectorGather( y->x );
   HYPRE_SStructVectorGetBoxValues( y->x, 0, (app->man->ilower), (app->man->iupper), 0, values_y );

   for( i = 0; i < (app->man->nlx)*(app->man->nly); i++ ){
      values_y[i] = alpha*values_x[i] + beta*values_y[i];
   }

   HYPRE_SStructVectorSetBoxValues( y->x, 0, (app->man->ilower), (app->man->iupper), 0, values_y );

   free( values_x );
   free( values_y );
   return 0;
}

/* --------------------------------------------------------------------
 * User access routine to spatial solution vectors and allows for user
 * output.  The default XBraid parameter of access_level=1, calls
 * my_Access only after convergence and at every time point.
 * -------------------------------------------------------------------- */
int
my_Access(braid_App           app,
          braid_Vector        u,
          braid_AccessStatus  astatus)
{
   double     tstart         = (app->man->tstart);
   double     tstop          = (app->man->tstop);
   int        nt             = (app->man->nt);

   double     rnorm, disc_err, t;
   int        iter, level, done, index, myid;
   char       filename[255], filename_mesh[255], filename_err[255], filename_sol[255];

   /* Retrieve current time from Status Object */
   braid_AccessStatusGetT(astatus, &t);

   /* Retrieve XBraid State Information from Status Object */
   MPI_Comm_rank(app->comm_x, &myid);
   braid_AccessStatusGetTILD(astatus, &t, &iter, &level, &done);
   braid_AccessStatusGetResidual(astatus, &rnorm);

   if(level == 0)
   {
      /* Print discretization error to screen for only final time */
      index = ((t - tstart) / ((tstop - tstart)/nt) + 0.1);
      compute_disc_err(app->man, u->x, t, app->e, &disc_err);
      if( (t == app->man->tstop) && myid == 0 ) {
         printf("\n  Discr. error         = %1.5e\n", disc_err);
         printf("\n  my_Access():  Braid iter %d,  discr. error at final time:  %1.4e\n", iter, disc_err);

      }

      /* Write the norm of the discretization error to a separate file for each time step */
      if( app->man->output_files ){
         sprintf(filename, "%s.iter%03d.time%07d", "ex-08.error_norm", iter, index);
         output_error_file(app->man, t, disc_err, filename);
      }
   }

   /* Write THREE GLVIS visualization files for the final time step:
    * (1) the discretization error (2) the true solution (3) the discrete solution */
   if( app->man->output_vis && (level == 0) && (t == app->man->tstop) )
   {
      sprintf(filename_mesh, "%s.iter%03d", "ex-08_mesh", iter);
      sprintf(filename_err, "%s.iter%03d", "ex-08_err_tstop", iter);
      sprintf(filename_sol, "%s.iter%03d", "ex-08_sol_tstop", iter);
      output_vis(app->man, u->x, t, filename_mesh, filename_err, filename_sol);
   }

   return 0;
}

/* --------------------------------------------------------------------
 * Compute norm of a spatial vector
 * -------------------------------------------------------------------- */
int
my_SpatialNorm(braid_App     app,
               braid_Vector  u,
               double       *norm_ptr)
{
   norm(u->x, norm_ptr);
   return 0;
}

/* --------------------------------------------------------------------
 * Return buffer size needed to pack one spatial braid_Vector.  Here the
 * vector contains one double at every grid point and thus, the buffer
 * size is the number of grid points.
 * -------------------------------------------------------------------- */
int
my_BufSize(braid_App           app,
           int                 *size_ptr,
           braid_BufferStatus  status)
{
    *size_ptr = (app->man->nlx)*(app->man->nly)*sizeof(double);
    return 0;
}

/* --------------------------------------------------------------------
 * Pack a braid_Vector into a buffer.
 * -------------------------------------------------------------------- */
int
my_BufPack(braid_App           app,
           braid_Vector        u,
           void                *buffer,
           braid_BufferStatus  status)
{
   double *dbuffer = buffer;

   /* Place the values in u into the buffer */
   HYPRE_SStructVectorGather( u->x );
   HYPRE_SStructVectorGetBoxValues( u->x, 0, app->man->ilower,
                          app->man->iupper, 0, &(dbuffer[0]) );

   /* Return the number of bytes actually packed */
   braid_BufferStatusSetSize( status, (app->man->nlx)*(app->man->nly)*sizeof(double) );
   return 0;
}

/* --------------------------------------------------------------------
 * Unpack a buffer and place into a braid_Vector
 * -------------------------------------------------------------------- */
int
my_BufUnpack(braid_App           app,
             void                *buffer,
             braid_Vector        *u_ptr,
             braid_BufferStatus  status)
{
   double    *dbuffer = buffer;
   my_Vector *u       = (my_Vector *) malloc( sizeof(my_Vector) );

   /* Set the values in u based on the values in the buffer */
   initialize_vector(app->man, &(u->x));
   HYPRE_SStructVectorSetBoxValues( u->x, 0, app->man->ilower,
                           app->man->iupper, 0, &(dbuffer[0]) );
   HYPRE_SStructVectorAssemble( u->x );
   *u_ptr = u;

   return 0;
}

/* --------------------------------------------------------------------
 * Main driver
 * -------------------------------------------------------------------- */
int main (int argc, char *argv[])
{
   /* Declare variables -- variables explained when they are set below */
   int print_usage                       = 0;
   int *runtime_max_iter_global          = NULL;
   MPI_Comm comm                         = MPI_COMM_WORLD;
   MPI_Comm comm_x, comm_t;
   int i, arg_index, myid, num_procs;

   braid_Core    core;
   my_App       *app = (my_App *) malloc(sizeof(my_App));
   double tol, mystarttime, myendtime, mytime, maxtime, cfl;
   int run_wrapper_tests, correct1, correct2;
   int print_level, access_level, max_nA, nA_max, max_levels, skip, min_coarse;
   int nrelax, nrelax0, cfactor, cfactor0, max_iter, fmg, res, storage, tnorm;
   int fullrnorm, use_seq_soln;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank( comm, &myid );

   /* Default parameters for LARGE problem (~10 min with 10 procs) */
   app->man  = (simulation_manager *) malloc(sizeof(simulation_manager));
   app->man->px              = 2;             /* 2x1 = 2 procs in space */
   app->man->py              = 1;             /* my processor number in the y-direction */
   app->man->nx              = 129;           /* spatial grid size (129x129 = 16384 DOFs) */
   app->man->ny              = 129;           /* spatial grid size */
   app->man->nt              = 2048;          /* number of time steps (large for ~10 min) */
   app->man->forcing         = 1;             /* Use nonzero forcing term */
   app->man->tol             = 1.0e-09;       /* PFMG halting tolerance */
   app->man->output_files    = 0;             /* Boolean, if 1 output the norm of the discretization error to a file for each time step */
   app->man->output_vis      = 0;             /* Boolean, if 1 output GLVIS files */
   app->man->dim_x           = 2;             /* Two dimensional problem */
   app->man->K               = 1.0;           /* Diffusion coefficient */
   app->man->nlx             = 129;           /* number of point ~local~ to this processor in the x-dim */
   app->man->nly             = 129;           /* number of point ~local~ to this processor in the y-dim */
   app->man->tstart          = 0.0;           /* global start time */
   app->man->object_type     = HYPRE_STRUCT;  /* Hypre Struct interface is used for solver */
   app->man->vartype         = HYPRE_SSTRUCT_VARIABLE_CELL;
   app->man->explicit        = 0;             /* Permanently turns off the explicit time stepping capability */
   cfl                       = 0.30;          /* CFL = K*(dt/dx^2 + dt/dy^2) is used to define dt and t-final */


   /* Default XBraid parameters */
   max_levels          = 16;              /* Max levels for XBraid solver (deeper hierarchy for large nt) */
   skip                = 1;               /* Boolean, whether to skip all work on first down cycle */
   min_coarse          = 3;               /* Minimum possible coarse grid size */
   nrelax              = 1;               /* Number of CF relaxation sweeps on all levels */
   nrelax0             = -1;              /* Number of CF relaxations only for level 0 -- overrides nrelax */
   tol                 = 1.0e-09;         /* Halting tolerance */
   tnorm               = 2;               /* Halting norm to use (see docstring below) */
   cfactor             = 2;               /* Coarsening factor */
   cfactor0            = -1;              /* Coarsening factor for only level 0 -- overrides cfactor */
   max_iter            = 50;              /* Maximum number of iterations */
   fmg                 = 0;               /* Boolean, if 1, do FMG cycle.  If 0, use a V cycle */
   res                 = 0;               /* Boolean, if 1, use my residual */
   storage             = -1;              /* Full storage on levels >= 'storage' */
   print_level         = 2;               /* Level of XBraid printing to the screen */
   access_level        = 1;               /* Frequency of calls to access routine: 1 is for only after simulation */
   run_wrapper_tests   = 0;               /* Run no simulation, only run wrapper tests */
   fullrnorm           = 0;               /* Do not compute full residual from user routine each iteration */
   use_seq_soln        = 0;               /* Use the solution from sequential time stepping as the initial guess */

   /* Other parameters specific to parallel in time */
   app->use_rand       = 1;               /* If 1, use a random initial guess, else use a zero initial guess */
   app->pt             = 1;               /* Number of processors in time */
   app->max_iter_x     = (int*) malloc( 2*sizeof(int) );
   app->max_iter_x[0]  = 100;             /* Maximum number of PFMG iters (the spatial solver from hypre) on XBraid level 0 */
   app->max_iter_x[1]  = 100;             /* Maximum number of PFMG iter on all coarse XBraid levels */

   /* Parse command line */
   arg_index = 0;
   while( arg_index < argc ){
      if( strcmp(argv[arg_index], "-pgrid") == 0 ){
         arg_index++;
         app->man->px = atoi(argv[arg_index++]);
         app->man->py = atoi(argv[arg_index++]);
         app->pt = atoi(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-nx") == 0 ){
         arg_index++;
         app->man->nx = atoi(argv[arg_index++]);
         app->man->ny = atoi(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-nt") == 0 ){
          arg_index++;
          app->man->nt = atoi(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-cfl") == 0 ){
          arg_index++;
          cfl = atof(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-forcing") == 0 ){
         arg_index++;
         app->man->forcing = 1;
      }
      else if( strcmp(argv[arg_index], "-pfmg_tol") == 0 ){
          arg_index++;
          app->man->tol = atof(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-use_seq_soln") == 0 ){
          arg_index++;
          use_seq_soln = atoi(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-output_files") == 0 ){
         arg_index++;
         app->man->output_files = 1;
      }
      else if( strcmp(argv[arg_index], "-output_vis") == 0 ){
         arg_index++;
         app->man->output_vis = 1;
      }
      else if( strcmp(argv[arg_index], "-ml") == 0 ){
          arg_index++;
          max_levels = atoi(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-skip") == 0 ){
          arg_index++;
          skip = atoi(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-mc") == 0 ){
          arg_index++;
          min_coarse = atoi(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-nu") == 0 ){
          arg_index++;
          nrelax = atoi(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-nu0") == 0 ){
          arg_index++;
          nrelax0 = atoi(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-tol") == 0 ){
          arg_index++;
          tol = atof(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-tnorm") == 0 ){
          arg_index++;
          tnorm = atoi(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-cf") == 0 ){
          arg_index++;
          cfactor = atoi(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-cf0") == 0 ){
          arg_index++;
          cfactor0 = atoi(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-mi") == 0 ){
          arg_index++;
          max_iter = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-fmg") == 0 ){
         arg_index++;
         fmg = 1;
      }
      else if ( strcmp(argv[arg_index], "-res") == 0 ){
         arg_index++;
         res = 1;
      }
      else if ( strcmp(argv[arg_index], "-storage") == 0 ){
         arg_index++;
         storage = atoi(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-print_level") == 0 ){
         arg_index++;
         print_level = atoi(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-access_level") == 0 ){
         arg_index++;
         access_level = atoi(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-run_wrapper_tests") == 0 ){
         arg_index++;
         run_wrapper_tests = 1;
      }
      else if ( strcmp(argv[arg_index], "-use_rand") == 0 ){
         arg_index++;
         app->use_rand = atoi(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-pfmg_mi") == 0 ){
         arg_index++;
         app->max_iter_x[0] = atoi(argv[arg_index++]);
         app->max_iter_x[1] = atoi(argv[arg_index++]);
      }
      else if( strcmp(argv[arg_index], "-fullrnorm") == 0 ){
         arg_index++;
         fullrnorm = 1;
      }
      else if( strcmp(argv[arg_index], "-help") == 0 ){
         print_usage = 1;
         break;
      }
      else{
         if(arg_index > 1){
            printf("UNUSED command line paramter %s\n", argv[arg_index]);
         }
         arg_index++;
      }
   }

   if((print_usage) && (myid == 0)){
      printf("\n");
      printf("Usage: %s [<options>]\n", argv[0]);
      printf("\n");
      printf(" Large-scale 2D heat equation example (~10 min with 10 procs)\n");
      printf(" -----------------------------------------------------------\n");
      printf("  -run_wrapper_tests                 : Only run the XBraid wrapper tests\n");
      printf("  -pgrid  <px py pt>                 : processors in each dimension (default: 2 1 1)\n");
      printf("  -nx  <nx ny>                       : 2D spatial problem size (default: 129 129)\n");
      printf("  -nt  <n>                           : number of time steps (default: 2048)\n");
      printf("  -cfl <cfl>                         : CFL number (default: 0.30)\n");
      printf("  -ml  <max_levels>                  : set max number of time levels (default: 16)\n");
      printf("  -skip <skip>                       : boolean, whether to skip all work on first down cycle (default: 1)\n");
      printf("  -mc  <min_coarse>                  : set min possible coarse level size (default: 3)\n");
      printf("  -nu  <nrelax>                      : set num F-C relaxations (default: 1)\n");
      printf("  -nu0 <nrelax>                      : set num F-C relaxations on level 0\n");
      printf("  -tol <tol>                         : set stopping tolerance (default: 1e-09)\n");
      printf("  -tnorm <tnorm>                     : set temporal norm (1: one-norm, 2: two-norm (default), 3: infinity-norm)\n");
      printf("  -cf  <cfactor>                     : set coarsening factor (default: 2)\n");
      printf("  -cf0  <cfactor>                    : set coarsening factor for level 0\n");
      printf("  -mi  <max_iter>                    : set max iterations (default: 50)\n");
      printf("  -pfmg_mi <max_iter max_iter_cheap> : maximum number of PFMG iterations (default: 100 100)\n");
      printf("  -pfmg_tol  <tol_x>                 : PFMG halting tolerance (default: 1e-09)\n");
      printf("  -fmg                               : use FMG cycling\n");
      printf("  -res                               : use my residual\n");
      printf("  -storage <level>                   : full storage on levels >= level\n");
      printf("  -forcing                           : consider non-zero RHS (default: ON)\n");
      printf("  -use_rand <bool>                   : if nonzero, use uniformly random initial guess for t>0\n");
      printf("  -fullrnorm                         : use user residual routine for full residual each iteration\n");
      printf("\n");
      printf(" Example run (10 processors, ~10 min):\n");
      printf("   mpirun -np 10 ex-08 -pgrid 2 1 5 -ml 16 -nt 2048 -nx 129 129 -mi 50\n");
      printf("\n");
   }

   if( print_usage ){
      MPI_Finalize();
      return (0);
   }

   /* Check the processor grid (px x py x pt = num_procs?). */
   MPI_Comm_size( comm, &num_procs );
   if( ((app->man->px)*(app->man->py)*(app->pt)) != num_procs)
   {
       if( myid == 0 )
           printf("Error: px x py x pt does not equal the number of processors!\n");
       MPI_Finalize();
       return (0);
   }

   /* Create communicators for the time and space dimensions */
   braid_SplitCommworld(&comm, (app->man->px)*(app->man->py), &comm_x, &comm_t);
   app->man->comm = comm_x;
   app->comm = comm;
   app->comm_t = comm_t;
   app->comm_x = comm_x;

   /* Determine position (pi, pj)  in the 2D processor grid,
    * 0 <= pi < px,   0 <= pj < py */
   MPI_Comm_rank( comm_x, &myid );
   app->man->pi = myid % (app->man->px);
   app->man->pj = ( (myid - app->man->pi)/(app->man->px) ) % (app->man->py);

   /* Define the 2D block of the global grid owned by this processor */
   GetDistribution_x( (app->man->nx), (app->man->px), (app->man->pi),
                      &(app->man->ilower[0]), &(app->man->iupper[0]) );
   GetDistribution_x( (app->man->ny), (app->man->py), (app->man->pj),
                      &(app->man->ilower[1]), &(app->man->iupper[1]) );

   /* Determine local problem size. */
   app->man->nlx = app->man->iupper[0] - app->man->ilower[0] + 1;
   app->man->nly = app->man->iupper[1] - app->man->ilower[1] + 1;

   /* Compute grid spacing. */
   app->man->dx = PI / (app->man->nx - 1);
   app->man->dy = PI / (app->man->ny - 1);

   /* Set time-step size */
   app->man->dt = (cfl*(app->man->dx)*(app->man->dx)*(app->man->dy)*(app->man->dy))/
                  (app->man->K*((app->man->dx)*(app->man->dx) + (app->man->dy)*(app->man->dy)));
   app->man->tstop =  app->man->tstart + app->man->nt*app->man->dt;

   /* Set up the variable type, grid, stencil and matrix graph. */
   setUp2Dgrid( comm_x, &(app->man->grid_x), app->man->dim_x,
                app->man->ilower, app->man->iupper, app->man->vartype, 1 );
   set5ptStencil( &(app->man->stencil), app->man->dim_x );
   setUpGraph( comm_x, &(app->man->graph), app->man->grid_x,
               app->man->object_type, app->man->stencil );

   /* Allocate items of app */
   max_nA = 512*max_levels; /* generous value */
   initialize_vector(app->man, &(app->e));
   app->A = (HYPRE_SStructMatrix*) malloc( max_nA*sizeof(HYPRE_SStructMatrix));
   app->dt_A = (double*) malloc( max_nA*sizeof(double) );
   for( i = 0; i < max_nA; i++ ) {
      app->dt_A[i] = -1.0;
   }
   app->nA = 0;
   app->max_nA = max_nA;

   /* Allocate memory for array of solvers. */
   app->solver = (HYPRE_StructSolver*) malloc( max_nA*sizeof(HYPRE_StructSolver));

   /* Array for tracking runtime iteration counts of PFMG. */
   app->runtime_max_iter = (int*) calloc( max_nA,  sizeof(int) );
   for( i = 0; i < max_nA; i++ )
      app->runtime_max_iter[i] = 0;

   if( run_wrapper_tests)
   {
      /* Run the XBraid wrapper tests */
      mytime = 0.0;
      for(i = 0; i < 2; i++){
         braid_TestInitAccess( app, comm_x, stdout, mytime, my_Init, my_Access, my_Free);
         braid_TestClone( app, comm_x, stdout, mytime, my_Init, my_Access, my_Free, my_Clone);
         braid_TestSum( app, comm_x, stdout, mytime, my_Init, my_Access, my_Free, my_Clone, my_Sum);
         braid_TestResidual(app, comm_x, stdout, mytime, app->man->dt, my_Init, my_Access, my_Free, my_Clone, my_Sum, my_SpatialNorm, my_Residual, my_Step);
         correct1 = braid_TestSpatialNorm( app, comm_x, stdout, mytime, my_Init, my_Free, my_Clone, my_Sum, my_SpatialNorm);
         correct2 = braid_TestBuf( app, comm_x, stdout, mytime, my_Init, my_Free, my_Sum, my_SpatialNorm, my_BufSize, my_BufPack, my_BufUnpack);
         mytime += app->man->dt;

         if( (correct1 == 0) || (correct2 == 0)) {
           printf("Failed: at least one of the tests failed\n");
         }
         else {
           printf("Passed: all tests passed\n");
         }
      }
   }
   else
   {
      /* Run XBraid simulation */

      mystarttime = MPI_Wtime();
      braid_Init(comm, comm_t, app->man->tstart, app->man->tstop, app->man->nt,
                 app, my_Step, my_Init, my_Clone, my_Free, my_Sum,
                 my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core);

      /* Set Braid parameters */
      braid_SetSkip( core, skip );
      braid_SetMaxLevels( core, max_levels );
      braid_SetMinCoarse( core, min_coarse );
      braid_SetPrintLevel( core, print_level);
      braid_SetAccessLevel( core, access_level);
      braid_SetNRelax(core, -1, nrelax);
      braid_SetSeqSoln(core, use_seq_soln);
      if (nrelax0 > -1) {
         braid_SetNRelax(core,  0, nrelax0);
      }
      braid_SetAbsTol(core, tol /
         sqrt( (app->man->dx)*(app->man->dy)*(app->man->dt)) );
      braid_SetTemporalNorm(core, tnorm);
      braid_SetCFactor(core, -1, cfactor);
      if (fullrnorm) {
        braid_SetFullRNormRes(core, my_Residual);
      }
      if( cfactor0 > 0 ) {
         braid_SetCFactor(core,  0, cfactor0);
      }
      braid_SetMaxIter(core, max_iter);
      if (fmg) {
         braid_SetFMG(core);
      }
      if (res) {
         braid_SetResidual(core, my_Residual);
      }
      if (storage >= -2) {
         braid_SetStorage(core, storage);
      }

      MPI_Comm_rank( comm, &myid );
      if( myid == 0 ) {
         printf("\n  --------------------- \n");
         printf("  ex-08: Large-scale 2D heat equation simulation\n");
         printf("  --------------------- \n\n");
         printf("  Problem configuration:\n");
         printf("    Spatial grid:     %d x %d\n", app->man->nx, app->man->ny);
         printf("    Spatial DOFs:     %d\n", app->man->nx * app->man->ny);
         printf("    Time steps:       %d\n", app->man->nt);
         printf("    MPI procs:        %d (space: %dx%d, time: %d)\n",
                num_procs, app->man->px, app->man->py, app->pt);
         printf("    CFL ratio:        %1.2e\n", app->man->K*((app->man->dt)/((app->man->dx)*(app->man->dx)) + (app->man->dt)/((app->man->dy)*(app->man->dy))));
         printf("    PFMG max iters:   %d (fine), %d (coarse)\n", app->max_iter_x[0], app->max_iter_x[1]);
         printf("\n");
      }

      /* This call "Drives" or runs the simulation */
      braid_Drive(core);

      /* Compute run time */
      myendtime = MPI_Wtime();
      mytime    = myendtime - mystarttime;
      MPI_Reduce( &mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm );

      /* Determine maximum number of iterations for hypre PFMG spatial solves at each time level */
      MPI_Allreduce( &(app->nA), &nA_max, 1, MPI_INT, MPI_MAX, comm );
      runtime_max_iter_global = (int*) malloc( nA_max*sizeof(int) );
      for( i = 0; i < nA_max; i++ ){
         MPI_Allreduce( &(app->runtime_max_iter[i]),
                        &runtime_max_iter_global[i], 1, MPI_INT, MPI_MAX, comm );
      }

      if( myid == 0 ) {
         printf("  --------------------- \n");
         printf("  End simulation \n");
         printf("  --------------------- \n\n");

         printf("  Spatial grid size:                %d,%d\n", app->man->nx, app->man->ny);
         printf("  Spatial mesh width (dx,dy):      (%1.2e, %1.2e)\n", app->man->dx, app->man->dy);
         printf("  CFL ratio:                        %1.2e\n", app->man->K*((app->man->dt)/((app->man->dx)*(app->man->dx)) + (app->man->dt)/((app->man->dy)*(app->man->dy))));
         printf("  Time step size:                   %1.2e\n", app->man->dt);
         printf("  Number of time steps:             %d\n", app->man->nt);
         printf("  Run time:                         %1.2e seconds (%.2f minutes)\n", maxtime, maxtime/60.0);
         printf("\n   Level   Max PFMG Iters\n");
         printf("  -----------------------\n");
         for(i = 0; i < nA_max; i++){
            printf("     %d           %d\n", i, runtime_max_iter_global[i]);
         }
         printf("\n");
      }

      braid_Destroy(core);
   }

   /* Free app->man structures */
   HYPRE_SStructGridDestroy( app->man->grid_x );
   HYPRE_SStructStencilDestroy( app->man->stencil );
   HYPRE_SStructGraphDestroy( app->man->graph );

   /* Free app-> structures */
   for( i = 0; i < app->nA; i++ ) {
      HYPRE_SStructMatrixDestroy( app->A[i] );
      HYPRE_StructPFMGDestroy( app->solver[i] );
   }
   HYPRE_SStructVectorDestroy( app->e );
   free( app->man );
   free( app->dt_A );
   free( app->A );
   free( app->solver );
   free( app->runtime_max_iter );
   free( app->max_iter_x );
   free( app );

   /* Other Free */
   if( runtime_max_iter_global != NULL) {
      free( runtime_max_iter_global );
   }
   MPI_Comm_free( &comm_x );
   MPI_Comm_free( &comm_t );

   MPI_Finalize();
   return 0;
}
