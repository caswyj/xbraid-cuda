#BHEADER**********************************************************************
#
# Copyright (c) 2013, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory. Written by
# Jacob Schroder, Rob Falgout, Tzanio Kolev, Ulrike Yang, Veselin
# Dobrev, et al. LLNL-CODE-660355. All rights reserved.
#
# This file is part of XBraid. For support, post issues to the XBraid Github page.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License (as published by the Free Software
# Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the terms and conditions of the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program; if not, write to the Free Software Foundation, Inc., 59
# Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
#EHEADER**********************************************************************

##################################################################
# Import machine specific compilers, options, flags, etc..
##################################################################

.PHONY: all braid cuda examples clean

# Default to building only MPI version if CUDA is not available
# Set CUDA_AVAILABLE=yes to build CUDA version
ifdef CUDA_AVAILABLE
ALL_TARGETS = braid cuda examples
else
ALL_TARGETS = braid examples
endif

all: $(ALL_TARGETS)

# Phony targets that just run make in subdirectories
braid:
	@$(MAKE) -C braid

cuda:
	@$(MAKE) -C cuda

examples: braid cuda
ifdef CUDA_AVAILABLE
	@$(MAKE) -C examples CUDA_AVAILABLE=yes
else
	@$(MAKE) -C examples
endif

drivers: braid
	@$(MAKE) -C drivers

clean:
	@$(MAKE) -C examples clean
	@$(MAKE) -C drivers clean
	@$(MAKE) -C braid clean
	@$(MAKE) -C cuda clean

info:
	@echo "MPICC     = `which $(MPICC)`"
	@echo "MPICXX    = `which $(MPICXX)`"
	@echo "MPIF90    = `which $(MPIF90)`"
	@echo "NVCC      = `which $(NVCC) 2>/dev/null || echo 'not found'`"
	@echo "CFLAGS    = $(CFLAGS)"
	@echo "CXXFLAGS  = $(CXXFLAGS)"
	@echo "FORTFLAGS = $(FORTFLAGS)"
	@echo "LFLAGS    = $(LFLAGS)"
