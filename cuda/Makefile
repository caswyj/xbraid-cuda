#BHEADER**********************************************************************
# Copyright (c) 2013, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
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
# CUDA compilers and flags
##################################################################

NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_70 --shared -Xcompiler -fPIC

# Include paths
INCLUDES = -I.. -I../braid -I.

# Source files
CUDA_SRCS = cuda_util.cu cuda_vector.cu cuda_relax.cu
C_SRCS = cuda_braid.c

# Object files
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
C_OBJS = $(C_SRCS:.c=.o)
OBJECTS = $(CUDA_OBJS) $(C_OBJS)

# Header files
HEADERS = cuda_braid.h cuda_vector.h cuda_relax.h cuda_util.h

.PHONY: all clean

all: libcudabraid.a

# Rule for CUDA files
%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Rule for C files (compile with nvcc since it includes cuda headers)
%.o: %.c $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

libcudabraid.a: $(OBJECTS)
	@echo "Building" $@ "..."
	ar cruv libcudabraid.a $(OBJECTS)
	ranlib libcudabraid.a

clean:
	rm -f *.o libcudabraid.a
