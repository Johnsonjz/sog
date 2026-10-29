# op/CMakeLists.txt — PyTorch custom-op library for SOG (``sogkspace::*``).
#
# Builds ``libsog_kspace_op.so``: the in-graph k-space solver
# (``sogkspace::solve_auto``), the charge-gauge projection
# (``sogkspace::neutralize``) and its backward reduction
# (``sogkspace::qmean``), plus the setup/diagnostics entry points.
#
# The kernels are the deployed ones: this target compiles
# ``lmp/sog_gpu.cu`` UNCHANGED (the same code ``kspace_style sog/kk``
# runs), so the op reproduces the deployed discretization exactly.
#
# This project is deliberately INDEPENDENT of LAMMPS: the op is loaded into a
# PyTorch process (training / freeze / ASE / a LAMMPS pair style via
# DeepPot), not into a LAMMPS plugin.  It therefore follows libtorch's C++
# ABI — do NOT apply the LAMMPS plugin's ``SOG_GLIBCXX_OLD_ABI`` here.
#
# Usage:
#   cmake -S op -B build_op \
#     -DTORCH_DIR=/path/to/torch/share/cmake/Torch \
#     -DCMAKE_BUILD_TYPE=Release
#   cmake --build build_op -j
#   # → build_op/libsog_kspace_op.so
#
# Load it from Python with ``sog.v2.kspace_op.load_op_library(path)`` or via
# the ``SOG_KS_OP_LIB`` environment variable.

cmake_minimum_required(VERSION 3.20)
project(sog_kspace_op VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SOG_WITH_MPI "Compile the replicated-grid MPI_Allreduce path" OFF)
set(SOG_CUDA_ARCHITECTURES "" CACHE STRING
    "CUDA architectures for the op (empty = CMake auto-detection)")

# ── torch (must match the ABI of the process that will load the op) ──
if(DEFINED TORCH_DIR)
  find_package(Torch REQUIRED PATHS ${TORCH_DIR} NO_DEFAULT_PATH)
else()
  find_package(Torch REQUIRED)
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

if(CMAKE_CXX_COMPILER MATCHES "mpiCC|mpicxx|mpiCC")
  message(FATAL_ERROR
    "CXX must be a plain C++ compiler (g++), not an MPI compiler wrapper: "
    "Inductor's PCH and torch's ABI probing break under mpiCC. "
    "Current: ${CMAKE_CXX_COMPILER}")
endif()

# ── sources ──
# ``sog_kspace_op.cc`` includes "sog_gpu.cuh" and "sog_green.h" — the
# include path below is resolved from the op/ and lmp/ trees.
set(SOG_OP_SOURCES
  sog_kspace/sog_kspace_op.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/../lmp/sog_gpu.cu
)

add_library(sog_kspace_op MODULE ${SOG_OP_SOURCES})

target_include_directories(sog_kspace_op PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/sog_kspace
  ${CMAKE_CURRENT_SOURCE_DIR}/../lmp
)

target_link_libraries(sog_kspace_op PRIVATE
  ${TORCH_LIBRARIES}
  CUDA::cufft
  CUDA::cudart
)

# Match libtorch's flags (this is what sets/keeps the C++11 ABI consistent).
target_compile_options(sog_kspace_op PRIVATE ${TORCH_CXX_FLAGS})

set_target_properties(sog_kspace_op PROPERTIES
  PREFIX "lib"
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
)

if(SOG_CUDA_ARCHITECTURES)
  set_target_properties(sog_kspace_op PROPERTIES
    CUDA_ARCHITECTURES ${SOG_CUDA_ARCHITECTURES})
endif()

if(APPLE)
  set_target_properties(sog_kspace_op PROPERTIES INSTALL_RPATH "@loader_path")
else()
  set_target_properties(sog_kspace_op PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

# ── optional MPI (replicated-grid Allreduce; single-rank builds work without) ──
if(SOG_WITH_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(sog_kspace_op PRIVATE MPI::MPI_CXX)
  target_compile_definitions(sog_kspace_op PRIVATE USE_MPI)
endif()

install(TARGETS sog_kspace_op LIBRARY DESTINATION lib)
