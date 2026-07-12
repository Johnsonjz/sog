# CMakeLists.txt — Standalone build for libsog_lmp.so (SOG kspace_style plugin)
#
# Build requirements:
#   - LAMMPS source tree (headers only, e.g. v30Mar2026)
#   - C++17 compiler
#   - FFTW3 or MKL (LAMMPS FFT dependency)
#
# Usage:
#   cmake -S . -B build \
#     -DLAMMPS_SOURCE_ROOT=/path/to/lammps_stable \
#     -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   # → build/libsog_lmp.so

cmake_minimum_required(VERSION 3.16)
project(sog_lmp VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Locate LAMMPS headers ──
if(NOT DEFINED LAMMPS_SOURCE_ROOT)
  message(FATAL_ERROR "LAMMPS_SOURCE_ROOT must be set to the LAMMPS source tree root.")
endif()

set(LAMMPS_SRC ${LAMMPS_SOURCE_ROOT}/src)
if(NOT EXISTS "${LAMMPS_SRC}/kspace.h")
  message(FATAL_ERROR "LAMMPS headers not found at ${LAMMPS_SRC} (missing kspace.h).")
endif()

# ── SOG kspace sources ──
set(SOG_SOURCES
  sog.cpp
)

set(SOG_HEADERS
  sog.h
  sog_spline.h
)

# ── Build shared library ──
add_library(sog_lmp SHARED ${SOG_SOURCES})

target_include_directories(sog_lmp PRIVATE
  ${LAMMPS_SRC}
  ${LAMMPS_SRC}/KSPACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# LAMMPS compile definitions (required for FFT3d, KSpace, etc.)
# Must match the target LAMMPS build's size setting (default LAMMPS_SMALLBIG); the old
# BIGBIG+SMALLSMALL pair was contradictory and SMALLSMALL is no longer supported.
target_compile_definitions(sog_lmp PRIVATE
  LAMMPS_SMALLBIG
)

# ── MPI (LAMMPS headers pull in mpi.h) ──
find_package(MPI REQUIRED)
target_link_libraries(sog_lmp PRIVATE MPI::MPI_CXX)

# ── FFT library (LAMMPS requires FFT3d) ──
# Default: try FFTW3. Override with -DFFT_LIB=<path> if needed.
find_library(FFTW3_LIB fftw3)
if(FFTW3_LIB)
  target_link_libraries(sog_lmp PRIVATE ${FFTW3_LIB})
  target_compile_definitions(sog_lmp PRIVATE FFT_FFTW3)
  message(STATUS "Using FFTW3: ${FFTW3_LIB}")
else()
  find_library(MKL_LIB mkl_rt)
  if(MKL_LIB)
    target_link_libraries(sog_lmp PRIVATE ${MKL_LIB})
    target_compile_definitions(sog_lmp PRIVATE FFT_MKL)
    message(STATUS "Using MKL: ${MKL_LIB}")
  else()
    message(WARNING "No FFT library found (FFTW3 or MKL). Link manually or set FFT_LIB.")
  endif()
endif()

# ── OpenMP (optional, for multi-threaded FFT) ──
find_package(OpenMP)
if(OpenMP_FOUND)
  target_link_libraries(sog_lmp PRIVATE OpenMP::OpenMP_CXX)
endif()

# ── Install ──
install(TARGETS sog_lmp LIBRARY DESTINATION lib)
