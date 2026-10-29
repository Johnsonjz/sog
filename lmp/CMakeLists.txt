# CMakeLists.txt — standalone build for libsog_lmp.so (SOG LAMMPS plugin)
#
# Build modes
#   native (default):        kspace_style sog + fix sog/response with the
#                            native charge-source path (atom->q) and the
#                            NativeChargeResponse no-op adapter.
#   -DSOG_WITH_DEEPMD=ON:    additionally compiles the DeepMD adapters
#                            (couplings #2/#3 against pair_style deepmd).
#   -DSOG_WITH_GPU=ON:       compile sog_gpu.cu (raw CUDA + cuFFT) instead
#                            of the CPU-only stub.
#
# Requirements:
#   - LAMMPS source tree (headers only, v30Mar2026 or newer for the FFT3d
#     signature sog.cpp uses)
#   - C++17 compiler
#   - FFTW3 or MKL (LAMMPS FFT dependency)
#   Deepmd mode additionally:
#   - DEEPMD_SOURCE_ROOT: deepmd-kit source dir (source/api_cc/include/DeepPot.h
#     must exist — this fork's C++ API with the charge-response extensions)
#   - libtorch headers (-DTORCH_DIR=<libtorch> or a findable Torch CMake config)
#   - libdeepmd_cc (-DDEEPMD_CC_LIBRARY=<path> or -DDEEPMD_BUILD_DIR=<build dir>)
#
# Usage:
#   cmake -S . -B build \
#     -DLAMMPS_SOURCE_ROOT=/path/to/lammps_stable \
#     -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   # → build/libsog_lmp.so
#
# Load in LAMMPS with `plugin load /path/to/libsog_lmp.so`.

cmake_minimum_required(VERSION 3.16)
project(sog_lmp VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SOG_WITH_DEEPMD "Build the DeepMD charge-source/response adapters" OFF)
option(SOG_WITH_GPU "Compile the CUDA SOG kspace (sog_gpu.cu)" OFF)
# The libstdc++ ABI must match the LAMMPS binary that will load the plugin.
# The integrated build_deepmd_integrated binary here was built with
# -D_GLIBCXX_USE_CXX11_ABI=0 (gcc-toolset-13); a stock conda-gcc LAMMPS
# build uses the new ABI and needs this OFF.
set(SOG_GLIBCXX_OLD_ABI ON CACHE BOOL
    "Compile with -D_GLIBCXX_USE_CXX11_ABI=0 to match the target LAMMPS binary")

# ── Locate LAMMPS headers ──
if(NOT DEFINED LAMMPS_SOURCE_ROOT)
  message(FATAL_ERROR "LAMMPS_SOURCE_ROOT must be set to the LAMMPS source tree root.")
endif()

set(LAMMPS_SRC ${LAMMPS_SOURCE_ROOT}/src)
if(NOT EXISTS "${LAMMPS_SRC}/kspace.h")
  message(FATAL_ERROR "LAMMPS headers not found at ${LAMMPS_SRC} (missing kspace.h).")
endif()

# ── LAMMPS version number (pair_deepmd.h requires it in deepmd mode) ──
function(sog_lammps_version_number version_header out_var)
  file(STRINGS ${version_header} line REGEX "LAMMPS_VERSION ")
  set(MONTHS x Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec)
  if(line MATCHES "\"([0-9]+) ([A-Za-z]+) ([0-9]+)\"")
    set(DAY ${CMAKE_MATCH_1})
    set(YEAR ${CMAKE_MATCH_3})
    string(SUBSTRING ${CMAKE_MATCH_2} 0 1 MONTH_FIRST)
    string(TOUPPER ${MONTH_FIRST} MONTH_FIRST)
    string(TOLOWER ${CMAKE_MATCH_2} MONTH_LOWER)
    string(SUBSTRING ${MONTH_LOWER} 1 -1 MONTH_REST)
    set(MONTH_CAP "${MONTH_FIRST}${MONTH_REST}")
    list(FIND MONTHS ${MONTH_CAP} MONTH_INDEX)
    if(MONTH_INDEX EQUAL -1)
      message(FATAL_ERROR "Cannot parse LAMMPS month '${CMAKE_MATCH_2}' in ${version_header}")
    endif()
    string(LENGTH ${DAY} DAY_LEN)
    if(DAY_LEN EQUAL 1)
      set(DAY "0${DAY}")
    endif()
    string(LENGTH ${MONTH_INDEX} MONTH_LEN)
    if(MONTH_LEN EQUAL 1)
      set(MONTH_INDEX "0${MONTH_INDEX}")
    endif()
    set(${out_var} "${YEAR}${MONTH_INDEX}${DAY}" PARENT_SCOPE)
  else()
    message(FATAL_ERROR "Cannot parse LAMMPS version from ${version_header}")
  endif()
endfunction()

# ── SOG sources ──
set(SOG_SOURCES
  sog.cpp
  fix_sog_response.cpp
  sogplugin.cpp
)

set(SOG_HEADERS
  sog.h
  sog_spline.h
  sog_gpu.cuh
  sog_charge_source.h
  sog_charge_response.h
  fix_sog_response.h
  adapters/native_charge_response.h
)

if(SOG_WITH_GPU)
  enable_language(CUDA)
  # sog_gpu.cu uses atomicAdd(double), which requires sm_60+; CMake's default
  # architecture (sm_52) fails with "no instance of overloaded function".
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING
        "CUDA architectures for sog_gpu.cu (>=60 for double atomicAdd)")
  endif()
  list(APPEND SOG_SOURCES sog_gpu.cu)
  find_package(CUDAToolkit REQUIRED)
else()
  # CPU-only stand-ins so the library carries no undefined sog_gpu_* symbols.
  list(APPEND SOG_SOURCES sog_gpu_stub.cpp)
endif()

# ── Deepmd adapters ──
if(SOG_WITH_DEEPMD)
  if(NOT DEFINED DEEPMD_SOURCE_ROOT)
    message(FATAL_ERROR
      "SOG_WITH_DEEPMD=ON requires -DDEEPMD_SOURCE_ROOT=<deepmd-kit/source> "
      "(this fork, whose api_cc has the charge-response extensions).")
  endif()
  if(NOT EXISTS "${DEEPMD_SOURCE_ROOT}/api_cc/include/DeepPot.h")
    message(FATAL_ERROR "DeepPot.h not found under ${DEEPMD_SOURCE_ROOT}/api_cc/include")
  endif()

  # torch headers (DeepPot.h pulls in torch/script.h)
  if(DEFINED TORCH_DIR)
    find_package(Torch REQUIRED PATHS ${TORCH_DIR} NO_DEFAULT_PATH)
  else()
    find_package(Torch QUIET)
    if(NOT Torch_FOUND)
      message(FATAL_ERROR
        "SOG_WITH_DEEPMD=ON needs libtorch: pass -DTORCH_DIR=<libtorch with CMake config>")
    endif()
  endif()

  # libdeepmd_cc (the C++ API): explicit path, else guess from the build dir
  if(DEFINED DEEPMD_CC_LIBRARY)
    set(DEEPMD_CC_LIB ${DEEPMD_CC_LIBRARY})
  else()
    find_library(DEEPMD_CC_LIB deepmd_cc
      HINTS ${DEEPMD_BUILD_DIR}/api_cc
            ${DEEPMD_SOURCE_ROOT}/../build/api_cc
            ${DEEPMD_SOURCE_ROOT}/../../build/api_cc)
    if(NOT DEEPMD_CC_LIB)
      message(FATAL_ERROR
        "libdeepmd_cc not found: pass -DDEEPMD_CC_LIBRARY=<path> or "
        "-DDEEPMD_BUILD_DIR=<deepmd-kit build dir>")
    endif()
  endif()

  list(APPEND SOG_SOURCES adapters/deepmd_charge_response.cpp)
  list(APPEND SOG_HEADERS
    adapters/deepmd_charge_source.h
    adapters/deepmd_charge_response.h
  )
endif()

# ── Build shared library ──
add_library(sog_lmp SHARED ${SOG_SOURCES} ${SOG_HEADERS})

target_include_directories(sog_lmp PRIVATE
  ${LAMMPS_SRC}
  ${LAMMPS_SRC}/KSPACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# LAMMPS compile definitions (required for FFT3d, KSpace, etc.)
# Must match the target LAMMPS build's size setting (default LAMMPS_SMALLBIG).
target_compile_definitions(sog_lmp PRIVATE
  LAMMPS_SMALLBIG
)

if(SOG_GLIBCXX_OLD_ABI)
  target_compile_definitions(sog_lmp PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
  message(STATUS "libstdc++ ABI: old (-D_GLIBCXX_USE_CXX11_ABI=0)")
else()
  message(STATUS "libstdc++ ABI: new (C++11)")
endif()

if(SOG_WITH_DEEPMD)
  sog_lammps_version_number(${LAMMPS_SRC}/version.h SOG_LAMMPS_VERSION_NUMBER)
  target_include_directories(sog_lmp PRIVATE
    ${DEEPMD_SOURCE_ROOT}/lmp
    ${DEEPMD_SOURCE_ROOT}/api_cc/include
    # deepmd's private headers (errors.h, ...) — PUBLIC on the deepmd_lib
    # CMake target, but we link a raw libdeepmd_cc.so so we add it by hand.
    ${DEEPMD_SOURCE_ROOT}/lib/include
    ${Torch_INCLUDE_DIRS}
  )
  target_compile_definitions(sog_lmp PRIVATE
    SOG_WITH_DEEPMD
    DP_USE_CXX_API
    LMPPLUGIN
    LAMMPS_VERSION_NUMBER=${SOG_LAMMPS_VERSION_NUMBER}
  )
  target_link_libraries(sog_lmp PRIVATE ${DEEPMD_CC_LIB} ${Torch_LIBRARIES})
  message(STATUS "SOG deepmd adapters: ON (libdeepmd_cc: ${DEEPMD_CC_LIB})")
else()
  message(STATUS "SOG deepmd adapters: OFF (native charge-source path only)")
endif()

# ── MPI (LAMMPS headers pull in mpi.h) ──
# A plugin resolves MPI symbols from the LAMMPS binary at runtime, so we only
# need the headers — never link a second MPI (deepmd-kit's plugin recipe does
# the same).  Without a system MPI dev install, fall back to LAMMPS' own
# serial stub mpi.h.
find_package(MPI QUIET)
if(MPI_FOUND)
  target_include_directories(sog_lmp PRIVATE ${MPI_CXX_INCLUDE_DIRS})
else()
  target_include_directories(sog_lmp PRIVATE ${LAMMPS_SRC}/STUBS)
  message(STATUS "MPI not found — using LAMMPS STUBS serial mpi.h")
endif()

# ── FFT (the fft3d_wrap.h dispatcher must match the LAMMPS binary) ──
# Default FFT_KISS: built into LAMMPS itself, zero external dependencies —
# the kissfft symbols resolve from the LAMMPS binary.  Override with
# -DFFT=FFTW3 or -DFFT=MKL only if the target LAMMPS binary was built that
# way (then the matching dev headers/library must be available).
set(FFT "KISS" CACHE STRING "FFT backend matching the target LAMMPS build (KISS, FFTW3, MKL)")
if(FFT STREQUAL "FFTW3")
  find_path(FFTW3_INCLUDE_DIR fftw3.h REQUIRED)
  find_library(FFTW3_LIB fftw3 REQUIRED)
  target_include_directories(sog_lmp PRIVATE ${FFTW3_INCLUDE_DIR})
  target_link_libraries(sog_lmp PRIVATE ${FFTW3_LIB})
  target_compile_definitions(sog_lmp PRIVATE FFT_FFTW3)
elseif(FFT STREQUAL "MKL")
  find_library(MKL_LIB mkl_rt REQUIRED)
  target_link_libraries(sog_lmp PRIVATE ${MKL_LIB})
  target_compile_definitions(sog_lmp PRIVATE FFT_MKL)
  message(WARNING "FFT=MKL also needs the MKL DFTI headers on the include path")
elseif(FFT STREQUAL "KISS")
  target_include_directories(sog_lmp PRIVATE ${LAMMPS_SRC}/KISSFFT)
  target_compile_definitions(sog_lmp PRIVATE FFT_KISS)
  message(STATUS "Using KISS FFT (built into the LAMMPS binary)")
else()
  message(FATAL_ERROR "FFT must be KISS, FFTW3 or MKL (got '${FFT}')")
endif()

if(SOG_WITH_GPU)
  target_link_libraries(sog_lmp PRIVATE CUDA::cufft CUDA::cudart)
  set_target_properties(sog_lmp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# ── OpenMP (optional, for multi-threaded FFT) ──
find_package(OpenMP)
if(OpenMP_FOUND)
  target_link_libraries(sog_lmp PRIVATE OpenMP::OpenMP_CXX)
endif()

# ── Install ──
install(TARGETS sog_lmp LIBRARY DESTINATION lib)
