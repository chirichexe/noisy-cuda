#!/bin/bash

# input validation
if [ -z "$1" ]; then
  echo "Error: You must provide an integer for the CUDA version (e.g., 1, 2, 3)."
  echo "Usage: $0 <integer_version>"
  exit 1
fi

# variables
CUDA_VERSION_INT=$1
CUDA_VERSION="v${CUDA_VERSION_INT}"
BUILD_DIR="build/cuda"
OUTPUT_DIR="tests/outputs/cuda-ncu"

REPORT_FILE="${OUTPUT_DIR}/noisy-cuda-test-${CUDA_VERSION}.ncu-rep"

# Check and Create Output Directory ============================================
if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "Creating output directory: ${OUTPUT_DIR}"
  mkdir -p "${OUTPUT_DIR}"
fi

echo "Starting Profiling for Version: ${CUDA_VERSION}"

# Compilation ==================================================================
# CMake Configuration
echo "Configuring CMake for ${BUILD_DIR} with -DCUDA_VERSION=${CUDA_VERSION}"
cmake -B "${BUILD_DIR}" -DUSE_CUDA=on -DCUDA_VERSION="${CUDA_VERSION}"

if [ $? -ne 0 ]; then
  echo "Error during CMake configuration."
  exit 1
fi

# build
echo "Building the project..."
cmake --build "${BUILD_DIR}"

if [ $? -ne 0 ]; then
  echo "Error during project compilation."
  exit 1
fi

echo "Compilation successful."
echo ""

# Profiling (ncu) =============================================================
echo "Starting profiling with NVIDIA Nsight Compute (ncu)..."

# Using sudo is strongly discouraged, but sometimes necessary to access GPU profiling features
# Make sure the user has the necessary permissions to run ncu without sudo if possible
sudo ncu -f -o "${REPORT_FILE}" --target-processes all --set full "./${BUILD_DIR}/noisy_cuda" -n -v -C 10

if [ $? -ne 0 ]; then
  echo "Error during ncu execution. Check permissions or executable path."
  exit 1
fi

echo ""
echo "Profiling completed. Report saved to: ${REPORT_FILE}"

# Open niught-compute ========================================================
echo ""
echo "Opening the report in Nsight Compute GUI (ncu-ui)..."
echo "NOTE: If the GUI does not open, you may need to launch it manually:"
echo ""
echo "ncu-ui ${REPORT_FILE}"
echo ""

# launch the GUI application in the background
ncu-ui "${REPORT_FILE}" &

if [ $? -ne 0 ]; then
  echo "Warning: 'ncu-ui' was not found or the launch failed."
  echo "Please open the report manually with 'ncu-ui ${REPORT_FILE}'"
  echo ""
fi

echo "Test finished. Bye."
