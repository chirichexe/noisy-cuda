/*
 * cuda_kernel.cu - specific run of a kernel CUDA implementation
 *
 */

/*
 * Copyright 2025 Davide Chirichella, Filippo Giulietti
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda_kernel.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void simple_print_kernel(int num_threads) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only print from first few threads to avoid clutter
    //if (thread_id < 10) {
        printf("Hello from CUDA! I'm thread %d (block %d, thread %d) of %d total threads\n", 
               thread_id, blockIdx.x, threadIdx.x, num_threads);
    //}
    
    // Thread 0 prints a summary
    if (thread_id == 0) {
        printf("\n\n=== CUDA Kernel Summary ===\n");
        printf("Total threads launched: %d\n", num_threads);
        printf("Grid dimensions: %d blocks\n", gridDim.x);
        printf("Block dimensions: %d threads\n", blockDim.x);
        printf("===========================\n\n");
    }
}

int run_cuda_kernel(int num_threads) {
    printf("Launching CUDA kernel with %d threads...\n", num_threads);
    
    // Simple kernel configuration
    int block_size = 256;
    int grid_size = (num_threads + block_size - 1) / block_size;
    
    printf("Kernel configuration: <<<%d blocks, %d threads>>>\n", grid_size, block_size);
    
    // Launch the kernel
    simple_print_kernel<<<grid_size, block_size>>>(num_threads);
    
    // Wait for kernel to complete
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    printf("CUDA kernel completed successfully!\n");
    return 0;
}