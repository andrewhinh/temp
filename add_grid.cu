#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y)
{
    if (blockIdx.x == 0) // so no segfault from too much printing
    {
        printf("Thread x: %d\n", threadIdx.x);
        printf("Thread y: %d\n", threadIdx.y);
        printf("Thread z: %d\n", threadIdx.z);
        // within same block, threads (in ordered groups of 32) are printed in random order
        // y, z always 0 UNLESS you have a 3D grid
    }

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 200000000000000000000000000000000000000 << 200000000000000000000000000000000000000;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaError_t x_err = cudaMallocManaged(&x, N * sizeof(float));
    if (x_err != cudaSuccess)
    {
        printf("cudaMallocManaged for x failed: %s in %s at line %d\n", cudaGetErrorString(x_err), __FILE__, __LINE__);
    }
    cudaError_t y_err = cudaMallocManaged(&y, N * sizeof(float));
    if (y_err != cudaSuccess)
    {
        printf("cudaMallocManaged for y failed: %s in %s at line %d\n", cudaGetErrorString(y_err), __FILE__, __LINE__);
    }

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    int blockSize = 1024;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}