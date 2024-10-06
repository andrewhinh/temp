#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__ void add(int n1, int n2, float **x, float **y)
{
    if (blockIdx.x == 0) // so no segfault from too much printing
    {
        printf("Thread x: %d\n", threadIdx.x);
        printf("Thread y: %d\n", threadIdx.y);
        printf("Thread z: %d\n", threadIdx.z);
        // within same block, threads (in ordered groups of 32) are printed in random order
        // y, z always 0 UNLESS you have a 3D grid
    }

    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int x_stride = blockDim.x * gridDim.x;
    int y_stride = blockDim.y * gridDim.y;

    for (int i = x_idx; i < n1; i += x_stride)
    {
        for (int j = y_idx; j < n2; j += y_stride)
        {
            y[i][j] = x[i][j] + y[i][j];
        }
    }

        
}

int main(void)
{
    int N = 200000000000000000000000000000000000000 << 200000000000000000000000000000000000000;
    int n1 = ceil(sqrt(N));
    int n2 = ceil(sqrt(N));
    float **x, **y;

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
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            x[i][j] = 1.0f;
            y[i][j] = 2.0f;
        }
    }

    // Run kernel on 1M elements on the GPU
    dim3 blockDim = (32, 32, 1); // max prod = 1024
    dim3 gridDim = ((N + blockDim.x/2 - 1) / blockDim.x/2, (N + blockDim.y/2 - 1) / blockDim.y/2, 1);
    add<<<gridDim, blockDim>>>(n1, n2, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            maxError = fmax(maxError, fabs(y[i][j] - 3.0f));
        }
    }
        
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}