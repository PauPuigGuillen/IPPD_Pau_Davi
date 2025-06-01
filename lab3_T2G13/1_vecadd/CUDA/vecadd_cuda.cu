#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define BLOCKSIZE 128

// CUDA kernel for vector addition with the required signature
__global__ void vecadd_cuda(double *A, double *B, double *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

// CUDA vector addition wrapper function with detailed timing
void vecadd_wrapper(double *h_A, double *h_B, double *h_C, const int N)
{
    double *d_A, *d_B, *d_C;
    size_t size = N * sizeof(double);
    float h2d_time = 0.0f, kernel_time = 0.0f, d2h_time = 0.0f;
    
    // Event for timing host to device, kernel execution, and device to host (in that order)
    cudaEvent_t start_h2d, end_h2d;
    cudaEvent_t start_kernel, end_kernel;
    cudaEvent_t start_d2h, end_d2h;
    
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&end_h2d);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&end_kernel);
    cudaEventCreate(&start_d2h);
    cudaEventCreate(&end_d2h);

    // Allocate device memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Time Host to Device copy
    cudaMemset(d_C, 0, size);
    cudaEventRecord(start_h2d, 0);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaEventRecord(end_h2d, 0);
    cudaEventSynchronize(end_h2d);
    cudaEventElapsedTime(&h2d_time, start_h2d, end_h2d);

    // Time kernel execution
    cudaEventRecord(start_kernel, 0);
    int numBlocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;
    vecadd_cuda<<<numBlocks, BLOCKSIZE>>>(d_A, d_B, d_C, N);
    //cudaDeviceSynchronize();
    cudaEventRecord(end_kernel, 0);
    cudaEventSynchronize(end_kernel);
    cudaEventElapsedTime(&kernel_time, start_kernel, end_kernel);

    // Time Device to Host copy
    cudaEventRecord(start_d2h, 0);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(end_d2h, 0);
    cudaEventSynchronize(end_d2h);
    cudaEventElapsedTime(&d2h_time, start_d2h, end_d2h);

    // Print timing information
    printf(" Copy A and B Host to Device elapsed time: %.9f seconds\n", h2d_time / 1000.0f);
    printf(" Kernel elapsed time: %.9f seconds\n", kernel_time / 1000.0f);
    printf(" Copy C Device to Host elapsed time: %.9f seconds\n", d2h_time / 1000.0f);
    printf(" Total elapsed time: %.9f seconds\n", (h2d_time + kernel_time + d2h_time) / 1000.0f);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Clean up events
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(end_h2d);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(end_kernel);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(end_d2h);
}

int main(int argc, char *argv[])
{
    int N;

    if (argc != 2)
    {
        printf("Usage: %s <vector size N>\n", argv[0]);
        return 1;
    }
    else
    {
        N = atoi(argv[1]);
    }
    printf("Vector size: %d\n", N);

    // Memory allocation
    double *A = (double *)malloc(N * sizeof(double));
    double *B = (double *)malloc(N * sizeof(double));
    double *C = (double *)malloc(N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Host memory allocation failed\n");
        return 1;
    }

    // Initialize vectors
    for (int i = 0; i < N; i++)
    {
        A[i] = (double)i;
        B[i] = 2.0 * (N - i);
    }

    // Call the wrapper with CUDA event timing
    vecadd_wrapper(A, B, C, N);

    // Validation
    int errors = 0;
    for (int i = 0; i < N; i++)
    {
        if (fabs(C[i] - (2.0 * N - i)) > 1e-6)
        {
            printf("Validation failed at index %d: C[%d] = %f, expected = %f\n", 
                   i, i, C[i], 2.0 * N - i);
            errors++;
        }
    }
    
    if (errors == 0) {
        printf("Validation successful! All values match expected results.\n");
    }

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}