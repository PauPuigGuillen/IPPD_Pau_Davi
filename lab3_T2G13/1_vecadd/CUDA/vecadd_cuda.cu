#define _POSIX_C_SOURCE 199309L
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

// CUDA vector addition wrapper function (renamed to avoid conflict with kernel)
void vecadd_wrapper(double *h_A, double *h_B, double *h_C, const int N)
{
    double *d_A, *d_B, *d_C;
    size_t size = N * sizeof(double);

    // Allocate device memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy vectors from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel with calculated grid size
    int numBlocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;
    vecadd_cuda<<<numBlocks, BLOCKSIZE>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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

    // Initialize vectors
    for (int i = 0; i < N; i++)
    {
        A[i] = (double)i;
        B[i] = 2.0 * (N - i);
    }

    // Vector addition timing
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    vecadd_wrapper(A, B, C, N);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1.0e9;
    printf("Elapsed time: %.9f seconds\n", elapsed);

    // Validation
    int errors = 0;
    for (int i = 0; i < N; i++)
    {
        if (fabs(C[i] - (2.0 * N - i)) > 1e-6)
        {
            printf("Validation failed at index %d: C[%d] = %f, expected = %f\n", 
                   i, i, C[i], 2.0 * N - i);
            errors++;
            if (errors > 10) {
                printf("Too many errors, stopping validation\n");
                break;
            }
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