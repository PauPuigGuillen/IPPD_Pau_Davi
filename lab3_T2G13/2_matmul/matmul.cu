#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cublas_v2.h>

#define BLOCKSIZE 16

// CUDA ERROR CHECK
#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                              \
        }                                                         \
    } while (0)

// TODO
// Sequential Matrix Multiplication
void matmul_seq(double *A, double *B, double *C, const int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i * N + j] = 0.0;
            for (int k = 0; k < N; k++)
            {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

// TODO
// Simple CUDA Matrix Multiplication Kernel
__global__ void matmul_naive_kernel(double *A, double *B, double *C, const int N)
{
    int row = blockIdx.y * BLOCKSIZE + threadIdx.y;
    int col = blockIdx.x * BLOCKSIZE + threadIdx.x;
    
    if (row < N && col < N) {
        double sum = 0.0;

        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }

}

// TODO
// Matrix Multiplication Kernel exploiting shared memory
__global__ void matmul_shared_kernel(double *A, double *B, double *C, const int N)
{
    __shared__ double A_tile[BLOCKSIZE][BLOCKSIZE];
    __shared__ double B_tile[BLOCKSIZE][BLOCKSIZE];
    
    int i = blockIdx.y * BLOCKSIZE + threadIdx.y;
    int j = blockIdx.x * BLOCKSIZE + threadIdx.x;
    
    double sum = 0.0;

    for (int tile = 0; tile < N; tile += BLOCKSIZE) {
        if (i < N && tile + threadIdx.x < N) {
            A_tile[threadIdx.y][threadIdx.x] = A[i * N + tile + threadIdx.x];
        }
        if (j < N && tile + threadIdx.y < N) {
            B_tile[threadIdx.y][threadIdx.x] = B[(tile + threadIdx.y) * N + j];
        }
        
        __syncthreads();

        if (i < N && j < N) {
            for (int k = 0; k < BLOCKSIZE && tile + k < N; k++) {
                sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
            }
        }
        
        __syncthreads();
    }

    if (i < N && j < N) {
        C[i * N + j] = sum;
    }
}

void validation(double *h_C, double *C, const int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double err = fabs(h_C[i * N + j] - C[i * N + j]);
            if (err > 1.0e-6)
            {
                printf("Error at C[%d][%d]: fabs( %f - %f ) = %e > %e\n", i, j, h_C[i * N + j], C[i * N + j], err, 1.0e-6);
                exit(1);
            }
        }
    }
}

void copy_A_B_H2D(double *h_A, double *h_B, double *d_A, double *d_B, const size_t bytes,
                  cudaEvent_t *event_start, cudaEvent_t *event_end, float *total_time_ms, const char *case_name)
{
    float time_ms = 0.0;
    CUDA_CHECK(cudaEventRecord(*event_start));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(*event_end));
    CUDA_CHECK(cudaEventSynchronize(*event_end));
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, *event_start, *event_end));
    printf("%s GPU H2D copy time: %.9f seconds\n", case_name, time_ms / 1000);
    *total_time_ms += time_ms;
}

void copy_C_D2H(double *h_C, double *d_C, const size_t bytes,
                cudaEvent_t *event_start, cudaEvent_t *event_end, float *total_time_ms, const char *case_name)
{
    float time_ms = 0.0;
    CUDA_CHECK(cudaEventRecord(*event_start));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(*event_end));
    CUDA_CHECK(cudaEventSynchronize(*event_end));
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, *event_start, *event_end));
    printf("%s GPU D2H copy time: %.9f seconds\n", case_name, time_ms / 1000);
    *total_time_ms += time_ms;
}

void init_C_gpu(double *h_C, double *d_C, const int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_C[i * N + j] = -1.0;
        }
    }

    CUDA_CHECK(cudaMemset(d_C, 0, N * N * sizeof(double)));
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s <matrix size NxN> <check>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int check = atoi(argv[2]);

    printf("Matrix size: %d x %d\n", N, N);

    //
    // Memory allocation
    //
    size_t bytes = N * N * sizeof(double);
    double *h_A = (double *)malloc(bytes);
    double *h_B = (double *)malloc(bytes);
    double *h_C = (double *)malloc(bytes);
    double *C = (double *)malloc(bytes);

    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    //
    // Matrices initialization
    //
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            // Row-major
            h_A[i * N + j] = drand48();
            h_B[i * N + j] = drand48();
            h_C[i * N + j] = -1.0;
            C[i * N + j] = -1.0;
        }
    }

    //
    // Sequential
    //
    if (check)
    {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        matmul_seq(h_A, h_B, C, N);

        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1.0e9;
        printf("Sequential elapsed time: %.9f seconds\n", elapsed);
    }
    else
    {
        printf("Sequential and validation deactivated\n");
    }

    //
    // GPU computations
    //
    cudaEvent_t event_start, event_end;
    float time_ms = 0.0;
    float total_time_ms = 0.0;
    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_end));

    //
    // Naive kernel
    //
    copy_A_B_H2D(h_A, h_B, d_A, d_B, bytes, &event_start, &event_end, &total_time_ms, "Naive");

    // TODO
    // Define threads per block and blocks in the grid
    dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blocksInGrid((N + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);

    CUDA_CHECK(cudaEventRecord(event_start));

    // TODO
    // Launch matmul_naive_kernel
    matmul_naive_kernel<<<blocksInGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(event_end));
    CUDA_CHECK(cudaEventSynchronize(event_end));
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, event_start, event_end));
    printf("Naive GPU kernel time: %.9f seconds\n", time_ms / 1000);
    total_time_ms += time_ms;
    time_ms = 0.0;

    copy_C_D2H(h_C, d_C, bytes, &event_start, &event_end, &total_time_ms, "Naive");

    printf("Naive GPU total time: %.9f seconds\n", total_time_ms / 1000);
    total_time_ms = 0.0;

    if (check)
        validation(h_C, C, N);

    //
    // Shared memory kernel
    //
    init_C_gpu(h_C, d_C, N);
    
    copy_A_B_H2D(h_A, h_B, d_A, d_B, bytes, &event_start, &event_end, &total_time_ms, "Shared");
    

    CUDA_CHECK(cudaEventRecord(event_start));
    // TODO
    // Launch matmul_shared_kernel
    matmul_shared_kernel<<<blocksInGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(event_end));
    CUDA_CHECK(cudaEventSynchronize(event_end));
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, event_start, event_end));
    printf("Shared GPU kernel time: %.9f seconds\n", time_ms / 1000);
    total_time_ms += time_ms;
    time_ms = 0.0;

    copy_C_D2H(h_C, d_C, bytes, &event_start, &event_end, &total_time_ms, "Shared");

    printf("Shared GPU total time: %.9f seconds\n", total_time_ms / 1000);
    total_time_ms = 0.0;

    if (check)
        validation(h_C, C, N);

    //
    // cuBLAS
    //
    init_C_gpu(h_C, d_C, N);
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    copy_A_B_H2D(h_A, h_B, d_A, d_B, bytes, &event_start, &event_end, &total_time_ms, "cuBLAS");

    CUDA_CHECK(cudaEventRecord(event_start));

    const double alpha = 1.0;
    const double beta = 0.0;
    cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B, N,
                d_A, N,
                &beta,
                d_C, N);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(event_end));
    CUDA_CHECK(cudaEventSynchronize(event_end));
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, event_start, event_end));
    printf("cuBLAS GPU kernel time: %.9f seconds\n", time_ms / 1000);
    total_time_ms += time_ms;
    time_ms = 0.0;

    copy_C_D2H(h_C, d_C, bytes, &event_start, &event_end, &total_time_ms, "cuBLAS");

    printf("cuBLAS GPU total time: %.9f seconds\n", total_time_ms / 1000);

    if (check)
        validation(h_C, C, N);

    free(h_A);
    free(h_B);
    free(h_C);
    free(C);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(event_start));
    CUDA_CHECK(cudaEventDestroy(event_end));
    cublasDestroy(cublas_handle);

    return 0;
}