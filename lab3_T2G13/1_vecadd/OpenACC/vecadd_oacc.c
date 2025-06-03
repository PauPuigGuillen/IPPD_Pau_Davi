#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <openacc.h>

// OpenACC implementation of vector addition
void vecadd_oacc(double *A, double *B, double *C, const int N)
{
    // Create a data region to explicitly manage memory transfers
    #pragma acc data copyin(A[0:N], B[0:N]) copyout(C[0:N])
    {
        // Parallelize the loop
        #pragma acc parallel loop
        for (int i = 0; i < N; i++)
        {
            C[i] = A[i] + B[i];
        }
    }
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

    //
    // Memory allocation
    //
    double *A = (double *)malloc(N * sizeof(double));
    double *B = (double *)malloc(N * sizeof(double));
    double *C = (double *)malloc(N * sizeof(double));

    //
    // Initialize vectors
    //
    for (int i = 0; i < N; i++)
    {
        A[i] = (double)i;
        B[i] = 2.0 * (N - i);
    }

    //
    // Vector addition
    //
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    vecadd_oacc(A, B, C, N);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1.0e9;
    printf("Elapsed time: %.9f seconds\n", elapsed);

    //
    // Validation
    //
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

    //
    // Free memory
    //
    free(A);
    free(B);
    free(C);

    return 0;
}