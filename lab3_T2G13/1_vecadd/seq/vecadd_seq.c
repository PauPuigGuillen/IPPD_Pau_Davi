#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// TODO
// Sequential vector addition
void vecadd_seq(double *A, double *B, double *C, const int N)
{
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
    // TODO


    //
    // Vector addition
    //
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    vecadd_seq(A, B, C, N);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1.0e9;
    printf("Elapsed time: %.9f seconds\n", elapsed);

    //
    // Validation
    //
    // TODO
    // Validate vector addition

    //
    // Free memory
    //
    free(A);
    free(B);
    free(C);

    return 0;
}