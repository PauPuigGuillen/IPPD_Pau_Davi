#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#ifndef M_PI
  #define M_PI 3.141592653589793238462643383279502984
#endif

#define K 32

void initialize(double *v, int N) {
    for (int i = 0; i < N; i++) {
    v[i] = (1 - pow(0.5 - (double)i/(double)N, 2)) * cos(2*M_PI*100* (i - 0.5)/N);
    }
}

// computes the argmax sequentially with a for loop
void argmax_seq(double *v, int N, double *m, int *idx_m) {
    *m = v[0];
    *idx_m = 0;
    for (int i = 1; i < N; i++){
        if (v[i] > *m){
            *m = v[i];
            *idx_m = i;
        }
    }
}

// computes the argmax in parallel with a for loop
void argmax_par(double *v, int N, double *m, int *idx_m) {

    double max_val = v[0];
    int idx = 0;
    #pragma omp parallel for reduction(max:max_val)
    for (int i = 1; i < N; i++){
        if (v[i] > max_val){
            max_val = v[i];
            #pragma omp critical
            {
                idx = i;
            }
        }
    }

    *m = max_val;
    *idx_m = idx;

}

// computes the argmax recursively and sequentially
void argmax_recursive(double *v, int N, double *m, int *idx_m) {

    if (N <= K) {
        *m = v[0];
        *idx_m = 0;
        for (int i = 1; i < N; i++) {
            if (v[i] > *m) {
                *m = v[i];
                *idx_m = i;
            }
        }
        return;
    }

    int mid = N / 2;

    double m1, m2;
    int idx1, idx2;

    argmax_recursive(v, mid, &m1, &idx1);
    argmax_recursive(v + mid, N - mid, &m2, &idx2);

    if (m1 >= m2) {
        *m = m1;
        *idx_m = idx1;
    } else {
        *m = m2;
        *idx_m = idx2 + mid;
    }
    
}

// computes the argmax recursively and in parallel using tasks
void argmax_recursive_tasks(double *v, int N, double *m, int *idx_m) {
    
    if (N <= K) {
        *m = v[0];
        *idx_m = 0;
        for (int i = 1; i < N; i++) {
            if (v[i] > *m) {
                *m = v[i];
                *idx_m = i;
            }
        }
        return;
    }

    int mid = N / 2;

    double m1, m2;
    int idx1, idx2;

    #pragma omp task shared(m1, idx1) if (mid > K)
    argmax_recursive(v, mid, &m1, &idx1);

    #pragma omp task shared(m2, idx2) if (N - mid > K)
    argmax_recursive(v + mid, N - mid, &m2, &idx2);

    #pragma omp taskwait

    if (m1 >= m2) {
        *m = m1;
        *idx_m = idx1;
    } else {
        *m = m2;
        *idx_m = idx2 + mid;
    }

}

int main (int argc, char* argv[]){
    int n;
    if(argc == 2) {
        n = atoi(argv[1]);
    } else {
        printf("Wrong number of paramethers\n");
        return -1;
    }

    double* array = (double *)malloc(n * sizeof(double)); 
    double* max = (double *)malloc(sizeof(double));
    int* idx = (int *)malloc(sizeof(int));
    initialize(array,n);

    printf("\nSequential Argmax\n");
    double start = omp_get_wtime();
    argmax_seq(array, n, max, idx);
    double end = omp_get_wtime();
    printf("Sequential runtime: %f\n", end - start);
    printf("The max is: %f, at index %d\n", *max, *idx);
    
    printf("\nParallel Argmax\n");
    *max = 0.0;
    *idx = 0;
    start = omp_get_wtime();
    argmax_par(array, n, max, idx);
    end = omp_get_wtime();
    printf("Parallel runtime: %f\n", end - start);
    printf("The max is: %f, at index %d\n", *max, *idx);

    printf("\nRecursive Argmax\n");
    *max = 0.0;
    *idx = 0;
    start = omp_get_wtime();
    argmax_recursive(array, n, max, idx);
    end = omp_get_wtime();
    printf("Recursive runtime: %f\n", end - start);
    printf("The max is: %f, at index %d\n", *max, *idx);

    printf("\nRecursive Tasks Argmax\n");
    *max = 0.0;
    *idx = 0;
    start = omp_get_wtime();
    argmax_recursive_tasks(array, n, max, idx);
    end = omp_get_wtime();
    printf("Recursive tasks runtime: %f\n", end - start);
    printf("The max is: %f, at index %d\n", *max, *idx);
    

    return 0;
}