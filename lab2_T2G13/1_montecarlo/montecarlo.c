#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <stdint.h>

typedef struct { uint64_t state; uint64_t inc; } pcg32_random_t;
double pcg32_random(pcg32_random_t* rng)
{
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    uint32_t ran_int = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    return (double)ran_int / (double)UINT32_MAX;
}



int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int N = 3;
    long NUM_SAMPLES = 1000000;
    long SEED = time(NULL);

    if (argc == 4)
    {
        N = atoi(argv[1]);
        NUM_SAMPLES = strtol(argv[2], NULL, 10);
        SEED = strtol(argv[3], NULL, 10);
    }

    pcg32_random_t rng;
    rng.state = SEED + rank;
    rng.inc = (rank << 16) | 0x3039;

    double ran = pcg32_random(&rng);

    double ratio = 0;
    double err = 0;
    double elapsed_time = 0;
    double start_time = MPI_Wtime();
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        double x = pcg32_random(&ran);
        double y = pcg32_random(&ran);
        double z = pcg32_random(&ran);
        double w = pcg32_random(&ran);
        double d = 0;
        
        if (N >= 1) d += x * x;
        if (N >= 2) d += y * y;
        if (N >= 3) d += z * z;
        if (N >= 4) d += w * w;
        
        if (d <= 1)
        {
            ratio++;
        }
    }
    ratio /= NUM_SAMPLES;
    err = sqrt(ratio * (1 - ratio) / NUM_SAMPLES);
    elapsed_time = MPI_Wtime() - start_time;
    
    double total_ratio = 0.0;
    double total_err = 0.0;
    double total_time = 0.0;
    
    MPI_Reduce(&ratio, &total_ratio, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    MPI_Reduce(&err, &total_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    MPI_Reduce(&elapsed_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        ratio = total_ratio / size;
        err = sqrt(ratio * (1 - ratio) / NUM_SAMPLES);
        elapsed_time = total_time;
        
        printf("Monte Carlo sphere/cube ratio estimation\n");
        printf("N: %ld samples, d: %d, seed %ld, size: %d\n", NUM_SAMPLES, N, SEED, size);
        printf("Ratio = %e (%e) Err: %e\n", ratio, ratio, err);
        printf("Elapsed time: %f seconds\n", elapsed_time);
    }
    MPI_Finalize();
    return 0;
}