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
    uint32_t xorshifted = ((oldstate >> 18u) ˆ oldstate) >> 27u;
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
    
    int N = 3; //dimension of the circle
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

    MPI_Finalize();
    return 0;
}